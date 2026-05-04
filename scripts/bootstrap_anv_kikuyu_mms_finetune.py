#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as f:
        shutil.copyfileobj(response, f)


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def patch_nested(config: dict[str, Any], path: list[str], value: Any) -> None:
    cur = config
    for key in path[:-1]:
        next_value = cur.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cur[key] = next_value
        cur = next_value
    cur[path[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap a real MMS Kikuyu fine-tuning run using the full checkpoint and prepared ANV manifests."
    )
    parser.add_argument("--prepared-dir", default="data/anv_kikuyu_mms_tts")
    parser.add_argument("--checkpoint-url", default="https://dl.fbaipublicfiles.com/mms/tts/full_model/kik.tar.gz")
    parser.add_argument("--output-dir", default="artifacts/anv_kikuyu_mms_finetune")
    parser.add_argument("--run-name", default="mms_kik_anv_ms")
    parser.add_argument("--download-checkpoint", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    prepared_dir = (repo_root / args.prepared_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    manifests_dir = prepared_dir / "manifests"
    train_filelist = prepared_dir / "filelists" / "train.txt"
    dev_filelist = prepared_dir / "filelists" / "dev.txt"
    speaker_map_path = prepared_dir / "speaker_map.json"
    prep_stats_path = prepared_dir / "prep_stats.json"
    for path, label in (
        (train_filelist, "train filelist"),
        (dev_filelist, "dev filelist"),
        (speaker_map_path, "speaker map"),
        (prep_stats_path, "prep stats"),
        (manifests_dir / "train.tsv", "train manifest bundle"),
        (manifests_dir / "dev.tsv", "dev manifest bundle"),
    ):
        ensure_file(path, label)

    speaker_map = load_json(speaker_map_path)
    prep_stats = load_json(prep_stats_path)
    dataset_speakers = len(speaker_map)
    vits_n_speakers = 0 if dataset_speakers <= 1 else dataset_speakers

    checkpoint_archive = output_dir / "downloads" / "kik_full_model.tar.gz"
    checkpoint_parent = output_dir / "base_checkpoint"
    checkpoint_dir = checkpoint_parent / "kik"
    if args.download_checkpoint:
        download_file(args.checkpoint_url, checkpoint_archive)
        checkpoint_parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(checkpoint_archive, "r:gz") as tar:
            tar.extractall(checkpoint_parent)

    ensure_file(checkpoint_dir / "G_100000.pth", "MMS generator checkpoint")
    ensure_file(checkpoint_dir / "D_100000.pth", "MMS discriminator checkpoint")
    ensure_file(checkpoint_dir / "config.json", "MMS config")
    ensure_file(checkpoint_dir / "vocab.txt", "MMS vocab")

    base_config = load_json(checkpoint_dir / "config.json")
    patched_config = json.loads(json.dumps(base_config))
    patch_nested(patched_config, ["data", "training_files"], str(train_filelist))
    patch_nested(patched_config, ["data", "validation_files"], str(dev_filelist))
    patch_nested(patched_config, ["data", "text_cleaners"], ["basic_cleaners"])
    patch_nested(patched_config, ["data", "sampling_rate"], int(prep_stats.get("sample_rate", 16000)))
    patch_nested(patched_config, ["train", "batch_size"], 16)
    patch_nested(patched_config, ["train", "fp16_run"], True)
    patch_nested(patched_config, ["train", "learning_rate"], 0.00005)
    patch_nested(patched_config, ["train", "epochs"], 10000)
    patch_nested(patched_config, ["data", "n_speakers"], vits_n_speakers)

    config_out = output_dir / "generated_config" / f"{args.run_name}.json"
    write_json(config_out, patched_config)

    launch_script = output_dir / "launch_finetune.sh"
    launch_script.parent.mkdir(parents=True, exist_ok=True)
    launch_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                'VITS_REPO="${VITS_REPO:-/content/vits}"',
                f'RUN_NAME="${{RUN_NAME:-{args.run_name}}}"',
                f'BOOTSTRAP_DIR="{output_dir.as_posix()}"',
                f'CONFIG_PATH="{config_out.as_posix()}"',
                f'BASE_CKPT_DIR="{checkpoint_dir.as_posix()}"',
                "",
                'if [ ! -d "$VITS_REPO" ]; then',
                '  git clone https://github.com/jaywalnut310/vits.git "$VITS_REPO"',
                "fi",
                'cd "$VITS_REPO"',
                'python - <<\'PY\'',
                "from pathlib import Path",
                "path = Path('train_ms.py')",
                "text = path.read_text(encoding='utf-8')",
                "text = text.replace(\"os.environ['MASTER_PORT'] = '80000'\", \"os.environ['MASTER_PORT'] = '29500'\")",
                "path.write_text(text, encoding='utf-8')",
                "symbols_path = Path('text/symbols.py')",
                "symbols_path.write_text(",
                "    \"import os\\n\"",
                "    \"vocab_file = os.environ.get('MMS_VOCAB_FILE')\\n\"",
                "    \"if vocab_file:\\n\"",
                "    \"    with open(vocab_file, encoding='utf-8') as f:\\n\"",
                "    \"        symbols = [line.rstrip('\\\\n') for line in f]\\n\"",
                "    \"else:\\n\"",
                "    \"    _pad = '_'\\n\"",
                "    \"    _punctuation = ';:,.!?¡¿—…\\\\\\\"«»“” '\\n\"",
                "    \"    _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'\\n\"",
                "    \"    _letters_ipa = \\\"ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ\\\"\\n\"",
                "    \"    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)\\n\"",
                "    \"SPACE_ID = symbols.index(' ') if ' ' in symbols else 0\\n\",",
                "    encoding='utf-8',",
                ")",
                "text_init_path = Path('text/__init__.py')",
                "text_init = text_init_path.read_text(encoding='utf-8')",
                "text_init = text_init.replace('symbol_id = _symbol_to_id[symbol]\\n    sequence += [symbol_id]', 'symbol_id = _symbol_to_id.get(symbol)\\n    if symbol_id is not None:\\n      sequence += [symbol_id]')",
                "text_init = text_init.replace('sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]', 'sequence = [_symbol_to_id[symbol] for symbol in cleaned_text if symbol in _symbol_to_id]')",
                "text_init_path.write_text(text_init, encoding='utf-8')",
                "setup_path = Path('monotonic_align/setup.py')",
                "setup_path.write_text(",
                "    \"from distutils.core import setup\\n\"",
                "    \"from Cython.Build import cythonize\\n\"",
                "    \"import numpy\\n\\n\"",
                "    \"setup(\\n\"",
                "    \"  name='monotonic_align',\\n\"",
                "    \"  ext_modules=cythonize('monotonic_align/core.pyx'),\\n\"",
                "    \"  include_dirs=[numpy.get_include()]\\n\"",
                "    \")\\n\",",
                "    encoding='utf-8',",
                ")",
                "PY",
                "pip install Cython 'librosa>=0.10.1' matplotlib phonemizer scipy tensorboard Unidecode",
                "python monotonic_align/setup.py build_ext --inplace",
                'mkdir -p "logs/$RUN_NAME"',
                'cp "$BASE_CKPT_DIR/G_100000.pth" "logs/$RUN_NAME/G_100000.pth"',
                'cp "$BASE_CKPT_DIR/D_100000.pth" "logs/$RUN_NAME/D_100000.pth"',
                'cp "$BASE_CKPT_DIR/vocab.txt" "logs/$RUN_NAME/vocab.txt"',
                'cp "$CONFIG_PATH" "logs/$RUN_NAME/config.json"',
                'export MMS_VOCAB_FILE="$BASE_CKPT_DIR/vocab.txt"',
                'python train_ms.py -c "$CONFIG_PATH" -m "$RUN_NAME"',
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    launch_script.chmod(0o755)

    summary = {
        "prepared_dir": str(prepared_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "generated_config": str(config_out),
        "launch_script": str(launch_script),
        "dataset_speakers": dataset_speakers,
        "vits_n_speakers": vits_n_speakers,
        "run_name": args.run_name,
        "notes": [
            "This bootstraps fine-tuning with the official VITS training code and the full MMS Kikuyu checkpoint.",
            "The launch script clones jaywalnut310/vits if VITS_REPO does not already exist.",
            "The prepared manifests remain speaker-preserving; use dominant_only in prep if multi-speaker convergence is unstable.",
        ],
    }
    write_json(output_dir / "bootstrap_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
