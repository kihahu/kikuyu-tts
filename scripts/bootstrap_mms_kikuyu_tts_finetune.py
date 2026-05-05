#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import yaml


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


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


def safe_extract_tar_gz(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Refusing to extract unsafe tar member: {member.name}")
        tar.extractall(root)


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def read_checkpoint_iteration(path: Path) -> int:
    try:
        import torch
    except ImportError:
        return 0

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return 0
    for key in ("iteration", "epoch", "global_step"):
        value = checkpoint.get(key)
        if isinstance(value, int):
            return value
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap MMS Kikuyu TTS fine-tuning with a prepared VITS filelist dataset."
    )
    parser.add_argument("--config", default="configs/train_mms_tts_kik_waxal.yaml")
    parser.add_argument("--prepared-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--checkpoint-url", default="")
    parser.add_argument("--train-filelist", default="")
    parser.add_argument("--dev-filelist", default="")
    parser.add_argument("--speaker-map", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--download-checkpoint", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(resolve_repo_path(repo_root, args.config))
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    prepared_dir = resolve_repo_path(
        repo_root,
        args.prepared_dir or dataset_cfg.get("prepared_dir", "data/waxal_kik_tts"),
    ).resolve()
    output_dir = resolve_repo_path(
        repo_root,
        args.output_dir or training_cfg.get("output_dir", "artifacts/mms_tts_kik_waxal_finetune"),
    ).resolve()
    run_name = args.run_name or training_cfg.get("run_name", "mms_kik_waxal_single_speaker")
    checkpoint_url = args.checkpoint_url or model_cfg.get(
        "full_checkpoint_url", "https://dl.fbaipublicfiles.com/mms/tts/full_model/kik.tar.gz"
    )

    train_filelist = prepared_dir / (args.train_filelist or dataset_cfg.get("train_filelist", "filelists/train.txt"))
    dev_filelist = prepared_dir / (args.dev_filelist or dataset_cfg.get("dev_filelist", "filelists/dev.txt"))
    speaker_map_path = prepared_dir / (args.speaker_map or dataset_cfg.get("speaker_map", "speaker_map.json"))
    prep_stats_path = prepared_dir / dataset_cfg.get("prep_stats", "prep_stats.json")

    for path, label in (
        (train_filelist, "train VITS filelist"),
        (dev_filelist, "dev VITS filelist"),
        (speaker_map_path, "speaker map"),
        (prep_stats_path, "prep stats"),
    ):
        ensure_file(path, label)

    speaker_map = load_json(speaker_map_path)
    prep_stats = load_json(prep_stats_path)
    dataset_speakers = len(speaker_map)
    vits_n_speakers = 0 if dataset_speakers <= 1 else dataset_speakers
    sample_rate = int(prep_stats.get("sample_rate") or prep_stats.get("config", {}).get("target_sample_rate") or 16000)

    checkpoint_archive = output_dir / "downloads" / "kik_full_model.tar.gz"
    checkpoint_parent = output_dir / "base_checkpoint"
    checkpoint_dir = checkpoint_parent / "kik"
    if args.download_checkpoint:
        download_file(checkpoint_url, checkpoint_archive)
        safe_extract_tar_gz(checkpoint_archive, checkpoint_parent)

    ensure_file(checkpoint_dir / "G_100000.pth", "MMS generator checkpoint")
    ensure_file(checkpoint_dir / "D_100000.pth", "MMS discriminator checkpoint")
    ensure_file(checkpoint_dir / "config.json", "MMS config")
    ensure_file(checkpoint_dir / "vocab.txt", "MMS vocab")

    base_iteration = read_checkpoint_iteration(checkpoint_dir / "G_100000.pth")
    requested_epochs = int(args.epochs or training_cfg.get("epochs", 10000))
    effective_epochs = requested_epochs
    if args.epochs and base_iteration and requested_epochs <= base_iteration:
        effective_epochs = base_iteration + requested_epochs

    base_config = load_json(checkpoint_dir / "config.json")
    patched_config = json.loads(json.dumps(base_config))
    patch_nested(patched_config, ["data", "training_files"], str(train_filelist))
    patch_nested(patched_config, ["data", "validation_files"], str(dev_filelist))
    patch_nested(patched_config, ["data", "text_cleaners"], ["basic_cleaners"])
    patch_nested(patched_config, ["data", "sampling_rate"], sample_rate)
    patch_nested(patched_config, ["train", "batch_size"], int(args.batch_size or training_cfg.get("batch_size", 16)))
    patch_nested(patched_config, ["train", "fp16_run"], bool(training_cfg.get("fp16_run", True)))
    patch_nested(patched_config, ["train", "learning_rate"], float(training_cfg.get("learning_rate", 0.00005)))
    patch_nested(patched_config, ["train", "epochs"], effective_epochs)
    patch_nested(patched_config, ["train", "eval_interval"], int(args.eval_interval or training_cfg.get("eval_interval", 200)))
    patch_nested(patched_config, ["train", "log_interval"], int(args.log_interval or training_cfg.get("log_interval", 50)))
    patch_nested(patched_config, ["train", "seed"], int(training_cfg.get("seed", 42)))
    patch_nested(patched_config, ["data", "n_speakers"], vits_n_speakers)

    config_out = output_dir / "generated_config" / f"{run_name}.json"
    write_json(config_out, patched_config)

    launch_script = output_dir / "launch_finetune.sh"
    launch_script.parent.mkdir(parents=True, exist_ok=True)
    launch_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                f'VITS_REPO="${{VITS_REPO:-{(output_dir / "vits").as_posix()}}}"',
                f'RUN_NAME="${{RUN_NAME:-{run_name}}}"',
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
                "mel_path = Path('mel_processing.py')",
                "mel_text = mel_path.read_text(encoding='utf-8')",
                "mel_text = mel_text.replace(",
                "    \"center=center, pad_mode='reflect', normalized=False, onesided=True)\",",
                "    \"center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)\",",
                ")",
                "mel_text = mel_text.replace(",
                "    \"mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)\",",
                "    \"mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)\",",
                ")",
                "mel_path.write_text(mel_text, encoding='utf-8')",
                "utils_path = Path('utils.py')",
                "utils_text = utils_path.read_text(encoding='utf-8')",
                "utils_text = utils_text.replace(",
                "    \"data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')\\n  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))\",",
                "    \"data = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()\",",
                ")",
                "utils_path.write_text(utils_text, encoding='utf-8')",
                "setup_path = Path('monotonic_align/setup.py')",
                "setup_path.write_text(",
                "    \"from distutils.core import setup\\n\"",
                "    \"from Cython.Build import cythonize\\n\"",
                "    \"import numpy\\n\\n\"",
                "    \"setup(\\n\"",
                "    \"  name='monotonic_align',\\n\"",
                "    \"  ext_modules=cythonize('core.pyx'),\\n\"",
                "    \"  include_dirs=[numpy.get_include()]\\n\"",
                "    \")\\n\",",
                "    encoding='utf-8',",
                ")",
                "nested_align = Path('monotonic_align/monotonic_align')",
                "nested_align.mkdir(parents=True, exist_ok=True)",
                "(nested_align / '__init__.py').write_text('', encoding='utf-8')",
                "PY",
                "pip install Cython 'librosa>=0.10.1' matplotlib phonemizer scipy tensorboard Unidecode",
                "(cd monotonic_align && python setup.py build_ext --inplace)",
                'mkdir -p "logs/$RUN_NAME"',
                'cp -n "$BASE_CKPT_DIR/G_100000.pth" "logs/$RUN_NAME/G_100000.pth"',
                'cp -n "$BASE_CKPT_DIR/D_100000.pth" "logs/$RUN_NAME/D_100000.pth"',
                'cp -n "$BASE_CKPT_DIR/vocab.txt" "logs/$RUN_NAME/vocab.txt"',
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
        "train_filelist": str(train_filelist),
        "dev_filelist": str(dev_filelist),
        "checkpoint_dir": str(checkpoint_dir),
        "generated_config": str(config_out),
        "launch_script": str(launch_script),
        "dataset_speakers": dataset_speakers,
        "vits_n_speakers": vits_n_speakers,
        "sample_rate": sample_rate,
        "run_name": run_name,
        "base_iteration": base_iteration,
        "requested_epochs": requested_epochs,
        "effective_epochs": effective_epochs,
        "hub_repo_id": cfg.get("hub", {}).get("repo_id", ""),
        "notes": [
            "This is the real MMS/VITS continuation path; Transformers VitsModel remains inference-only.",
            "For Waxal v1, use the single-speaker filelists by default for a cleaner voice.",
            "Run the generated launch script on a GPU machine or through the HF Jobs wrapper.",
        ],
    }
    write_json(output_dir / "bootstrap_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
