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
    num_speakers = len(speaker_map)

    checkpoint_archive = output_dir / "downloads" / "kik_full_model.tar.gz"
    checkpoint_dir = output_dir / "base_checkpoint" / "kik"
    if args.download_checkpoint:
        download_file(args.checkpoint_url, checkpoint_archive)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(checkpoint_archive, "r:gz") as tar:
            tar.extractall(checkpoint_dir)

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
    patch_nested(patched_config, ["model", "n_speakers"], num_speakers)

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
                "pip install -r requirements.txt",
                'cd monotonic_align && python setup.py build_ext --inplace && cd ..',
                'mkdir -p "logs/$RUN_NAME"',
                'cp "$BASE_CKPT_DIR/G_100000.pth" "logs/$RUN_NAME/G_100000.pth"',
                'cp "$BASE_CKPT_DIR/D_100000.pth" "logs/$RUN_NAME/D_100000.pth"',
                'cp "$BASE_CKPT_DIR/vocab.txt" "logs/$RUN_NAME/vocab.txt"',
                'cp "$CONFIG_PATH" "logs/$RUN_NAME/config.json"',
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
        "num_speakers": num_speakers,
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
