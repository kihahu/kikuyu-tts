#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


def run(command: list[str], cwd: Path | None = None) -> None:
    print(f"+ {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def merge_resume_checkpoint(local_output_dir: Path, drive_output_dir: Path) -> Path | None:
    if not drive_output_dir.exists():
        return None
    checkpoints = sorted(drive_output_dir.glob("checkpoint_*"))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    local_target = local_output_dir / latest.name
    if local_target.exists():
        return local_target
    shutil.copytree(latest, local_target)
    return local_target


def snapshot_to_drive(local_output_dir: Path, drive_output_dir: Path) -> None:
    if not local_output_dir.exists():
        return
    drive_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(local_output_dir.glob("checkpoint_*"))
    for checkpoint in checkpoints:
        target = drive_output_dir / checkpoint.name
        if target.exists():
            continue
        shutil.copytree(checkpoint, target)


def maybe_push_to_hf(local_output_dir: Path, repo_id: str, message: str) -> None:
    if not repo_id:
        return
    run(["huggingface-cli", "upload", repo_id, str(local_output_dir), ".", "--commit-message", message])


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab helper to launch and resume Kikuyu VITS-from-scratch training.")
    parser.add_argument(
        "--config",
        default="configs/train_kikuyu_vits_scratch_colab.yaml",
        help="Training config file.",
    )
    parser.add_argument(
        "--trainer-repo",
        default="/content/TTS",
        help="Path to cloned Coqui TTS repository in Colab runtime.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint found on Drive.")
    parser.add_argument("--push-hf", action="store_true", help="Upload latest run directory to HF Hub.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = read_yaml(config_path)

    ckpt_cfg = config["checkpointing"]
    local_output_dir = Path(ckpt_cfg["local_output_dir"]).resolve()
    drive_output_dir = Path(ckpt_cfg["drive_output_dir"]).resolve()
    local_output_dir.mkdir(parents=True, exist_ok=True)

    coqui_config = {
        "run_name": config["experiment"]["name"],
        "output_path": str(local_output_dir),
        "datasets": [
            {
                "formatter": "ljspeech",
                "dataset_name": "waxal-kikuyu",
                "meta_file_train": config["data"]["train_manifest"],
                "meta_file_val": config["data"]["dev_manifest"],
                "path": ".",
                "language": "kik",
            }
        ],
        "audio": {"sample_rate": config["data"]["sample_rate"]},
        "model": "vits",
        "batch_size": config["training"]["per_device_batch_size"],
        "eval_batch_size": max(1, config["training"]["per_device_batch_size"] // 2),
        "num_loader_workers": config["training"]["num_loader_workers"],
        "epochs": config["training"]["epochs"],
        "save_step": config["training"]["save_every_steps"],
        "print_step": config["training"]["log_every_steps"],
        "eval_step": config["training"]["eval_every_steps"],
        "mixed_precision": config["training"]["precision"] == "fp16",
        "lr": config["training"]["learning_rate"],
        "grad_clip": config["training"]["gradient_clip_norm"],
        "text_cleaner": "phoneme_cleaners",
        "use_phonemes": False,
        "compute_input_seq_cache": True,
    }
    resolved_coqui_config = local_output_dir / "coqui_vits_config.yaml"
    write_yaml(resolved_coqui_config, coqui_config)

    trainer_repo = Path(args.trainer_repo).resolve()
    if not trainer_repo.exists():
        raise FileNotFoundError(f"Trainer repo not found: {trainer_repo}")

    train_cmd = [
        "python",
        "TTS/bin/train_tts.py",
        "--config_path",
        str(resolved_coqui_config),
    ]

    resume_ckpt = None
    if args.resume:
        resume_ckpt = merge_resume_checkpoint(local_output_dir, drive_output_dir)
    if resume_ckpt:
        train_cmd.extend(["--continue_path", str(resume_ckpt)])

    run(train_cmd, cwd=trainer_repo)
    snapshot_to_drive(local_output_dir, drive_output_dir)

    if args.push_hf:
        maybe_push_to_hf(
            local_output_dir=local_output_dir,
            repo_id=ckpt_cfg.get("hf_repo_id", ""),
            message=f"Update checkpoints for {config['experiment']['name']}",
        )

    run_report = {
        "config": str(config_path),
        "resume_checkpoint": str(resume_ckpt) if resume_ckpt else "",
        "local_output_dir": str(local_output_dir),
        "drive_output_dir": str(drive_output_dir),
    }
    with (local_output_dir / "run_report.json").open("w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)
    print(json.dumps(run_report, indent=2))


if __name__ == "__main__":
    main()
