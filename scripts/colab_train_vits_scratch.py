#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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


SPECIAL_VOCAB = frozenset({"<pad>", "<unk>", "<bos>", "<eos>"})


def coqui_characters_config(repo_root: Path, tokenizer_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build Coqui `CharactersConfig` from our vocab.json so TTS does not default to ASCII-only graphemes."""
    rel = tokenizer_cfg.get("vocab_path", "artifacts/tokenizer_kikuyu_char/vocab.json")
    extra = (tokenizer_cfg.get("extra_characters") or "").strip()
    path = (repo_root / rel).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    graphemes: set[str] = set()
    for key in raw:
        if key in SPECIAL_VOCAB:
            continue
        if len(key) == 1:
            graphemes.add(key)
    for ch in extra:
        graphemes.add(ch)
    # Coqui appends pad/eos/bos/blank as " " and would duplicate a literal space in `characters`.
    graphemes.discard(" ")
    s = "".join(sorted(graphemes))
    if not s:
        raise ValueError(
            f"No graphemes found in {path} (after special tokens). Build vocab with build_kikuyu_vocab.py first."
        )
    return {
        "characters": s,
        "punctuations": "",
        "pad": " ",
        "eos": " ",
        "bos": " ",
        "blank": " ",
        "is_unique": True,
        "is_sorted": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab helper to launch and resume Kikuyu VITS-from-scratch training.")
    parser.add_argument(
        "--config",
        default="configs/train_kikuyu_vits_scratch_colab.yaml",
        help="Training config file.",
    )
    parser.add_argument(
        "--trainer-repo",
        default=None,
        help="Working directory for the training process. If a Coqui TTS git clone is present "
        "(TTS/bin/train_tts.py), that script is used; otherwise `python -m TTS.bin.train_tts` "
        "is used (e.g. after `pip install coqui-tts`). Defaults to the kikuyu-tts repo root "
        "next to configs/.",
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

    repo_root = config_path.parent.parent.resolve()
    trainer_repo = Path(args.trainer_repo).resolve() if args.trainer_repo else repo_root
    gcn = config["training"]["gradient_clip_norm"]
    grad_clip = gcn if isinstance(gcn, list) else [gcn, gcn]

    cc_chars = coqui_characters_config(repo_root, config.get("tokenizer") or {})
    coqui_config = {
        "run_name": config["experiment"]["name"],
        "output_path": str(local_output_dir),
        "datasets": [
            {
                "formatter": "coqui",
                "dataset_name": "waxal-kikuyu",
                "meta_file_train": config["data"]["train_coqui"],
                "meta_file_val": config["data"]["dev_coqui"],
                "path": str(repo_root),
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
        "grad_clip": grad_clip,
        "text_cleaner": "phoneme_cleaners",
        "use_phonemes": False,
        "compute_input_seq_cache": True,
        "characters": cc_chars,
    }
    resolved_coqui_config = local_output_dir / "coqui_vits_config.yaml"
    write_yaml(resolved_coqui_config, coqui_config)

    if not trainer_repo.exists():
        raise FileNotFoundError(f"Trainer repo not found: {trainer_repo}")
    train_tts_script = trainer_repo / "TTS" / "bin" / "train_tts.py"
    if train_tts_script.is_file():
        train_cmd = [
            sys.executable,
            str(train_tts_script),
            "--config_path",
            str(resolved_coqui_config),
        ]
    else:
        train_cmd = [
            sys.executable,
            "-m",
            "TTS.bin.train_tts",
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
        "trainer_repo": str(trainer_repo),
        "resume_checkpoint": str(resume_ckpt) if resume_ckpt else "",
        "local_output_dir": str(local_output_dir),
        "drive_output_dir": str(drive_output_dir),
    }
    with (local_output_dir / "run_report.json").open("w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)
    print(json.dumps(run_report, indent=2))


if __name__ == "__main__":
    main()
