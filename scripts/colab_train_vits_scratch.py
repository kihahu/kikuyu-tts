#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

MULTISPACE_RE = re.compile(r"\s+")

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
BLANK_TOKEN = "<BLNK>"


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


def normalize_for_training(text: str) -> str:
    return MULTISPACE_RE.sub(" ", (text or "").strip().lower())


def load_vocab_tokens(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected {path} to contain a JSON object mapping token->id.")
    try:
        ordered = sorted(raw.items(), key=lambda item: int(item[1]))
    except Exception as exc:  # pragma: no cover - defensive config validation
        raise ValueError(f"Expected integer token ids in {path}.") from exc
    return [token for token, _ in ordered]


def coqui_characters_config(repo_root: Path, tokenizer_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a VITS-native CharactersConfig from our vocab.json."""
    rel = tokenizer_cfg.get("vocab_path", "artifacts/tokenizer_kikuyu_char/vocab.json")
    extra = (tokenizer_cfg.get("extra_characters") or "").strip()
    path = (repo_root / rel).resolve()
    graphemes: set[str] = set()
    for token in load_vocab_tokens(path):
        if token in SPECIAL_VOCAB:
            continue
        if len(token) == 1:
            graphemes.add(token)
    for ch in extra:
        graphemes.add(ch)
    if not graphemes:
        raise ValueError(
            f"No graphemes found in {path} (after special tokens). Build vocab with build_kikuyu_vocab.py first."
        )
    punctuations = "".join(sorted(ch for ch in graphemes if ch.isspace() or not ch.isalnum()))
    characters = "".join(sorted(ch for ch in graphemes if ch not in punctuations))
    return {
        "characters_class": "TTS.tts.models.vits.VitsCharacters",
        "characters": characters,
        "punctuations": punctuations,
        "phonemes": "",
        "pad": PAD_TOKEN,
        "eos": EOS_TOKEN,
        "bos": BOS_TOKEN,
        "blank": BLANK_TOKEN,
        "is_unique": True,
        "is_sorted": True,
    }


def validate_manifest_characters(
    repo_root: Path,
    manifest_paths: list[str],
    allowed_chars: set[str],
) -> None:
    unsupported: dict[str, list[str]] = {}
    for rel in manifest_paths:
        path = (repo_root / rel).resolve()
        with path.open("r", encoding="utf-8") as f:
            next(f, None)  # header
            for idx, line in enumerate(f, start=2):
                parts = line.rstrip("\n").split("|")
                if len(parts) < 2:
                    continue
                text = normalize_for_training(parts[1])
                missing = sorted({ch for ch in text if ch not in allowed_chars})
                if missing:
                    unsupported[f"{path}:{idx}"] = missing
                    if len(unsupported) >= 5:
                        break
        if len(unsupported) >= 5:
            break
    if unsupported:
        preview = "; ".join(f"{loc} -> {''.join(chars)}" for loc, chars in unsupported.items())
        raise ValueError(
            "Tokenizer vocab does not cover the training text used by Coqui. "
            f"Rebuild the vocab and rerun training. First mismatches: {preview}"
        )


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

    tokenizer_cfg = config.get("tokenizer") or {}
    cc_chars = coqui_characters_config(repo_root, tokenizer_cfg)
    allowed_chars = set(cc_chars["characters"]) | set(cc_chars["punctuations"])
    validate_manifest_characters(
        repo_root,
        [config["data"]["train_coqui"], config["data"]["dev_coqui"]],
        allowed_chars,
    )
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
        "text_cleaner": "basic_cleaners",
        "use_phonemes": False,
        "phoneme_language": None,
        "add_blank": bool(tokenizer_cfg.get("add_blank_token", True)),
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
