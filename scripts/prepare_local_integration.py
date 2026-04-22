#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare selected checkpoint for local kikuyu-tts inference and future mlx-audio conversion."
    )
    parser.add_argument("--best-checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--eval-summary-csv", required=True)
    parser.add_argument("--out-dir", default="artifacts/local_integration/kikuyu_vits_best")
    parser.add_argument("--model-id", default="kikuyu-vits-scratch-waxal")
    args = parser.parse_args()

    best_ckpt_dir = Path(args.best_checkpoint_dir).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    eval_summary_csv = Path(args.eval_summary_csv).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not best_ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {best_ckpt_dir}")

    model_dir = out_dir / "model"
    tokenizer_out = out_dir / "tokenizer"
    reports_out = out_dir / "reports"

    copy_if_exists(best_ckpt_dir, model_dir)
    copy_if_exists(tokenizer_dir, tokenizer_out)
    copy_if_exists(eval_summary_csv, reports_out / "tts_eval_summary.csv")

    integration_meta = {
        "model_id": args.model_id,
        "format": "vits",
        "sample_rate": 16000,
        "checkpoint_path": str(model_dir),
        "tokenizer_path": str(tokenizer_out),
        "eval_summary_csv": str(reports_out / "tts_eval_summary.csv"),
        "next_steps": [
            "Wire this checkpoint into local kikuyu-tts synthesis script.",
            "Validate sample prompts on local Apple Silicon runtime.",
            "Attempt architecture-aligned conversion path for mlx-audio once VITS support is finalized.",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "integration_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(integration_meta, f, indent=2, ensure_ascii=False)

    print(f"Prepared local integration bundle at {out_dir}")


if __name__ == "__main__":
    main()
