#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("train", "dev", "test")


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def count_filelist(path: Path) -> int:
    return len(read_lines(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine single-speaker MMS/VITS TTS filelists.")
    parser.add_argument("--anchor-dir", required=True, type=Path, help="Prepared Waxal-style anchor directory.")
    parser.add_argument("--target-dir", required=True, type=Path, help="Prepared ANV-style target directory.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--anchor-train-filelist", default="filelists/train_single_speaker.txt")
    parser.add_argument("--anchor-dev-filelist", default="filelists/dev_single_speaker.txt")
    parser.add_argument("--anchor-test-filelist", default="filelists/test_single_speaker.txt")
    parser.add_argument("--target-train-filelist", default="filelists/train.txt")
    parser.add_argument("--target-dev-filelist", default="filelists/dev.txt")
    parser.add_argument("--target-test-filelist", default="filelists/test.txt")
    parser.add_argument("--anchor-repeat", type=int, default=1)
    parser.add_argument("--target-repeat", type=int, default=1)
    args = parser.parse_args()

    if args.anchor_repeat < 0 or args.target_repeat < 0:
        raise ValueError("repeat counts must be non-negative")
    if args.anchor_repeat == 0 and args.target_repeat == 0:
        raise ValueError("at least one source repeat must be positive")

    anchor_dir = args.anchor_dir.resolve()
    target_dir = args.target_dir.resolve()
    output_dir = args.output_dir.resolve()
    filelists_dir = output_dir / "filelists"

    split_paths = {
        "train": (
            anchor_dir / args.anchor_train_filelist,
            target_dir / args.target_train_filelist,
        ),
        "dev": (
            anchor_dir / args.anchor_dev_filelist,
            target_dir / args.target_dev_filelist,
        ),
        "test": (
            anchor_dir / args.anchor_test_filelist,
            target_dir / args.target_test_filelist,
        ),
    }

    stats: dict[str, object] = {
        "mode": "mixed_single_speaker",
        "anchor_dir": str(anchor_dir),
        "target_dir": str(target_dir),
        "anchor_repeat": args.anchor_repeat,
        "target_repeat": args.target_repeat,
        "sample_rate": 16000,
        "splits": {},
    }
    split_stats: dict[str, dict[str, int]] = {}

    for split, (anchor_path, target_path) in split_paths.items():
        if not anchor_path.is_file():
            raise FileNotFoundError(f"missing anchor {split} filelist: {anchor_path}")
        if not target_path.is_file():
            raise FileNotFoundError(f"missing target {split} filelist: {target_path}")

        anchor_lines = read_lines(anchor_path)
        target_lines = read_lines(target_path)
        combined = anchor_lines * args.anchor_repeat + target_lines * args.target_repeat
        if not combined:
            raise RuntimeError(f"combined {split} filelist is empty")
        write_lines(filelists_dir / f"{split}.txt", combined)
        split_stats[split] = {
            "anchor_rows": len(anchor_lines),
            "target_rows": len(target_lines),
            "combined_rows": len(combined),
        }

    stats["splits"] = split_stats
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "speaker_map.json").open("w", encoding="utf-8") as f:
        json.dump({"mixed_single_speaker": 0}, f, indent=2, ensure_ascii=False)
    with (output_dir / "prep_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)

    for split in SPLITS:
        count_filelist(filelists_dir / f"{split}.txt")


if __name__ == "__main__":
    main()
