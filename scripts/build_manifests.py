#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from normalize_kikuyu import normalize_kikuyu_text


def load_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = row.get("audio_path", "").strip()
            text = row.get("text", "").strip()
            speaker = row.get("speaker", "").strip() or "unknown_speaker"
            duration = float(row.get("duration", "0") or 0)
            if not audio_path or not text:
                continue
            rows.append(
                {
                    "audio_path": audio_path,
                    "text": normalize_kikuyu_text(text),
                    "speaker": speaker,
                    "duration": duration,
                }
            )
    return rows


def split_by_speaker(rows: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    speakers = sorted({r["speaker"] for r in rows})
    random.Random(seed).shuffle(speakers)
    n = len(speakers)
    n_train = int(n * 0.85)
    n_dev = int(n * 0.10)
    train_s = set(speakers[:n_train])
    dev_s = set(speakers[n_train : n_train + n_dev])
    test_s = set(speakers[n_train + n_dev :])

    train = [r for r in rows if r["speaker"] in train_s]
    dev = [r for r in rows if r["speaker"] in dev_s]
    test = [r for r in rows if r["speaker"] in test_s]
    return train, dev, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/dev/test JSONL manifests for Kikuyu TTS.")
    parser.add_argument("--input-csv", required=True, help="CSV with columns: audio_path,text,speaker,duration")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(Path(args.input_csv))
    train, dev, test = split_by_speaker(rows, seed=args.seed)
    out = Path(args.output_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    write_jsonl(out / "test.jsonl", test)
    print(f"train={len(train)} dev={len(dev)} test={len(test)}")


if __name__ == "__main__":
    main()
