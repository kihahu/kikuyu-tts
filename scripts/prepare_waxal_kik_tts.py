#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, DatasetDict, load_dataset


MULTISPACE_RE = re.compile(r"\s+")
PUNCT_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
}


@dataclass(frozen=True)
class PrepConfig:
    dataset_name: str
    dataset_config: str
    split: str
    target_sample_rate: int
    min_duration_sec: float
    max_duration_sec: float
    min_rms: float
    seed: int
    dev_ratio: float
    test_ratio: float


def normalize_kikuyu_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    for src, dst in PUNCT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.lower().strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def pick_first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            return row[key]
    return None


def safe_speaker_id(raw_value: Any) -> str:
    if raw_value is None:
        return "unknown_speaker"
    speaker = str(raw_value).strip()
    if not speaker:
        return "unknown_speaker"
    speaker = re.sub(r"[^a-zA-Z0-9_-]+", "_", speaker)
    return speaker or "unknown_speaker"


def calc_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    return float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))


def speaker_disjoint_split(
    rows: list[dict[str, Any]],
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    speakers = sorted({row["speaker"] for row in rows})
    if len(speakers) < 3:
        raise ValueError("Need at least 3 speakers for speaker-disjoint train/dev/test split.")

    rng = random.Random(seed)
    rng.shuffle(speakers)

    n_total = len(speakers)
    n_test = max(1, int(round(n_total * test_ratio)))
    n_dev = max(1, int(round(n_total * dev_ratio)))
    n_train = n_total - n_dev - n_test
    if n_train < 1:
        raise ValueError("Split ratios leave no speakers for train split. Adjust dev/test ratios.")

    train_speakers = set(speakers[:n_train])
    dev_speakers = set(speakers[n_train : n_train + n_dev])
    test_speakers = set(speakers[n_train + n_dev :])

    train_rows = [row for row in rows if row["speaker"] in train_speakers]
    dev_rows = [row for row in rows if row["speaker"] in dev_speakers]
    test_rows = [row for row in rows if row["speaker"] in test_speakers]

    if not train_rows or not dev_rows or not test_rows:
        raise ValueError("Split resulted in an empty subset. Adjust ratios or filters.")

    return train_rows, dev_rows, test_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def write_coqui_pipe_manifest(path: Path, rows: list[dict[str, Any]], repo_root: Path) -> None:
    """Coqui TTS 'coqui' formatter: header audio_file|text|speaker_name, paths rel. to repo root."""
    path.parent.mkdir(parents=True, exist_ok=True)
    root = repo_root.resolve()
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("audio_file|text|speaker_name\n")
        for row in rows:
            ap = Path(row["audio_path"]).resolve()
            rel = ap.relative_to(root)
            t = str(row["text"]).replace("\n", " ").replace("|", " ")
            spk = str(row["speaker"])
            f.write(f"{rel.as_posix()}|{t}|{spk}\n")


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "text", "speaker", "duration_sec", "sample_rate", "rms"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare WaxalNLP kik_tts for VITS training with speaker-disjoint manifests."
    )
    parser.add_argument("--dataset-name", default="google/WaxalNLP")
    parser.add_argument("--dataset-config", default="kik_tts")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/waxal_kik_tts")
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--min-duration-sec", type=float, default=0.6)
    parser.add_argument("--max-duration-sec", type=float, default=25.0)
    parser.add_argument("--min-rms", type=float, default=0.0035)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    args = parser.parse_args()

    cfg = PrepConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        target_sample_rate=args.target_sample_rate,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        min_rms=args.min_rms,
        seed=args.seed,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
    )

    output_dir = Path(args.output_dir).resolve()
    clips_dir = output_dir / "clips"
    manifests_dir = output_dir / "manifests"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    dataset: DatasetDict | Any = load_dataset(cfg.dataset_name, cfg.dataset_config)
    if cfg.split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split '{cfg.split}' not found. Available splits: {available}")

    split_ds = dataset[cfg.split].cast_column("audio", Audio(sampling_rate=cfg.target_sample_rate))
    rows: list[dict[str, Any]] = []
    skip_reasons: defaultdict[str, int] = defaultdict(int)

    for idx, row in enumerate(split_ds):
        audio = row.get("audio")
        text_raw = pick_first(row, ("normalized_text", "sentence", "text", "transcript"))
        speaker_raw = pick_first(row, ("speaker_id", "speaker", "client_id", "voice_id"))

        if audio is None:
            skip_reasons["missing_audio"] += 1
            continue
        if text_raw is None:
            skip_reasons["missing_text"] += 1
            continue

        samples = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio["sampling_rate"])
        duration_sec = float(len(samples) / sample_rate) if sample_rate > 0 else 0.0
        if duration_sec < cfg.min_duration_sec:
            skip_reasons["too_short"] += 1
            continue
        if duration_sec > cfg.max_duration_sec:
            skip_reasons["too_long"] += 1
            continue

        rms = calc_rms(samples)
        if rms < cfg.min_rms:
            skip_reasons["too_silent"] += 1
            continue

        text = normalize_kikuyu_text(str(text_raw))
        if not text:
            skip_reasons["empty_text_after_normalize"] += 1
            continue

        speaker = safe_speaker_id(speaker_raw)
        rel_path = Path("clips") / f"{speaker}_{idx:07d}.wav"
        abs_path = output_dir / rel_path
        sf.write(abs_path, samples, sample_rate, subtype="PCM_16")

        rows.append(
            {
                "audio_path": str(abs_path),
                "text": text,
                "speaker": speaker,
                "duration_sec": round(duration_sec, 4),
                "sample_rate": sample_rate,
                "rms": round(rms, 6),
            }
        )

    if not rows:
        raise RuntimeError("No rows survived filtering. Relax duration/RMS thresholds and retry.")

    train_rows, dev_rows, test_rows = speaker_disjoint_split(
        rows=rows,
        dev_ratio=cfg.dev_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    write_manifest_csv(manifests_dir / "all.csv", rows)
    write_jsonl(manifests_dir / "train.jsonl", train_rows)
    write_jsonl(manifests_dir / "dev.jsonl", dev_rows)
    write_jsonl(manifests_dir / "test.jsonl", test_rows)
    repo_root = output_dir.parent.parent
    write_coqui_pipe_manifest(manifests_dir / "train_coqui.txt", train_rows, repo_root)
    write_coqui_pipe_manifest(manifests_dir / "dev_coqui.txt", dev_rows, repo_root)

    stats = {
        "config": cfg.__dict__,
        "total_rows_after_filter": len(rows),
        "split_sizes": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "test": len(test_rows),
        },
        "unique_speakers": len({row["speaker"] for row in rows}),
        "skip_reasons": dict(skip_reasons),
    }
    write_stats(output_dir / "prep_stats.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
