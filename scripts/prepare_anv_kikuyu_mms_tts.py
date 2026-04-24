#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
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
DEFAULT_SPLITS = ("train", "dev", "dev_test")


def normalize_kikuyu_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    for src, dst in PUNCT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.lower().strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def safe_name(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return text or default


def choose_text(row: dict[str, Any], include_unscripted: bool) -> tuple[str | None, str]:
    row_type = str(row.get("type") or "").strip().lower()
    actual = str(row.get("actualSentence") or "").strip()
    transcript = str(row.get("transcript") or "").strip()

    if actual:
        return actual, "scripted"
    if include_unscripted and transcript:
        return transcript, "unscripted"
    return None, row_type or "unknown"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utt_id",
                "audio_path",
                "text",
                "speaker",
                "speaker_id",
                "split",
                "source_type",
                "dialect",
                "domain",
                "duration_sec",
                "num_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_vits_filelist(path: Path, rows: list[dict[str, Any]], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            rel = Path(row["audio_path"]).resolve().relative_to(root.resolve())
            text = str(row["text"]).replace("\n", " ").replace("|", " ")
            speaker_id = int(row["speaker_id"])
            f.write(f"{rel.as_posix()}|{speaker_id}|{text}\n")


def write_manifest_bundle(path_prefix: Path, rows: list[dict[str, Any]], root: Path) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = path_prefix.with_suffix(".tsv")
    txt_path = path_prefix.with_suffix(".txt")
    uid_path = path_prefix.with_suffix(".uid")
    spk_path = path_prefix.with_suffix(".spk")
    lang_path = path_prefix.with_suffix(".lang")

    with (
        tsv_path.open("w", encoding="utf-8") as tsv_f,
        txt_path.open("w", encoding="utf-8") as txt_f,
        uid_path.open("w", encoding="utf-8") as uid_f,
        spk_path.open("w", encoding="utf-8") as spk_f,
        lang_path.open("w", encoding="utf-8") as lang_f,
    ):
        tsv_f.write(f"{root.resolve().as_posix()}\n")
        for row in rows:
            rel = Path(row["audio_path"]).resolve().relative_to(root.resolve())
            tsv_f.write(f"{rel.as_posix()}\t{row['num_samples']}\n")
            txt_f.write(f"{row['text']}\n")
            uid_f.write(f"{row['utt_id']}\n")
            spk_f.write(f"{row['speaker']}\n")
            lang_f.write("kik 1\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Anv-ke/kikuyu for MMS/fairseq TTS fine-tuning with normalized audio and manifests."
    )
    parser.add_argument("--dataset-name", default="Anv-ke/kikuyu")
    parser.add_argument("--output-dir", default="data/anv_kikuyu_mms_tts")
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--max-duration-sec", type=float, default=15.0)
    parser.add_argument("--include-unscripted", action="store_true")
    parser.add_argument("--speaker-mode", choices=("all", "dominant_only"), default="all")
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    clips_dir = output_dir / "clips"
    manifests_dir = output_dir / "manifests"
    filelists_dir = output_dir / "filelists"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    filelists_dir.mkdir(parents=True, exist_ok=True)

    dataset: DatasetDict | Any = load_dataset(args.dataset_name)
    missing = [split for split in DEFAULT_SPLITS if split not in dataset]
    if missing:
        raise ValueError(f"Missing expected splits in {args.dataset_name}: {missing}")

    speaker_counts: defaultdict[str, int] = defaultdict(int)
    accepted_rows: list[dict[str, Any]] = []
    skip_reasons: defaultdict[str, int] = defaultdict(int)

    for split in DEFAULT_SPLITS:
        ds = dataset[split].cast_column("audio", Audio(sampling_rate=args.target_sample_rate))
        written_for_split = 0
        for idx, row in enumerate(ds):
            if args.max_rows_per_split and written_for_split >= args.max_rows_per_split:
                break

            audio = row.get("audio")
            if audio is None:
                skip_reasons["missing_audio"] += 1
                continue

            text_raw, source_type = choose_text(row, include_unscripted=args.include_unscripted)
            if not text_raw:
                skip_reasons["missing_text"] += 1
                continue

            text = normalize_kikuyu_text(text_raw)
            if not text:
                skip_reasons["empty_text_after_normalize"] += 1
                continue

            samples = np.asarray(audio["array"], dtype=np.float32)
            if samples.ndim == 2:
                samples = samples.mean(axis=1)
            sample_rate = int(audio["sampling_rate"])
            duration_sec = float(len(samples) / sample_rate) if sample_rate > 0 else 0.0
            if duration_sec < args.min_duration_sec:
                skip_reasons["too_short"] += 1
                continue
            if duration_sec > args.max_duration_sec:
                skip_reasons["too_long"] += 1
                continue

            speaker = safe_name(row.get("recorder_uuid"), "unknown_speaker")
            speaker_counts[speaker] += 1

            dialect = safe_name(row.get("sentenceDialect"), "unknown_dialect")
            uid = safe_name(row.get("mediaPathId"), f"{split}_{idx:07d}")
            utt_id = f"{split}_{uid}"
            rel_path = Path("clips") / split / speaker / f"{utt_id}.wav"
            abs_path = output_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(abs_path, samples, sample_rate, subtype="PCM_16")

            accepted_rows.append(
                {
                    "utt_id": utt_id,
                    "audio_path": str(abs_path),
                    "text": text,
                    "speaker": speaker,
                    "split": split,
                    "source_type": source_type,
                    "dialect": dialect,
                    "domain": str(row.get("domain") or ""),
                    "duration_sec": round(duration_sec, 4),
                    "num_samples": int(len(samples)),
                    "sample_rate": sample_rate,
                }
            )
            written_for_split += 1

    if not accepted_rows:
        raise RuntimeError("No rows survived filtering. Relax filters or enable unscripted rows.")

    dominant_speaker = max(speaker_counts.items(), key=lambda item: (item[1], item[0]))[0]
    if args.speaker_mode == "dominant_only":
        accepted_rows = [row for row in accepted_rows if row["speaker"] == dominant_speaker]

    speaker_map = {speaker: idx for idx, speaker in enumerate(sorted({row["speaker"] for row in accepted_rows}))}
    for row in accepted_rows:
        row["speaker_id"] = int(speaker_map[row["speaker"]])

    rows_by_split: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted_rows:
        rows_by_split[row["split"]].append(row)

    for split in DEFAULT_SPLITS:
        rows = rows_by_split.get(split, [])
        write_jsonl(manifests_dir / f"{split}.jsonl", rows)
        write_manifest_bundle(manifests_dir / split, rows, output_dir)
        write_vits_filelist(filelists_dir / f"{split}.txt", rows, output_dir)

    write_summary_csv(output_dir / "all_rows.csv", accepted_rows)
    with (output_dir / "speaker_map.json").open("w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2, ensure_ascii=False)
    stats = {
        "dataset_name": args.dataset_name,
        "splits": list(DEFAULT_SPLITS),
        "include_unscripted": bool(args.include_unscripted),
        "speaker_mode": args.speaker_mode,
        "sample_rate": args.target_sample_rate,
        "dominant_speaker": dominant_speaker,
        "total_rows": len(accepted_rows),
        "rows_per_split": {split: len(rows_by_split.get(split, [])) for split in DEFAULT_SPLITS},
        "unique_speakers": len({row["speaker"] for row in accepted_rows}),
        "speaker_map_path": str(output_dir / "speaker_map.json"),
        "skip_reasons": dict(skip_reasons),
        "outputs": {
            "manifests_dir": str(manifests_dir),
            "filelists_dir": str(filelists_dir),
            "clips_dir": str(clips_dir),
        },
    }
    with (output_dir / "prep_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
