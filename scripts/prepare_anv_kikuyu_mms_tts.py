#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
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
DEFAULT_SPLITS = ("train", "dev", "test")
HF_TOKEN_CACHE_PATH = Path("~/.cache/huggingface/token").expanduser()


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
    transcription = str(row.get("transcription") or "").strip()
    transcript = str(row.get("transcript") or "").strip()

    if actual:
        return actual, "scripted"
    if transcription and (row_type != "unscripted" or include_unscripted):
        return transcription, row_type or "transcription"
    if transcript and (row_type != "unscripted" or include_unscripted):
        return transcript, row_type or "transcript"
    return None, row_type or "unknown"


def read_cached_hf_token() -> str | None:
    if not HF_TOKEN_CACHE_PATH.is_file():
        return None
    token = HF_TOKEN_CACHE_PATH.read_text(encoding="utf-8").strip()
    return token or None


def build_hub_auth(cli_token: str | None = None) -> dict[str, str]:
    if cli_token is not None and str(cli_token).strip():
        return {"token": str(cli_token).strip()}
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        raw = os.environ.get(key)
        if raw and raw.strip():
            return {"token": raw.strip()}
    cached_token = read_cached_hf_token()
    if cached_token:
        return {"token": cached_token}
    return {}


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
                "source_split",
                "source_type",
                "dialect",
                "domain",
                "duration_sec",
                "num_samples",
                "sample_rate",
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


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"`{name}` is required on PATH.")


def decode_audio_record(audio: dict[str, Any], target_sample_rate: int) -> tuple[np.ndarray, int]:
    if "array" in audio and audio["array"] is not None:
        samples = np.asarray(audio["array"], dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        return samples, int(audio["sampling_rate"])

    require_command("ffmpeg")
    source_path = audio.get("path")
    suffix = Path(str(source_path or "audio.bin")).suffix or ".bin"
    with tempfile.TemporaryDirectory(prefix="anv-audio-decode-") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / f"input{suffix}"
        output_path = tmp_dir / "output.wav"
        raw_bytes = audio.get("bytes")
        if raw_bytes is not None:
            input_path.write_bytes(raw_bytes)
        elif source_path:
            input_path = Path(source_path)
        else:
            raise ValueError("Audio record has neither decoded array, bytes, nor path.")

        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(target_sample_rate),
                "-f",
                "wav",
                str(output_path),
            ],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode ANV audio payload from {source_path or '<bytes>'}")

        samples, sample_rate = sf.read(str(output_path), always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return samples.astype(np.float32), int(sample_rate)


def summarize_audio_record(audio: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": type(audio).__name__,
        "keys": sorted(str(key) for key in audio.keys()),
        "has_array": audio.get("array") is not None,
        "has_bytes": audio.get("bytes") is not None,
        "has_path": bool(audio.get("path")),
    }
    if audio.get("path"):
        summary["path"] = str(audio.get("path"))
    if audio.get("bytes") is not None:
        summary["bytes_len"] = len(audio["bytes"])
    if audio.get("sampling_rate") is not None:
        summary["sampling_rate"] = int(audio["sampling_rate"])
    return summary


def run_one_row_probe(args: argparse.Namespace) -> None:
    ds = load_dataset(
        args.dataset_name,
        split=args.probe_split,
        streaming=True,
        **build_hub_auth(args.token),
    ).cast_column("audio", Audio(decode=False))

    skip_reasons: defaultdict[str, int] = defaultdict(int)
    for idx, row in enumerate(ds):
        if args.probe_max_rows and idx >= args.probe_max_rows:
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

        result: dict[str, Any] = {
            "event": "one_row_probe_ok",
            "dataset_name": args.dataset_name,
            "split": args.probe_split,
            "idx": idx,
            "text": text,
            "text_len": len(text),
            "source_type": source_type,
            "speaker": safe_name(row.get("recorder_uuid"), "unknown_speaker"),
            "dialect": safe_name(row.get("sentenceDialect"), "unknown_dialect"),
            "domain": str(row.get("domain") or ""),
            "mediaPathId": str(row.get("mediaPathId") or ""),
            "audio": summarize_audio_record(audio),
            "skip_reasons": dict(skip_reasons),
        }

        if args.probe_decode_audio:
            samples, sample_rate = decode_audio_record(audio, args.target_sample_rate)
            result["decoded_audio"] = {
                "sample_rate": sample_rate,
                "num_samples": int(len(samples)),
                "duration_sec": round(float(len(samples) / sample_rate), 4) if sample_rate > 0 else 0.0,
            }

        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return

    raise RuntimeError(
        f"No usable row found in split {args.probe_split!r} after scanning "
        f"{args.probe_max_rows or 'all'} rows. Skips: {dict(skip_reasons)}"
    )


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
    parser.add_argument("--speaker-mode", choices=("all", "dominant_only", "single_speaker"), default="all")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--probe-one-row",
        action="store_true",
        help="Stream one split, validate one audio/text row, print metadata, and exit before preparation.",
    )
    parser.add_argument("--probe-split", default="train")
    parser.add_argument("--probe-max-rows", type=int, default=200)
    parser.add_argument(
        "--probe-decode-audio",
        action="store_true",
        help="Decode the probed row with ffmpeg to validate audio bytes. Omit for the cheapest metadata-only probe.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token for gated datasets. Defaults to HF_TOKEN, HUGGING_FACE_HUB_TOKEN, then ~/.cache/huggingface/token.",
    )
    args = parser.parse_args()

    if args.probe_one_row:
        run_one_row_probe(args)
        os._exit(0)
        return

    output_dir = Path(args.output_dir).resolve()
    clips_dir = output_dir / "clips"
    manifests_dir = output_dir / "manifests"
    filelists_dir = output_dir / "filelists"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    filelists_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": args.train_split,
        "dev": args.dev_split,
        "test": args.test_split,
    }
    dataset: DatasetDict | Any = load_dataset(
        args.dataset_name,
        streaming=args.streaming,
        **build_hub_auth(args.token),
    )
    missing = [source_split for source_split in split_map.values() if source_split not in dataset]
    if missing:
        raise ValueError(f"Missing expected splits in {args.dataset_name}: {missing}")
    print(
        json.dumps(
            {
                "event": "dataset_loaded",
                "dataset_name": args.dataset_name,
                "split_map": split_map,
                "streaming": bool(args.streaming),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    speaker_counts: defaultdict[str, int] = defaultdict(int)
    accepted_rows: list[dict[str, Any]] = []
    skip_reasons: defaultdict[str, int] = defaultdict(int)

    for output_split, source_split in split_map.items():
        print(
            json.dumps(
                {
                    "event": "split_start",
                    "source_split": source_split,
                    "output_split": output_split,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        ds = dataset[source_split].cast_column("audio", Audio(decode=False))
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

            try:
                samples, sample_rate = decode_audio_record(audio, args.target_sample_rate)
            except Exception as exc:
                skip_reasons["audio_decode_failed"] += 1
                print(
                    json.dumps(
                        {
                            "event": "audio_decode_failed",
                            "source_split": source_split,
                            "output_split": output_split,
                            "idx": idx,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
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
            uid = safe_name(row.get("mediaPathId"), f"{output_split}_{idx:07d}")
            utt_id = f"{output_split}_{uid}"
            rel_path = Path("clips") / output_split / speaker / f"{utt_id}.wav"
            abs_path = output_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(abs_path, samples, sample_rate, subtype="PCM_16")

            accepted_rows.append(
                {
                    "utt_id": utt_id,
                    "audio_path": str(abs_path),
                    "text": text,
                    "speaker": speaker,
                    "split": output_split,
                    "source_split": source_split,
                    "source_type": source_type,
                    "dialect": dialect,
                    "domain": str(row.get("domain") or ""),
                    "duration_sec": round(duration_sec, 4),
                    "num_samples": int(len(samples)),
                    "sample_rate": sample_rate,
                }
            )
            written_for_split += 1
            if written_for_split == 1 or written_for_split % 500 == 0:
                print(
                    json.dumps(
                        {
                            "event": "accepted_rows",
                            "source_split": source_split,
                            "output_split": output_split,
                            "count": written_for_split,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        print(
            json.dumps(
                {
                    "event": "split_done",
                    "source_split": source_split,
                    "output_split": output_split,
                    "accepted": written_for_split,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if not accepted_rows:
        raise RuntimeError("No rows survived filtering. Relax filters or enable unscripted rows.")

    dominant_speaker = max(speaker_counts.items(), key=lambda item: (item[1], item[0]))[0]
    if args.speaker_mode == "dominant_only":
        accepted_rows = [row for row in accepted_rows if row["speaker"] == dominant_speaker]
    elif args.speaker_mode == "single_speaker":
        for row in accepted_rows:
            row["speaker"] = "anv_single_speaker"

    speaker_map = {speaker: idx for idx, speaker in enumerate(sorted({row["speaker"] for row in accepted_rows}))}
    for row in accepted_rows:
        row["speaker_id"] = int(speaker_map[row["speaker"]])

    rows_by_split: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted_rows:
        rows_by_split[row["split"]].append(row)

    for split in split_map:
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
        "split_map": split_map,
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
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    if args.streaming:
        os._exit(0)


if __name__ == "__main__":
    main()
