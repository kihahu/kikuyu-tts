#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC


DEFAULT_MODEL_ID = "kihahu/mms-asr-kik-waxal-ctc"
DEFAULT_TARGET_LANG = "kik"
TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Caption:
    index: int
    start_seconds: float
    end_seconds: float
    text: str


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"`{name}` is required on PATH.")


def run(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def download_youtube_audio(url: str, work_dir: Path) -> Path:
    require_command("yt-dlp")
    output_template = str(work_dir / "youtube_audio.%(ext)s")
    run(
        [
            "yt-dlp",
            "--no-playlist",
            "-f",
            "bestaudio/best",
            "-o",
            output_template,
            url,
        ]
    )
    candidates = sorted(work_dir.glob("youtube_audio.*"))
    if not candidates:
        raise RuntimeError("yt-dlp finished but no audio file was created.")
    return candidates[0]


def convert_to_wav_16k_mono(input_path: Path, output_path: Path) -> None:
    require_command("ffmpeg")
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-f",
            "wav",
            str(output_path),
        ]
    )


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sample_rate)


def format_vtt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d}.{ms:03d}"


def chunk_ranges(
    *,
    total_samples: int,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[int, int]]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive.")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be >= 0 and < chunk_seconds.")

    chunk_samples = max(1, int(round(chunk_seconds * sample_rate)))
    stride_samples = max(1, int(round((chunk_seconds - overlap_seconds) * sample_rate)))
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_samples:
        end = min(total_samples, start + chunk_samples)
        ranges.append((start, end))
        if end == total_samples:
            break
        start += stride_samples
    return ranges


def transcribe_audio_chunks(
    *,
    audio: np.ndarray,
    sample_rate: int,
    processor: AutoProcessor,
    model: Wav2Vec2ForCTC,
    device: str,
    chunk_seconds: float,
    overlap_seconds: float,
    min_rms: float,
) -> list[Caption]:
    captions: list[Caption] = []
    ranges = chunk_ranges(
        total_samples=len(audio),
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    cue_index = 1
    for start, end in ranges:
        chunk = audio[start:end]
        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64))) if len(chunk) else 0.0
        if rms < min_rms:
            continue

        inputs = processor(
            chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            logits = model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(predicted_ids)[0].strip()
        if not text:
            continue

        captions.append(
            Caption(
                index=cue_index,
                start_seconds=start / sample_rate,
                end_seconds=end / sample_rate,
                text=text,
            )
        )
        cue_index += 1
    return captions


def write_vtt(path: Path, captions: list[Caption]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for caption in captions:
        lines.append(str(caption.index))
        lines.append(
            f"{format_vtt_time(caption.start_seconds)} --> {format_vtt_time(caption.end_seconds)}"
        )
        lines.append(caption.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, captions: list[Caption]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "index": caption.index,
            "start_seconds": caption.start_seconds,
            "end_seconds": caption.end_seconds,
            "text": caption.text,
        }
        for caption in captions
    ]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_device(raw: str | None) -> str:
    if raw:
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Kikuyu WebVTT captions from a YouTube URL or local audio file."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="YouTube URL to download with yt-dlp.")
    source.add_argument("--audio-file", type=Path, help="Existing audio/video file to caption.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANG)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HF token for private models. Defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN.",
    )
    parser.add_argument("--device", default=None, help="Override device: cpu, mps, or cuda.")
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.5)
    parser.add_argument("--min-rms", type=float, default=0.002)
    parser.add_argument("--output", type=Path, default=Path("artifacts/captions/kikuyu.vtt"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--keep-wav", type=Path, default=None, help="Optional path to save the 16 kHz mono WAV.")
    parser.add_argument("--print-captions", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        target_lang=args.target_lang,
        token=args.token,
    )
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id, token=args.token).to(device)
    model.eval()

    with tempfile.TemporaryDirectory(prefix="kikuyu-youtube-caption-") as tmp:
        work_dir = Path(tmp)
        source_audio = download_youtube_audio(args.url, work_dir) if args.url else args.audio_file
        if source_audio is None:
            raise RuntimeError("No input source provided.")
        wav_path = args.keep_wav or (work_dir / "audio_16k_mono.wav")
        convert_to_wav_16k_mono(source_audio, wav_path)

        audio, sample_rate = load_audio(wav_path)
        if sample_rate != TARGET_SAMPLE_RATE:
            raise RuntimeError(f"Expected {TARGET_SAMPLE_RATE} Hz audio, got {sample_rate} Hz.")

        captions = transcribe_audio_chunks(
            audio=audio,
            sample_rate=sample_rate,
            processor=processor,
            model=model,
            device=device,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            min_rms=args.min_rms,
        )

    write_vtt(args.output, captions)
    if args.json_output:
        write_json(args.json_output, captions)
    if args.print_captions:
        for caption in captions:
            print(
                f"[{format_vtt_time(caption.start_seconds)} --> "
                f"{format_vtt_time(caption.end_seconds)}] {caption.text}"
            )
    print(f"Wrote {len(captions)} captions to {args.output}")


if __name__ == "__main__":
    main()
