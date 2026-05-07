#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub.errors import RepositoryNotFoundError
from transformers import AutoProcessor, Wav2Vec2ForCTC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.synthesize_mms_vits_checkpoint import (
    load_checkpoint_for_synthesis,
    pick_device,
    synthesize_text,
)


DEFAULT_ASR_MODEL_ID = "kihahu/mms-asr-kik-waxal-ctc"
DEFAULT_ASR_PROCESSOR_ID = "facebook/mms-1b-all"
DEFAULT_TTS_REPO_ID = "kihahu/mms-tts-kik-waxal-v1"
DEFAULT_TTS_CHECKPOINT = "mms_vits_finetune/vits/logs/mms_kik_waxal_single_speaker/G_77100.pth"
DEFAULT_TTS_CONFIG = "mms_vits_finetune/vits/logs/mms_kik_waxal_single_speaker/config.json"
DEFAULT_TTS_VOCAB = "mms_vits_finetune/vits/logs/mms_kik_waxal_single_speaker/vocab.txt"
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
    run(["yt-dlp", "--no-playlist", "-f", "bestaudio/best", "-o", output_template, url])
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

        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True)
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

        captions.append(Caption(cue_index, start / sample_rate, end / sample_rate, text))
        cue_index += 1
    return captions


def write_vtt(path: Path, captions: list[Caption]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for caption in captions:
        lines.append(str(caption.index))
        lines.append(f"{format_vtt_time(caption.start_seconds)} --> {format_vtt_time(caption.end_seconds)}")
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


def build_hub_auth(cli_token: str | None = None) -> dict[str, str]:
    if cli_token is not None and str(cli_token).strip():
        return {"token": str(cli_token).strip()}
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        raw = os.environ.get(key)
        if raw and raw.strip():
            return {"token": raw.strip()}
    return {}


def resolve_pretrained_source(raw: str) -> str:
    path = Path(raw).expanduser()
    if path.is_dir() or path.is_absolute() or raw.startswith((".", "~")):
        return str(path.resolve())
    return raw


def is_local_model_path(model_id: str) -> bool:
    path = Path(model_id)
    return path.is_dir() or path.is_absolute()


def walk_exception_causes(exc: BaseException):
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        yield cur
        seen.add(id(cur))
        cur = cur.__cause__


def exception_blob(exc: BaseException) -> str:
    return " ".join(str(part).lower() for part in walk_exception_causes(exc))


def is_hub_access_failure(exc: BaseException) -> bool:
    blob = exception_blob(exc)
    for part in walk_exception_causes(exc):
        if isinstance(part, RepositoryNotFoundError):
            return True
    return any(
        snippet in blob
        for snippet in (
            "401",
            "403",
            "404",
            "invalid token",
            "unauthorized",
            "repository not found",
            "not a valid model identifier",
        )
    )


def exit_on_hub_access_failure(exc: OSError, hub_auth: dict[str, str]) -> None:
    if not is_hub_access_failure(exc):
        return
    auth_message = "using this token" if hub_auth else "and no token was passed"
    raise SystemExit(
        f"Hugging Face could not read the requested model repo {auth_message}. Verify repo access and token permissions.\n"
        f"Underlying error:\n  {exc}"
    ) from exc


def processor_files_missing(exc: BaseException) -> bool:
    blob = exception_blob(exc)
    return (
        "unrecognized processing class" in blob
        or "can't instantiate a processor" in blob
        or "preprocessor_config.json" in blob
        or "processor_config.json" in blob
        or "tokenizer_config.json" in blob
        or "special_tokens_map.json" in blob
    )


def load_processor(
    model_id: str,
    target_lang: str,
    processor_id: str | None,
    revision: str | None,
    hub_auth: dict[str, str],
) -> AutoProcessor:
    hub_kw = {**hub_auth}
    if revision is not None:
        hub_kw["revision"] = revision
    if processor_id:
        return AutoProcessor.from_pretrained(processor_id, target_lang=target_lang, **hub_kw)
    try:
        return AutoProcessor.from_pretrained(model_id, target_lang=target_lang, **hub_kw)
    except (OSError, ValueError) as exc:
        if isinstance(exc, OSError):
            exit_on_hub_access_failure(exc, hub_auth)
        if not processor_files_missing(exc):
            raise
    return AutoProcessor.from_pretrained(DEFAULT_ASR_PROCESSOR_ID, target_lang=target_lang, **hub_auth)


def load_asr(
    *,
    model_id: str,
    processor_id: str,
    revision: str | None,
    target_lang: str,
    token: str | None,
    device: str,
) -> tuple[AutoProcessor, Wav2Vec2ForCTC]:
    model_id = resolve_pretrained_source(model_id)
    hub_revision = None if is_local_model_path(model_id) else revision
    hub_auth = build_hub_auth(token)
    rev_kw = {**hub_auth}
    if hub_revision is not None:
        rev_kw["revision"] = hub_revision

    processor = load_processor(
        model_id=model_id,
        target_lang=target_lang,
        processor_id=processor_id or None,
        revision=hub_revision,
        hub_auth=hub_auth,
    )
    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=target_lang, **rev_kw)
    except OSError as exc:
        if "adapter." in str(exc) and "does not appear to have a file named" in str(exc):
            model = Wav2Vec2ForCTC.from_pretrained(model_id, **rev_kw)
        else:
            exit_on_hub_access_failure(exc, hub_auth)
            raise
    model.eval().to(device)
    return processor, model


def transcript_from_captions(captions: list[Caption]) -> str:
    parts: list[str] = []
    previous = ""
    for caption in captions:
        text = " ".join(caption.text.split())
        if not text or text == previous:
            continue
        parts.append(text)
        previous = text
    return " ".join(parts)


def write_transcript(path: Path, transcript: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript + "\n", encoding="utf-8")


def synthesize_captions(
    captions: list[Caption],
    *,
    repo_id: str,
    checkpoint: str,
    config: str,
    vocab: str,
    vits_repo: Path,
    device: str,
    output_wav: Path,
    gap_seconds: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
) -> None:
    tts = load_checkpoint_for_synthesis(
        repo_id=repo_id,
        checkpoint=checkpoint,
        config=config,
        vocab=vocab,
        vits_repo=vits_repo,
        device=device,
    )
    sample_rate = int(tts.hps.data.sampling_rate)
    gap = np.zeros(max(0, int(round(gap_seconds * sample_rate))), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for caption in captions:
        text = " ".join(caption.text.split())
        if not text:
            continue
        audio = synthesize_text(
            tts,
            text,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )
        if len(audio) == 0:
            continue
        if chunks and len(gap):
            chunks.append(gap)
        chunks.append(audio)

    if not chunks:
        raise RuntimeError("No TTS audio was generated from the ASR transcript.")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, np.concatenate(chunks), sample_rate, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/listen to Kikuyu YouTube audio with fine-tuned ASR, then synthesize the transcript with fine-tuned MMS/VITS TTS."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="YouTube URL to download with yt-dlp.")
    source.add_argument("--audio-file", type=Path, help="Existing audio/video file.")
    parser.add_argument("--asr-model-id", default=DEFAULT_ASR_MODEL_ID)
    parser.add_argument("--asr-processor-id", default=DEFAULT_ASR_PROCESSOR_ID)
    parser.add_argument("--asr-revision", default=None)
    parser.add_argument("--target-lang", default="kik")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HF token for private models. Defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN.",
    )
    parser.add_argument("--asr-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--tts-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.5)
    parser.add_argument("--min-rms", type=float, default=0.002)
    parser.add_argument("--tts-repo-id", default=DEFAULT_TTS_REPO_ID)
    parser.add_argument("--tts-checkpoint", default=DEFAULT_TTS_CHECKPOINT)
    parser.add_argument("--tts-config", default=DEFAULT_TTS_CONFIG)
    parser.add_argument("--tts-vocab", default=DEFAULT_TTS_VOCAB)
    parser.add_argument("--vits-repo", type=Path, default=Path("artifacts/vits_inference"))
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--gap-seconds", type=float, default=0.2)
    parser.add_argument("--output-wav", type=Path, default=Path("artifacts/youtube_tts/kikuyu_tts.wav"))
    parser.add_argument("--transcript-output", type=Path, default=Path("artifacts/youtube_tts/transcript.txt"))
    parser.add_argument("--captions-output", type=Path, default=Path("artifacts/youtube_tts/captions.vtt"))
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/youtube_tts/captions.json"))
    parser.add_argument("--keep-source-wav", type=Path, default=None)
    parser.add_argument("--print-transcript", action="store_true")
    args = parser.parse_args()

    asr_device = pick_device(args.asr_device)
    tts_device = pick_device(args.tts_device)
    if asr_device == "mps":
        # Wav2Vec2 CTC inference is more reliable on CPU than MPS for long chunked runs.
        asr_device = "cpu"

    processor, asr_model = load_asr(
        model_id=args.asr_model_id,
        processor_id=args.asr_processor_id,
        revision=args.asr_revision,
        target_lang=args.target_lang,
        token=args.token,
        device=asr_device,
    )

    with tempfile.TemporaryDirectory(prefix="kikuyu-youtube-asr-tts-") as tmp:
        work_dir = Path(tmp)
        source_audio = download_youtube_audio(args.url, work_dir) if args.url else args.audio_file
        if source_audio is None:
            raise RuntimeError("No input source provided.")
        wav_path = args.keep_source_wav or (work_dir / "audio_16k_mono.wav")
        convert_to_wav_16k_mono(source_audio, wav_path)
        audio, sample_rate = load_audio(wav_path)
        if sample_rate != 16_000:
            raise RuntimeError(f"Expected 16000 Hz audio, got {sample_rate} Hz.")

        captions = transcribe_audio_chunks(
            audio=audio,
            sample_rate=sample_rate,
            processor=processor,
            model=asr_model,
            device=asr_device,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            min_rms=args.min_rms,
        )

    transcript = transcript_from_captions(captions)
    if not transcript:
        raise RuntimeError("ASR produced no transcript; try lowering --min-rms or increasing --chunk-seconds.")

    write_vtt(args.captions_output, captions)
    write_json(args.json_output, captions)
    write_transcript(args.transcript_output, transcript)
    if args.print_transcript:
        print(transcript)

    synthesize_captions(
        captions,
        repo_id=args.tts_repo_id,
        checkpoint=args.tts_checkpoint,
        config=args.tts_config,
        vocab=args.tts_vocab,
        vits_repo=args.vits_repo,
        device=tts_device,
        output_wav=args.output_wav,
        gap_seconds=args.gap_seconds,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        length_scale=args.length_scale,
    )

    manifest = {
        "asr_model_id": args.asr_model_id,
        "tts_repo_id": args.tts_repo_id,
        "tts_checkpoint": args.tts_checkpoint,
        "caption_count": len(captions),
        "transcript_output": str(args.transcript_output),
        "captions_output": str(args.captions_output),
        "json_output": str(args.json_output),
        "output_wav": str(args.output_wav),
    }
    manifest_path = args.output_wav.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote transcript to {args.transcript_output}")
    print(f"Wrote captions to {args.captions_output}")
    print(f"Wrote synthesized audio to {args.output_wav}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
