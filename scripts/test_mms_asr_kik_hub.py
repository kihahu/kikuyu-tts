#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC


DEFAULT_MODEL_ID = "kihahu/mms-asr-kik-waxal-ctc"
DEFAULT_TARGET_LANG = "kik"


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe one audio file with the final MMS Kikuyu ASR Hub model."
    )
    parser.add_argument("audio", type=Path, help="Path to WAV/FLAC/etc.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANG)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HF token for private models. Defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        target_lang=args.target_lang,
        token=args.token,
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_id,
        token=args.token,
    ).to(args.device)
    model.eval()

    audio, sample_rate = load_audio_mono(args.audio)
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs.input_values.to(args.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print(transcription)


if __name__ == "__main__":
    main()
