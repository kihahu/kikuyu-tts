#!/usr/bin/env python3
"""Run ASR inference on one audio file using a fine-tuned MMS Wav2Vec2 CTC checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC


def load_audio_mono_16k(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != 16_000:
        duration = len(data) / float(sr)
        n_new = max(1, int(duration * 16_000))
        t_old = np.linspace(0.0, duration, num=len(data), endpoint=False)
        t_new = np.linspace(0.0, duration, num=n_new, endpoint=False)
        data = np.interp(t_new, t_old, data).astype(np.float32)
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Transcribe one WAV/FLAC/etc. with a saved MMS ASR checkpoint.")
    p.add_argument("--model-dir", type=Path, required=True, help="Directory with config, model.safetensors, tokenizer, etc.")
    p.add_argument("--audio", type=Path, required=True, help="Path to audio file (resampled to 16 kHz mono internally).")
    p.add_argument(
        "--target-lang",
        default="kik",
        help="MMS language id (must match training; default kik).",
    )
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(str(args.model_dir), target_lang=args.target_lang)
    model = Wav2Vec2ForCTC.from_pretrained(str(args.model_dir), target_lang=args.target_lang)
    model.eval().to(device)

    wav = load_audio_mono_16k(args.audio)
    inputs = processor(wav, sampling_rate=16_000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    print(text)


if __name__ == "__main__":
    main()
