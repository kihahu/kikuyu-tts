#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Kikuyu text with a Hugging Face MMS/VITS TTS model.")
    parser.add_argument("--text", default="", help="Text to synthesize. Use --input for longer text.")
    parser.add_argument("--input", default="", help="UTF-8 text file to synthesize.")
    parser.add_argument("--model", default="facebook/mms-tts-kik", help="HF model id or local HF-format model path.")
    parser.add_argument("--output", default="artifacts/tts_eval/single.wav")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    text = args.text
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    text = " ".join(text.split())
    if not text:
        raise ValueError("Provide text with --text or --input.")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = VitsModel.from_pretrained(args.model).to(device)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        waveform = model(**inputs).waveform[0].detach().cpu().numpy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, int(model.config.sampling_rate), subtype="PCM_16")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
