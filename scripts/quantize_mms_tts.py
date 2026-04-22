#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, VitsModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize MMS-TTS checkpoint for smaller local inference footprint.")
    parser.add_argument("--model", default="facebook/mms-tts-kik")
    parser.add_argument("--output-dir", default="models/mms-tts-kik-quantized")
    parser.add_argument("--dtype", choices=["float16", "int8"], default="int8")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = VitsModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.dtype == "float16":
        model = model.half()
    else:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
