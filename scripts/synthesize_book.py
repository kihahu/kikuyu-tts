#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kikuyu_tts.chunking import chunk_text, normalize_text
from kikuyu_tts.mlx_fork import load_model


def split_kikuyu_blocks(text: str, max_chars: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [c.text for c in chunk_text(normalized, max_chars=max_chars)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Kikuyu text into audio with MLX TTS.")
    parser.add_argument("--input", required=True, help="Translated Kikuyu text file.")
    parser.add_argument("--model", required=True, help="MLX TTS model path or HF repo id.")
    parser.add_argument("--output-dir", default="artifacts/audio")
    parser.add_argument("--voice", default="", help="Optional voice preset if model supports it.")
    parser.add_argument("--format", default="wav", choices=["wav", "mp3"])
    parser.add_argument("--chunk-size", type=int, default=700, help="Text chunk size for long books.")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    chunks = split_kikuyu_blocks(text, max_chars=args.chunk_size)
    if not chunks:
        raise ValueError("Input text is empty after normalization.")

    model = load_model(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, block in enumerate(chunks):
        kwargs = {}
        if args.voice:
            kwargs["voice"] = args.voice
        generated = list(model.generate(text=block, **kwargs))
        if not generated:
            continue
        audio_file = output_dir / f"audio_{idx:04d}.{args.format}"
        generated[0].save(str(audio_file))


if __name__ == "__main__":
    main()
