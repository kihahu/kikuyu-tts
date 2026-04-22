#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full English->Kikuyu text+audio pipeline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--translated-output", default="artifacts/kikuyu_translated.txt")
    parser.add_argument("--chunk-json", default="artifacts/kikuyu_chunks.json")
    parser.add_argument("--audio-output-dir", default="artifacts/audio")
    parser.add_argument("--model", default="facebook/mms-tts-kik", help="TTS model path or HF repo id")
    parser.add_argument("--quantized-model-dir", default="models/mms-tts-kik-quantized")
    parser.add_argument("--quantize-first", action="store_true", help="Quantize model before synthesis.")
    parser.add_argument("--quantize-dtype", default="float16", choices=["float16", "int8"])
    parser.add_argument("--voice", default="")
    parser.add_argument("--audio-format", default="wav", choices=["wav", "mp3"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    translate_script = root / "scripts" / "translate_book.py"
    synth_script = root / "scripts" / "synthesize_book.py"
    quantize_script = root / "scripts" / "quantize_mms_tts.py"

    model_for_synthesis = args.model
    quantized_dir = Path(args.quantized_model_dir)
    if args.quantize_first:
        run(
            [
                sys.executable,
                str(quantize_script),
                "--model",
                args.model,
                "--output-dir",
                str(quantized_dir),
                "--dtype",
                args.quantize_dtype,
            ]
        )
        model_for_synthesis = str(quantized_dir)
    elif quantized_dir.exists():
        model_for_synthesis = str(quantized_dir)

    run(
        [
            sys.executable,
            str(translate_script),
            "--input",
            args.input,
            "--output",
            args.translated_output,
            "--sidecar-json",
            args.chunk_json,
        ]
    )
    synth_cmd = [
        sys.executable,
        str(synth_script),
        "--input",
        args.translated_output,
        "--model",
        model_for_synthesis,
        "--output-dir",
        args.audio_output_dir,
        "--format",
        args.audio_format,
    ]
    if args.voice:
        synth_cmd.extend(["--voice", args.voice])
    run(synth_cmd)


if __name__ == "__main__":
    main()
