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
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for English text -> Kikuyu text -> Waxal G_77100 audio."
    )
    parser.add_argument("--input", required=True, help="UTF-8 English input file.")
    parser.add_argument("--output-wav", default="artifacts/english_to_kikuyu_audio/output.wav")
    parser.add_argument("--translated-output", default="artifacts/english_to_kikuyu_audio/kikuyu.txt")
    parser.add_argument("--manifest-json", default="artifacts/english_to_kikuyu_audio/manifest.json")
    parser.add_argument("--chunk-wav-dir", default="artifacts/english_to_kikuyu_audio/chunks")
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--translation-backend", choices=("nllb", "identity"), default="nllb")
    parser.add_argument("--translation-device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--tts-device", default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    pipeline_script = root / "scripts" / "english_to_kikuyu_audio.py"
    command = [
        sys.executable,
        str(pipeline_script),
        "--input",
        args.input,
        "--output-wav",
        args.output_wav,
        "--translated-output",
        args.translated_output,
        "--manifest-json",
        args.manifest_json,
        "--chunk-wav-dir",
        args.chunk_wav_dir,
        "--translation-backend",
        args.translation_backend,
        "--translation-device",
        args.translation_device,
        "--tts-device",
        args.tts_device,
    ]
    if args.max_chunks:
        command.extend(["--max-chunks", str(args.max_chunks)])
    run(command)


if __name__ == "__main__":
    main()
