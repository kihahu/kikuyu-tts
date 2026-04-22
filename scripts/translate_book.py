#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from kikuyu_tts.chunking import chunk_text


DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_SRC_LANG = "eng_Latn"
DEFAULT_TGT_LANG = "kik_Latn"


def translate_chunks(
    *,
    model_name: str,
    src_lang: str,
    tgt_lang: str,
    chunks: list[str],
    max_new_tokens: int,
) -> list[str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    outputs: list[str] = []
    for text in chunks:
        encoded = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        translated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
        )
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        outputs.append(decoded[0].strip())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English text to Kikuyu text with NLLB.")
    parser.add_argument("--input", required=True, help="Input text file path.")
    parser.add_argument("--output", required=True, help="Output translated text path.")
    parser.add_argument("--sidecar-json", default="", help="Optional chunk-level JSON output path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NLLB model name.")
    parser.add_argument("--src-lang", default=DEFAULT_SRC_LANG, help="Source language code.")
    parser.add_argument("--tgt-lang", default=DEFAULT_TGT_LANG, help="Target language code.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation token cap.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    text = input_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, max_chars=args.chunk_size)
    translated = translate_chunks(
        model_name=args.model,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        chunks=[c.text for c in chunks],
        max_new_tokens=args.max_new_tokens,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(translated), encoding="utf-8")

    if args.sidecar_json:
        sidecar_path = Path(args.sidecar_json)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"chunk_index": chunks[i].index, "source_text": chunks[i].text, "translated_text": translated[i]}
            for i in range(len(chunks))
        ]
        sidecar_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
