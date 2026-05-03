#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

from datasets import load_dataset
from transformers import AutoProcessor

from normalize_kikuyu import normalize_kikuyu_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report OOV characters for a Kikuyu ASR dataset after normalization."
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Hugging Face dataset repo, e.g. google/WaxalNLP",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Dataset config name, e.g. kik_tts",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Text column used for transcripts.",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/mms-1b-all",
        help="Processor source for tokenizer vocab.",
    )
    parser.add_argument(
        "--target-lang",
        default="kik",
        help="MMS target language code.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many OOV chars to print per split.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap per split (0 means all rows).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_dataset(args.dataset_name, args.dataset_config)
    processor = AutoProcessor.from_pretrained(args.model_name, target_lang=args.target_lang)
    unk_id = processor.tokenizer.unk_token_id

    print(f"Dataset: {args.dataset_name} ({args.dataset_config or 'default'})")
    print(f"Text column: {args.text_column}")
    print(f"Tokenizer vocab size: {len(processor.tokenizer.get_vocab())}")
    print()

    for split, ds in raw.items():
        if args.text_column not in ds.column_names:
            print(f"[{split}] missing text column '{args.text_column}', skipping")
            print()
            continue

        unk_example_counts: Counter[str] = Counter()
        rows_with_unk = 0
        total_unk_tokens = 0
        checked_rows = 0

        iterable = ds
        if args.max_rows > 0:
            limit = min(args.max_rows, len(ds))
            iterable = ds.select(range(limit))

        for row in iterable:
            checked_rows += 1
            normalized = normalize_kikuyu_text(str(row[args.text_column] or ""))
            input_ids = processor(text=normalized).input_ids
            unk_count = sum(1 for token_id in input_ids if token_id == unk_id)
            if unk_count:
                rows_with_unk += 1
                total_unk_tokens += unk_count
                unk_example_counts.update([normalized[:120]])

        unk_rate = (rows_with_unk / checked_rows * 100.0) if checked_rows else 0.0
        print(
            f"[{split}] rows={checked_rows} rows_with_unk={rows_with_unk} "
            f"({unk_rate:.2f}%) total_unk_tokens={total_unk_tokens}"
        )

        if unk_example_counts:
            print(f"  Top {args.top_k} normalized examples with <unk>:")
            for text, count in unk_example_counts.most_common(args.top_k):
                print(f"    {count}: {text}")
        else:
            print("  No <unk> tokens after normalization.")
        print()


if __name__ == "__main__":
    main()
