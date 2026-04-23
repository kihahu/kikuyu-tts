#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow "from scripts import ..." and "import kikuyu_orthography_wikipedia" in Colab
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from kikuyu_orthography_wikipedia import wikipedia_orthography_chars


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def iter_manifest_texts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if text:
                yield text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a character-level tokenizer vocabulary from JSONL manifests.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--out-dir", default="artifacts/tokenizer_kikuyu_char")
    parser.add_argument(
        "--extra-chars",
        default="",
        help="Extra characters to always add (e.g. loanwords) even if they never appear in the manifests, e.g. 'ñÑ'.",
    )
    parser.add_argument(
        "--wikipedia-orthography",
        action="store_true",
        help="Union reference graphemes: en.wikipedia.org/wiki/Kikuyu_language (Alphabet) and frozen "
        "excerpts from www.omniglot.com/writing/kikuyu.htm, per scripts/kikuyu_orthography_wikipedia.py.",
    )
    parser.add_argument(
        "--wikipedia-offline",
        action="store_true",
        help="With --wikipedia-orthography, do not use the network; use the frozen alphabet line in that module.",
    )
    args = parser.parse_args()

    chars = set()
    for manifest in (Path(args.train_manifest), Path(args.dev_manifest)):
        for text in iter_manifest_texts(manifest):
            chars.update(text)
    for ch in (args.extra_chars or ""):
        chars.add(ch)
    if args.wikipedia_orthography:
        fetch = not args.wikipedia_offline
        wiki = wikipedia_orthography_chars(use_fetch=fetch)
        n0 = len(chars)
        chars |= wiki
        print(
            f"| wikipedia-orthography: |wiki|={len(wiki)} fetch={fetch} "
            f"(manifest had {n0} codepoints, merged to {len(chars)})",
            file=sys.stderr,
        )

    ordered_chars = sorted(chars)
    vocab_tokens = SPECIAL_TOKENS + ordered_chars
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    tokenizer_config = {
        "model_type": "char",
        "special_tokens": {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
        },
        "vocab_size": len(vocab),
    }
    with (out_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    print(f"Wrote vocab with {len(vocab)} tokens to {out_dir}")


if __name__ == "__main__":
    main()
