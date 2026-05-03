#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import unicodedata


MULTISPACE_RE = re.compile(r"\s+")
KIKUYU_MMS_LABEL_CHARS = set(" '-0124abcdefghijklmnopqrstuvwyzĩũʼ")
ORTHOGRAPHY_TRANSLATION = str.maketrans(
    {
        "ì": "ĩ",
        "í": "ĩ",
        "î": "ĩ",
        "ī": "ĩ",
        "Ì": "ĩ",
        "Í": "ĩ",
        "Î": "ĩ",
        "Ī": "ĩ",
        "ù": "ũ",
        "ú": "ũ",
        "û": "ũ",
        "ū": "ũ",
        "Ù": "ũ",
        "Ú": "ũ",
        "Û": "ũ",
        "Ū": "ũ",
    }
)
PUNCT_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": " ",
    "\u201d": " ",
    "\u2013": "-",
    "\u2014": "-",
    "_": " ",
}


def normalize_kikuyu_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    for src, dst in PUNCT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.translate(ORTHOGRAPHY_TRANSLATION).lower()
    text = "".join(char if char in KIKUYU_MMS_LABEL_CHARS else " " for char in text)
    return MULTISPACE_RE.sub(" ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Kikuyu text.")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(normalize_kikuyu_text(args.text))


if __name__ == "__main__":
    main()
