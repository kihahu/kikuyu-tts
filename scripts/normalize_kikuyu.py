#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import unicodedata


MULTISPACE_RE = re.compile(r"\s+")


def normalize_kikuyu_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.lower().strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Kikuyu text.")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(normalize_kikuyu_text(args.text))


if __name__ == "__main__":
    main()
