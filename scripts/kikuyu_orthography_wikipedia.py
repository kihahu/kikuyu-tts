#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kikuyu (Gĩkũyũ) reference character inventory from open references:

1) **Wikipedia** — *Kikuyu language*, “Alphabet”
   https://en.wikipedia.org/wiki/Kikuyu_language

2) **Omniglot** — *Kikuyu* (African reference alphabet; chart also as Excel on site)
   https://www.omniglot.com/writing/kikuyu.htm

Wikipedia: the cited alphabet line is 20 lower-case letters (no ``f l p q s v x z``,
plus ``ĩ`` and ``ũ``): ``a b c d e g h i ĩ j k m n o r t u ũ w y``.

Omniglot’s HTML is often delivered as a JS challenge to bots, so we **freeze** the
sample verse and native-name gloss published on the page (see
``OMNIGLOT_FROZEN_TEXTS``) and union their characters. The on-page alphabet chart
is an image/Excel; update the frozen lines if the site text changes.
"""
from __future__ import annotations

import json
import re
import ssl
import urllib.parse
import urllib.request

# Fallback if API fetch fails (must match the cited Alphabet section of the article).
# Single-space-separated list, same order as the Wikipedia "The Kikuyu alphabet is:" line.
WIKI_ALPHABET_LINE = "a b c d e g h i ĩ j k m n o r t u ũ w y"

# Common punctuation / apostrophes in Kikuyu text (Wikipedia + running copy).
# ng' uses ASCII apostrophe; we also see ’ in some copy (normalize to ' in data).
KIKUYU_COMMON_SUPPLEMENT = "'ʼ"

# Frozen excerpts from https://www.omniglot.com/writing/kikuyu.htm (as published,
# “Sample text (John 1:1)” and the native name line). Union with Wikipedia base.
# Last reviewed: 2026-04 (page “Page last modified: 22.02.24” on site).
OMNIGLOT_FROZEN_TEXTS: tuple[str, ...] = (
    # John 1:1 sample
    "Kĩambĩrĩianĩ Ũhoro aarĩ o kuo, na aatũire harĩ Ngai, nake aarĩ o Ngai.",
    # “native name for the language is Gĩkũyũ …”
    "Gĩkũyũ",
)


def _upper_equiv(ch: str) -> str:
    if len(ch) != 1:
        return ""
    m = {
        "ĩ": "Ĩ",
        "Ĩ": "Ĩ",
        "ũ": "Ũ",
        "Ũ": "Ũ",
    }
    if ch in m:
        return m[ch]
    o = ch.upper()
    return o if len(o) == 1 else ""


def chars_from_wikipedia_alphabet_line(line: str) -> set[str]:
    """Parse the space-separated alphabet line into a set of single codepoints."""
    out: set[str] = set()
    for part in line.split():
        p = part.strip()
        if len(p) == 1:
            out.add(p)
        else:
            # e.g. accidental digraph; still take codepoints
            for c in p:
                if c.isalpha() or c in "ĩũĨŨ":
                    out.add(c)
    for ch in KIKUYU_COMMON_SUPPLEMENT:
        if len(ch) == 1:
            out.add(ch)
    u = set()
    for c in out:
        if len(c) == 1:
            u.add(c)
            uu = _upper_equiv(c)
            if uu and len(uu) == 1:
                u.add(uu)
    return u


def chars_from_omniglot_frozen_texts() -> set[str]:
    """Unique characters in the published Omniglot snippets (punctuation kept)."""
    s: set[str] = set()
    for t in OMNIGLOT_FROZEN_TEXTS:
        s.update(t)
    return s


def wikipedia_orthography_chars(*, use_fetch: bool) -> set[str]:
    """
    Base grapheme set: Wikipedia Kikuyu alphabet line (+ fetch if enabled)
    and frozen Omniglot page excerpts, merged. Uppercase mirrors apply only to
    the Wikipedia line path (see `chars_from_wikipedia_alphabet_line`); the
    Omniglot sample already includes mixed case as on the page.
    """
    base = chars_from_wikipedia_alphabet_line(WIKI_ALPHABET_LINE)
    if use_fetch:
        fetched = _fetch_alphabet_line_from_api()
        if fetched is not None:
            base = chars_from_wikipedia_alphabet_line(fetched) | base
    omni = chars_from_omniglot_frozen_texts()
    return base | omni


WIKI_API = "https://en.wikipedia.org/w/api.php"


def _fetch_alphabet_line_from_api() -> str | None:
    """Load full page extract, then pick the 'a b c d e ... ĩ ... ũ' alphabet line."""
    return _fetch_alphabet_line_fulltext()


def _fetch_alphabet_line_fulltext() -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "titles": "Kikuyu language",
        "prop": "extracts",
        "explaintext": 1,
    }
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "kikuyu-tts/1.0 (orthography; contact: https://github.com/kihahu/kikuyu-tts)"},
        )
        with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
            data = json.load(resp)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    pages = (data.get("query") or {}).get("pages") or {}
    for _pid, p in pages.items():
        ex = p.get("extract")
        if ex:
            return _find_alphabet_line(ex) or _find_alphabet_line_full(ex)
    return None


_ALPHABET_FOLLOW = re.compile(
    r"(?:Kikuyu|G[ĩi]k[ũu]y[ũu]) alphabet is:\s*\n\s*([a-zA-ZĩũĨŨ\s\u0129\u0169\u0128\u0168]+?)(?=\n\n|## |$)",
    re.MULTILINE,
)


def _find_alphabet_line(text: str) -> str | None:
    m = _ALPHABET_FOLLOW.search(text)
    if m:
        return m.group(1).strip()
    return None


def _find_alphabet_line_full(text: str) -> str | None:
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("##") or s.startswith("["):
            continue
        if re.match(r"^[a-zA-Zĩũ\.\s'ʼ]+$", s) and "ĩ" in s and "j" in s and len(s) > 20:
            return s
    return None
