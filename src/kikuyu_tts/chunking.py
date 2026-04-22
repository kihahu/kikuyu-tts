from __future__ import annotations

from dataclasses import dataclass
import re


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    words = sentence.split()
    if not words:
        return []
    out: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            out.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        out.append(" ".join(current))
    return out


def chunk_text(text: str, max_chars: int = 800) -> list[TextChunk]:
    if max_chars < 64:
        raise ValueError("max_chars must be at least 64")
    normalized = normalize_text(text)
    paragraphs = split_paragraphs(normalized)
    chunks: list[TextChunk] = []
    idx = 0

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(TextChunk(index=idx, text=paragraph))
            idx += 1
            continue

        sentences = SENTENCE_SPLIT_RE.split(paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                if current:
                    chunks.append(TextChunk(index=idx, text=current.strip()))
                    idx += 1
                    current = ""
                for small_piece in split_long_sentence(sentence, max_chars=max_chars):
                    chunks.append(TextChunk(index=idx, text=small_piece))
                    idx += 1
                continue

            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(TextChunk(index=idx, text=current.strip()))
                idx += 1
                current = sentence

        if current:
            chunks.append(TextChunk(index=idx, text=current.strip()))
            idx += 1
    return chunks
