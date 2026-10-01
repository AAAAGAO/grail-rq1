"""Text normalization and query-focused evidence truncation for RQ1."""

from __future__ import annotations

import re


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_.$]*|\d+", value.lower())


def compact_text(value: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", value)).strip()
    return value[:max_chars] + ("…" if len(value) > max_chars else "")


def query_relevant_text(value: str, query: str, max_chars: int) -> tuple[str, bool]:
    """Select a contiguous query-relevant window instead of blindly keeping the prefix."""
    clean = re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", value)).strip()
    if len(clean) <= max_chars:
        return clean, False
    query_terms = set(tokenize(query))
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", clean) if part.strip()]
    if not sentences:
        return compact_text(clean, max_chars), True
    best = max(range(len(sentences)), key=lambda i: (len(query_terms & set(tokenize(sentences[i]))), -i))
    selected = sentences[best]
    left, right = best - 1, best + 1
    while True:
        options = []
        if left >= 0:
            options.append((left, sentences[left] + " " + selected))
        if right < len(sentences):
            options.append((right, selected + " " + sentences[right]))
        fitting = [(i, text) for i, text in options if len(text) <= max_chars]
        if not fitting:
            break
        i, selected = max(fitting, key=lambda item: len(query_terms & set(tokenize(sentences[item[0]]))))
        if i == left:
            left -= 1
        else:
            right += 1
    return compact_text(selected, max_chars), True
