from __future__ import annotations

import re


SPEAKER_PREFIX_RE = re.compile(r"^\s*[^:\n]{1,40}:\s*", re.IGNORECASE)
NON_RU_LETTERS_RE = re.compile(r"[^а-яё]+", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")


def normalize_text_for_tone(text: str, normalize_yo: bool = True) -> str:
    lines = []

    for line in text.splitlines():
        line = SPEAKER_PREFIX_RE.sub(" ", line)
        lines.append(line)

    text = " ".join(lines).lower()

    if normalize_yo:
        text = text.replace("ё", "е")

    text = NON_RU_LETTERS_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()

    return text