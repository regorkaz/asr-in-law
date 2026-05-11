from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

from .text import normalize_text_for_tone


@dataclass(frozen=True)
class MetricSegment:
    speaker: str
    text: str
    start: float | None = None
    end: float | None = None


SPEAKER_MAP = {
    "юрист": "LAWYER",
    "lawyer": "LAWYER",
    "адвокат": "LAWYER",

    "клиент": "CLIENT",
    "client": "CLIENT",

    "unknown": "UNKNOWN",
    "неизвестно": "UNKNOWN",
}


COLON_SPEAKER_RE = re.compile(
    r"^\s*(?P<speaker>юрист|клиент|lawyer|client|адвокат)\s*:\s*(?P<text>.*)$",
    flags=re.IGNORECASE,
)

BRACKET_SPEAKER_RE = re.compile(
    r"^\s*\[(?P<speaker>[^\]]+)\]\s*:?\s*(?P<text>.*)$",
    flags=re.IGNORECASE,
)


def normalize_speaker_label(label: str | None) -> str:
    if not label:
        return "UNKNOWN"

    key = str(label).strip().lower()
    return SPEAKER_MAP.get(key, str(label).strip().upper())


def parse_reference_transcript(path: str | Path) -> list[MetricSegment]:
    path = Path(path)
    segments: list[MetricSegment] = []

    current_speaker = "UNKNOWN"

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        m = COLON_SPEAKER_RE.match(stripped)
        if m:
            current_speaker = normalize_speaker_label(m.group("speaker"))
            text = m.group("text").strip()
        else:
            m = BRACKET_SPEAKER_RE.match(stripped)
            if m:
                current_speaker = normalize_speaker_label(m.group("speaker"))
                text = m.group("text").strip()
            else:
                text = stripped

        if not text:
            continue

        segments.append(
            MetricSegment(
                speaker=current_speaker,
                text=text,
            )
        )

    return segments


def parse_predicted_transcript_json(path: str | Path) -> list[MetricSegment]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    segments: list[MetricSegment] = []

    for item in data.get("segments", []):
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        segments.append(
            MetricSegment(
                speaker=normalize_speaker_label(item.get("speaker")),
                text=text,
                start=float(item["start_time"]) if item.get("start_time") is not None else None,
                end=float(item["end_time"]) if item.get("end_time") is not None else None,
            )
        )

    return segments


def flatten_normalized_words(segments: list[MetricSegment]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    for seg in segments:
        norm = normalize_text_for_tone(seg.text)
        for word in norm.split():
            if word:
                out.append((word, seg.speaker))

    return out


def normalized_text_from_segments(segments: list[MetricSegment]) -> str:
    return " ".join(word for word, _ in flatten_normalized_words(segments)).strip()

def read_reference_raw_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def raw_text_from_predicted_segments(path: str | Path) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    texts = []

    for item in data.get("segments", []):
        text = str(item.get("text", "")).strip()
        if text:
            texts.append(text)

    return " ".join(texts)