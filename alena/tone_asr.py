from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(slots=True)
class NormalizedPhrase:
    text: str
    start_time: float
    end_time: float
    raw: Any
    asr_confidence: float | None = None


def _get_first(obj: Any, names: list[str], default: Any = None) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def normalize_phrase(obj: Any) -> NormalizedPhrase:
    text = _get_first(obj, ['text', 'transcript', 'sentence', 'phrase'], '')
    start = _get_first(obj, ['start_time', 'start', 'begin', 't_start', 'start_ts'], 0.0)
    end = _get_first(obj, ['end_time', 'end', 'finish', 't_end', 'end_ts'], start)
    conf = _get_first(obj, ['confidence', 'score', 'probability', 'asr_confidence'], None)
    return NormalizedPhrase(text=str(text).strip(), start_time=float(start), end_time=float(end), raw=obj, asr_confidence=float(conf) if conf is not None else None)

class ToneStreamingASR:
    def __init__(self) -> None:
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        from tone import StreamingCTCPipeline
        self._pipeline = StreamingCTCPipeline.from_hugging_face()

    @property
    def pipeline(self):
        self.load()
        return self._pipeline

    def forward_chunk(self, audio_chunk: np.ndarray, state: Any) -> tuple[list[NormalizedPhrase], Any]:
        self.load()
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        try:
            new_phrases, new_state = self.pipeline.forward(audio_chunk, state)
        except Exception:
            import torch
            tensor = torch.from_numpy(audio_chunk).float()
            new_phrases, new_state = self.pipeline.forward(tensor, state)
        normalized = [normalize_phrase(p) for p in (new_phrases or [])]
        return normalized, new_state

    def finalize(self, state: Any) -> tuple[list[NormalizedPhrase], Any]:
        self.load()
        new_phrases, new_state = self.pipeline.finalize(state)
        normalized = [normalize_phrase(p) for p in (new_phrases or [])]
        return normalized, new_state
