from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import threading
from .config import SETTINGS

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
    return NormalizedPhrase(
        text=str(text).strip(),
        start_time=float(start),
        end_time=float(end),
        raw=obj.model_dump() if hasattr(obj, "model_dump") else (
            obj.__dict__ if hasattr(obj, "__dict__") else str(obj)
        ),
        asr_confidence=float(conf) if conf is not None else None,
    )

class ToneStreamingASR:
    def __init__(self) -> None:
        self._pipeline = None
        self._chunk_size = None
        self._padding = None
        self._lock = threading.Lock()

    def _configure_splitter(self) -> None:
        splitter = getattr(self._pipeline, "logprob_splitter", None)

        if splitter is None:
            return

        frame_ms = 30.0

        max_phrase_frames = max(
            1,
            int(round(SETTINGS.tone_max_phrase_duration_ms / frame_ms)),
        )

        min_silence_frames = max(
            1,
            int(round(SETTINGS.tone_min_silence_duration_ms / frame_ms)),
        )

        splitter.MAX_PHRASE_DURATION = max_phrase_frames
        splitter.MIN_SILENCE_DURATION = min_silence_frames
        splitter.SILENCE_THRESHOLD = SETTINGS.tone_silence_threshold

    def load(self) -> None:
        if self._pipeline is not None:
            return

        from tone import StreamingCTCPipeline, DecoderType

        decoder_name = SETTINGS.tone_decoder_type.strip().lower().replace("-", "_")

        if decoder_name in {"beam", "beam_search", "kenlm"}:
            decoder_type = DecoderType.BEAM_SEARCH
        elif decoder_name in {"greedy"}:
            decoder_type = DecoderType.GREEDY
        else:
            raise ValueError(
                f"Unsupported T-one decoder type: {SETTINGS.tone_decoder_type}. "
                "Use 'beam_search'/'kenlm' or 'greedy'."
            )

        self._pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=decoder_type)
        self._configure_splitter()
        self._chunk_size = self._pipeline.CHUNK_SIZE
        self._padding = self._pipeline.PADDING

    @property
    def chunk_size(self) -> int:
        self.load()
        return int(self._chunk_size)

    @property
    def padding(self) -> int:
        self.load()
        return int(self._padding)

    @property
    def pipeline(self):
        self.load()
        return self._pipeline

    def forward_chunk(self, audio_chunk: np.ndarray, state: Any) -> tuple[list[NormalizedPhrase], Any]:
        self.load()
        audio_chunk = np.asarray(audio_chunk, dtype=np.int32)

        if audio_chunk.ndim != 1:
            audio_chunk = audio_chunk.reshape(-1)

        if audio_chunk.shape[0] != self.chunk_size:
            if audio_chunk.shape[0] < self.chunk_size:
                audio_chunk = np.pad(audio_chunk, (0, self.chunk_size - audio_chunk.shape[0]))
            else:
                audio_chunk = audio_chunk[: self.chunk_size]

        # Lock protects shared pipeline when several sessions are active.
        with self._lock:
            new_phrases, new_state = self.pipeline.forward(audio_chunk, state)

        normalized = [normalize_phrase(p) for p in (new_phrases or [])]
        return normalized, new_state

    def finalize(self, state: Any) -> tuple[list[NormalizedPhrase], Any]:
        self.load()

        with self._lock:
            new_phrases, new_state = self.pipeline.finalize(state)

        normalized = [normalize_phrase(p) for p in (new_phrases or [])]
        return normalized, new_state
