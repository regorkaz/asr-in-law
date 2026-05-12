from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import time
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ChunkTiming:
    chunk_index: int
    audio_start_s: float
    audio_end_s: float
    duration_s: float
    asr_processing_s: float | None = None
    speaker_processing_s: float | None = None
    total_processing_s: float | None = None
    emitted_segments: int = 0
    emitted_wall_s: float | None = None
    stream_lag_s: float | None = None


@dataclass(slots=True)
class SegmentTiming:
    segment_id: int
    audio_start_s: float
    audio_end_s: float
    duration_s: float
    emitted_wall_s: float
    segment_latency_s: float
    availability_delay_s: float


@dataclass(slots=True)
class TimingRecorder:
    created_wall: float = field(default_factory=time.perf_counter)
    stream_start_wall: float | None = None
    finalized_wall: float | None = None
    chunks: list[ChunkTiming] = field(default_factory=list)
    segments: list[SegmentTiming] = field(default_factory=list)

    def ensure_stream_started(self) -> None:
        if self.stream_start_wall is None:
            self.stream_start_wall = time.perf_counter()

    def wall_elapsed(self) -> float:
        """
        Wall-clock elapsed from actual first audio processing, not from session creation.
        """
        if self.stream_start_wall is None:
            return 0.0
        end = self.finalized_wall or time.perf_counter()
        return end - self.stream_start_wall

    def mark_finalized(self) -> None:
        self.finalized_wall = time.perf_counter()

    def add_chunk(
        self,
        *,
        chunk_index: int,
        audio_start_s: float,
        audio_end_s: float,
        duration_s: float,
        asr_processing_s: float | None,
        speaker_processing_s: float | None,
        total_processing_s: float | None,
        emitted_segments: int,
    ) -> None:
        self.ensure_stream_started()

        emitted_wall_s = self.wall_elapsed()

        # Approximate lag for real-time stream-file / microphone:
        # if audio_end_s is ahead of wall time -> negative is clipped to 0;
        # if wall time is ahead of audio_end_s -> processing/output is lagging.
        stream_lag_s = max(0.0, emitted_wall_s - audio_end_s)

        self.chunks.append(
            ChunkTiming(
                chunk_index=chunk_index,
                audio_start_s=audio_start_s,
                audio_end_s=audio_end_s,
                duration_s=duration_s,
                asr_processing_s=asr_processing_s,
                speaker_processing_s=speaker_processing_s,
                total_processing_s=total_processing_s,
                emitted_segments=emitted_segments,
                emitted_wall_s=emitted_wall_s,
                stream_lag_s=stream_lag_s,
            )
        )

    def add_segment(self, *, segment_id: int, audio_start_s: float, audio_end_s: float) -> None:
        self.ensure_stream_started()

        emitted_wall_s = self.wall_elapsed()

        duration_s = max(0.0, audio_end_s - audio_start_s)

        segment_latency_s = emitted_wall_s - audio_end_s
        segment_latency_s = max(0.0, segment_latency_s)

        availability_delay_s = emitted_wall_s - audio_start_s
        availability_delay_s = max(0.0, availability_delay_s)

        self.segments.append(
            SegmentTiming(
                segment_id=segment_id,
                audio_start_s=audio_start_s,
                audio_end_s=audio_end_s,
                duration_s=duration_s,
                emitted_wall_s=emitted_wall_s,
                segment_latency_s=segment_latency_s,
                availability_delay_s=availability_delay_s,
            )
        )

    def to_dict(self, finalized: bool = False) -> dict[str, Any]:
        if finalized:
            self.mark_finalized()

        audio_duration_s = 0.0
        if self.chunks:
            audio_duration_s = max(c.audio_end_s for c in self.chunks)

        wall_clock_s = self.wall_elapsed()

        return {
            "finalized": finalized,
            "audio_duration_s": audio_duration_s,
            "wall_clock_s": wall_clock_s,
            "chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "audio_start_s": c.audio_start_s,
                    "audio_end_s": c.audio_end_s,
                    "duration_s": c.duration_s,
                    "asr_processing_s": c.asr_processing_s,
                    "speaker_processing_s": c.speaker_processing_s,
                    "total_processing_s": c.total_processing_s,
                    "emitted_segments": c.emitted_segments,
                    "emitted_wall_s": c.emitted_wall_s,
                    "stream_lag_s": c.stream_lag_s,
                }
                for c in self.chunks
            ],
            "segments": [
                {
                    "segment_id": s.segment_id,
                    "audio_start_s": s.audio_start_s,
                    "audio_end_s": s.audio_end_s,
                    "emitted_wall_s": s.emitted_wall_s,
                    "segment_latency_s": s.segment_latency_s,
                    "availability_delay_s": s.availability_delay_s,
                }
                for s in self.segments
            ],
        }

    def write_json(self, path: str | Path, finalized: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(finalized=finalized), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def summarize_values(values: list[float]) -> dict[str, float | None]:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}

    arr = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }