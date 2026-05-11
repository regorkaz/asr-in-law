from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import json

import jiwer
import numpy as np

from .parse import MetricSegment, flatten_normalized_words


def compute_asr_metrics_from_text(ref_raw_text: str, hyp_raw_text: str) -> dict[str, Any]:

    from .text import normalize_text_for_tone

    ref_text = normalize_text_for_tone(ref_raw_text)
    hyp_text = normalize_text_for_tone(hyp_raw_text)

    ref_words = ref_text.split()
    hyp_words = hyp_text.split()

    if not ref_words:
        return {
            "ref_words": 0,
            "hyp_words": len(hyp_words),
            "wer": None,
            "cer": None,
            "equal": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
        }

    out = jiwer.process_words(ref_text, hyp_text)

    ref_chars = "".join(ref_words)
    hyp_chars = "".join(hyp_words)
    cer = jiwer.cer(ref_chars, hyp_chars) if ref_chars else None

    return {
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "wer": out.wer,
        "cer": cer,
        "equal": out.hits,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
    }


def compute_speaker_metrics(
    ref_segments: list[MetricSegment],
    hyp_segments: list[MetricSegment],
) -> dict[str, Any]:
    ref_tagged = flatten_normalized_words(ref_segments)
    hyp_tagged = flatten_normalized_words(hyp_segments)

    ref_words = [w for w, _ in ref_tagged]
    hyp_words = [w for w, _ in hyp_tagged]

    ref_text = " ".join(ref_words)
    hyp_text = " ".join(hyp_words)

    if not ref_words:
        return {
            "speaker_accuracy": None,
            "speaker_compared_words": 0,
            "speaker_correct_words": 0,
            "speaker_coverage": None,
            "unknown_rate": None,
            "per_speaker": {},
            "confusion_matrix": {},
        }

    out = jiwer.process_words(ref_text, hyp_text)
    chunks = out.alignments[0] if out.alignments else []

    compared = 0
    correct = 0
    unknown_count = 0

    per_speaker: dict[str, dict[str, int]] = defaultdict(
        lambda: {"ref_words": 0, "compared": 0, "correct": 0}
    )
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _, ref_speaker in ref_tagged:
        per_speaker[ref_speaker]["ref_words"] += 1

    for ch in chunks:
        n_ref = ch.ref_end_idx - ch.ref_start_idx
        n_hyp = ch.hyp_end_idx - ch.hyp_start_idx

        if ch.type == "equal":
            n = min(n_ref, n_hyp)

        elif ch.type == "substitute":
            if n_ref != n_hyp:
                continue
            n = n_ref

        else:
            continue

        for i in range(n):
            r_idx = ch.ref_start_idx + i
            h_idx = ch.hyp_start_idx + i

            if r_idx >= len(ref_tagged) or h_idx >= len(hyp_tagged):
                continue

            ref_speaker = ref_tagged[r_idx][1]
            hyp_speaker = hyp_tagged[h_idx][1]

            compared += 1
            per_speaker[ref_speaker]["compared"] += 1
            confusion[ref_speaker][hyp_speaker] += 1

            if hyp_speaker == "UNKNOWN":
                unknown_count += 1

            if ref_speaker == hyp_speaker:
                correct += 1
                per_speaker[ref_speaker]["correct"] += 1

    return {
        "speaker_accuracy": correct / compared if compared else None,
        "speaker_compared_words": compared,
        "speaker_correct_words": correct,
        "speaker_coverage": compared / len(ref_words) if ref_words else None,
        "unknown_rate": unknown_count / compared if compared else None,
        "per_speaker": {
            speaker: {
                "accuracy": stats["correct"] / stats["compared"] if stats["compared"] else None,
                "coverage": stats["compared"] / stats["ref_words"] if stats["ref_words"] else None,
                "ref_words": stats["ref_words"],
                "compared_words": stats["compared"],
                "correct_words": stats["correct"],
            }
            for speaker, stats in sorted(per_speaker.items())
        },
        "confusion_matrix": {
            ref_speaker: dict(preds)
            for ref_speaker, preds in sorted(confusion.items())
        },
    }


def _summary(values: list[float | None]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]

    if not clean:
        return {
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }

    arr = np.asarray(clean, dtype=np.float64)

    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }

def _overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _segment_processing_times(
    *,
    segments: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    chunk_time_key: str,
) -> list[float | None]:
    out: list[float | None] = []

    for seg in segments:
        seg_start = seg.get("audio_start_s")
        seg_end = seg.get("audio_end_s")

        if seg_start is None or seg_end is None:
            out.append(None)
            continue

        seg_start = float(seg_start)
        seg_end = float(seg_end)

        if seg_end <= seg_start:
            out.append(None)
            continue

        total = 0.0
        has_value = False

        for ch in chunks:
            ch_start = ch.get("audio_start_s")
            ch_end = ch.get("audio_end_s")
            value = ch.get(chunk_time_key)

            if ch_start is None or ch_end is None or value is None:
                continue

            ch_start = float(ch_start)
            ch_end = float(ch_end)

            if _overlap_duration(seg_start, seg_end, ch_start, ch_end) <= 0:
                continue

            total += float(value)
            has_value = True

        out.append(total if has_value else None)

    return out

def summarize_timings(timings_path: str | Path | None) -> dict[str, Any]:
    if timings_path is None:
        return {}

    path = Path(timings_path)

    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))

    chunks = data.get("chunks", [])
    segments = data.get("segments", [])

    audio_duration_s = float(data.get("audio_duration_s") or 0.0)
    wall_clock_s = float(data.get("wall_clock_s") or 0.0)

    asr_times = [c.get("asr_processing_s") for c in chunks]
    speaker_times = [c.get("speaker_processing_s") for c in chunks]
    total_times = [c.get("total_processing_s") for c in chunks]
    segment_durations = [
        s.get("duration_s")
        if s.get("duration_s") is not None
        else (
            float(s.get("audio_end_s")) - float(s.get("audio_start_s"))
            if s.get("audio_end_s") is not None and s.get("audio_start_s") is not None
            else None
        )
        for s in segments
    ]

    segment_latencies = [s.get("segment_latency_s") for s in segments]

    availability_delays = [
        s.get("availability_delay_s")
        if s.get("availability_delay_s") is not None
        else (
            float(s.get("duration_s", 0.0)) + float(s.get("segment_latency_s", 0.0))
            if s.get("segment_latency_s") is not None
            else None
        )
        for s in segments
    ]

    stream_lags = [c.get("stream_lag_s") for c in chunks]

    asr_segment_times = _segment_processing_times(
        segments=segments,
        chunks=chunks,
        chunk_time_key="asr_processing_s",
    )

    speaker_segment_times = _segment_processing_times(
        segments=segments,
        chunks=chunks,
        chunk_time_key="speaker_processing_s",
    )

    total_segment_times = _segment_processing_times(
        segments=segments,
        chunks=chunks,
        chunk_time_key="total_processing_s",
    )

    asr_sum = sum(float(x) for x in asr_times if x is not None)
    speaker_sum = sum(float(x) for x in speaker_times if x is not None)
    total_sum = sum(float(x) for x in total_times if x is not None)

    clean_segment_durations = [
        float(x) for x in segment_durations
        if x is not None and np.isfinite(float(x))
    ]

    long_segments = {
        "gt_5s": sum(1 for x in clean_segment_durations if x > 5.0),
        "gt_10s": sum(1 for x in clean_segment_durations if x > 10.0),
        "gt_15s": sum(1 for x in clean_segment_durations if x > 15.0),
        "gt_20s": sum(1 for x in clean_segment_durations if x > 20.0),
    }

    return {
        "audio_duration_s": audio_duration_s,
        "wall_clock_s": wall_clock_s,

        "overall_rtf": wall_clock_s / audio_duration_s if audio_duration_s > 0 else None,

        "asr_processing_rtf": asr_sum / audio_duration_s if audio_duration_s > 0 else None,
        "speaker_processing_rtf": speaker_sum / audio_duration_s if audio_duration_s > 0 else None,
        "total_processing_rtf": total_sum / audio_duration_s if audio_duration_s > 0 else None,

        "n_chunks": len(chunks),
        "n_segments": len(segments),

        "asr_chunk_s": _summary(asr_times),
        "speaker_chunk_s": _summary(speaker_times),
        "total_chunk_s": _summary(total_times),

        "asr_segment_s": _summary(asr_segment_times),
        "speaker_segment_s": _summary(speaker_segment_times),
        "total_segment_s": _summary(total_segment_times),

        "segment_duration_s": _summary(segment_durations),
        "segment_latency_s": _summary(segment_latencies),
        "availability_delay_s": _summary(availability_delays),
        "stream_lag_s": _summary(stream_lags),

        "long_segments": long_segments,
        "first_segment_latency_s": segment_latencies[0] if segment_latencies else None,
    }


def compute_all_metrics(
    ref_segments: list[MetricSegment],
    hyp_segments: list[MetricSegment],
    timings_path: str | Path | None = None,
    ref_raw_text: str | None = None,
    hyp_raw_text: str | None = None,
) -> dict[str, Any]:
    if ref_raw_text is None:
        ref_raw_text = " ".join(seg.text for seg in ref_segments)

    if hyp_raw_text is None:
        hyp_raw_text = " ".join(seg.text for seg in hyp_segments)

    return {
        "asr": compute_asr_metrics_from_text(ref_raw_text, hyp_raw_text),
        "speaker": compute_speaker_metrics(ref_segments, hyp_segments),
        "timing": summarize_timings(timings_path),
    }