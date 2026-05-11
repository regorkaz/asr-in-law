from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .compute import compute_all_metrics
from .parse import (
    parse_reference_transcript,
    parse_predicted_transcript_json,
    read_reference_raw_text,
    raw_text_from_predicted_segments,
)


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def _fmt_s(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f} s"


def _fmt_ms(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1000:.0f} ms"


def evaluate_one(
    *,
    name: str,
    ref_path: str | Path,
    pred_path: str | Path,
    timings_path: str | Path | None = None,
) -> dict[str, Any]:
    ref_segments = parse_reference_transcript(ref_path)
    hyp_segments = parse_predicted_transcript_json(pred_path)

    ref_raw_text = read_reference_raw_text(ref_path)
    hyp_raw_text = raw_text_from_predicted_segments(pred_path)

    metrics = compute_all_metrics(
        ref_segments=ref_segments,
        hyp_segments=hyp_segments,
        timings_path=timings_path,
        ref_raw_text=ref_raw_text,
        hyp_raw_text=hyp_raw_text,
    )

    return {
        "name": name,
        "ref_path": str(ref_path),
        "pred_path": str(pred_path),
        "timings_path": str(timings_path) if timings_path else None,
        "n_reference_segments": len(ref_segments),
        "n_predicted_segments": len(hyp_segments),
        "metrics": metrics,
    }


def _load_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"name", "ref", "pred"}
    missing = required - set(reader.fieldnames or [])

    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    return rows


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    def values(path: list[str]) -> list[float]:
        out = []
        for r in reports:
            cur: Any = r
            for key in path:
                cur = cur.get(key, {})
            if isinstance(cur, (int, float)):
                out.append(float(cur))
        return out

    def mean_or_none(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return {
        "files": len(reports),
        "macro_asr": {
            "wer": mean_or_none(values(["metrics", "asr", "wer"])),
            "cer": mean_or_none(values(["metrics", "asr", "cer"])),
        },
        "macro_speaker": {
            "speaker_accuracy": mean_or_none(values(["metrics", "speaker", "speaker_accuracy"])),
            "speaker_coverage": mean_or_none(values(["metrics", "speaker", "speaker_coverage"])),
            "unknown_rate": mean_or_none(values(["metrics", "speaker", "unknown_rate"])),
        },
        "macro_timing": {
            "overall_rtf": mean_or_none(values(["metrics", "timing", "overall_rtf"])),
            "asr_processing_rtf": mean_or_none(values(["metrics", "timing", "asr_processing_rtf"])),
            "total_processing_rtf": mean_or_none(values(["metrics", "timing", "total_processing_rtf"])),
        },
    }


def print_one_report(report: dict[str, Any]) -> None:
    m = report["metrics"]
    asr = m.get("asr", {})
    speaker = m.get("speaker", {})
    timing = m.get("timing", {})

    print("=" * 72)
    print(f"Evaluation: {report['name']}")
    print("=" * 72)
    print(f"Reference segments: {report['n_reference_segments']}")
    print(f"Predicted segments: {report['n_predicted_segments']}")

    print("\n--- ASR quality ---")
    print(f"WER: {_fmt_pct(asr.get('wer'))}")
    print(f"CER: {_fmt_pct(asr.get('cer'))}")
    print(f"Ref words / Hyp words: {asr.get('ref_words')} / {asr.get('hyp_words')}")
    print(
        f"Equal: {asr.get('equal', 0)}  "
        f"Subs: {asr.get('substitutions', 0)}  "
        f"Dels: {asr.get('deletions', 0)}  "
        f"Ins: {asr.get('insertions', 0)}"
    )

    print("\n--- Speaker attribution ---")
    print(f"Word-level speaker accuracy: {_fmt_pct(speaker.get('speaker_accuracy'))}")
    print(f"Speaker coverage: {_fmt_pct(speaker.get('speaker_coverage'))}")
    print(f"UNKNOWN rate on compared words: {_fmt_pct(speaker.get('unknown_rate'))}")
    print(
        f"Compared words: {speaker.get('speaker_compared_words', 0)} "
        f"(correct: {speaker.get('speaker_correct_words', 0)})"
    )

    for spk, stats in (speaker.get("per_speaker") or {}).items():
        print(
            f"  {spk}: "
            f"acc={_fmt_pct(stats.get('accuracy'))}, "
            f"coverage={_fmt_pct(stats.get('coverage'))}, "
            f"ref_words={stats.get('ref_words')}, "
            f"compared={stats.get('compared_words')}"
        )

    confusion = speaker.get("confusion_matrix") or {}
    if confusion:
        print("\nSpeaker confusion matrix:")
        for ref_spk, preds in confusion.items():
            print(f"  ref={ref_spk}: {preds}")

    if timing:
        print("\n--- Timing ---")
        print(f"Audio duration: {_fmt_s(timing.get('audio_duration_s'))}")
        print(f"Wall-clock: {_fmt_s(timing.get('wall_clock_s'))}")
        print(f"Overall RTF: {timing.get('overall_rtf'):.3f}" if timing.get("overall_rtf") is not None else "Overall RTF: n/a")
        print(f"ASR processing RTF: {timing.get('asr_processing_rtf'):.3f}" if timing.get("asr_processing_rtf") is not None else "ASR processing RTF: n/a")
        print(f"Total processing RTF: {timing.get('total_processing_rtf'):.3f}" if timing.get("total_processing_rtf") is not None else "Total processing RTF: n/a")
        print(f"Chunks: {timing.get('n_chunks')}")
        print(f"Segments: {timing.get('n_segments')}")

        asr_chunk = timing.get("asr_chunk_s") or {}
        speaker_chunk = timing.get("speaker_chunk_s") or {}
        total_chunk = timing.get("total_chunk_s") or {}

        asr_segment = timing.get("asr_segment_s") or {}
        speaker_segment = timing.get("speaker_segment_s") or {}
        total_segment = timing.get("total_segment_s") or {}

        segment_duration = timing.get("segment_duration_s") or {}
        latency = timing.get("segment_latency_s") or {}
        availability_delay = timing.get("availability_delay_s") or {}
        stream_lag = timing.get("stream_lag_s") or {}
        long_segments = timing.get("long_segments") or {}

        print(
            "ASR chunk mean / p50 / p95 / max: "
            f"{_fmt_ms(asr_chunk.get('mean'))} / "
            f"{_fmt_ms(asr_chunk.get('p50'))} / "
            f"{_fmt_ms(asr_chunk.get('p95'))} / "
            f"{_fmt_ms(asr_chunk.get('max'))}"
        )

        print(
            "Total chunk mean / p50 / p95 / max: "
            f"{_fmt_ms(total_chunk.get('mean'))} / "
            f"{_fmt_ms(total_chunk.get('p50'))} / "
            f"{_fmt_ms(total_chunk.get('p95'))} / "
            f"{_fmt_ms(total_chunk.get('max'))}"
        )

        print(
            "Speaker chunk mean / p50 / p95 / max: "
            f"{_fmt_ms(speaker_chunk.get('mean'))} / "
            f"{_fmt_ms(speaker_chunk.get('p50'))} / "
            f"{_fmt_ms(speaker_chunk.get('p95'))} / "
            f"{_fmt_ms(speaker_chunk.get('max'))}"
        )

        print(
            "ASR segment mean / p50 / p95 / max: "
            f"{_fmt_ms(asr_segment.get('mean'))} / "
            f"{_fmt_ms(asr_segment.get('p50'))} / "
            f"{_fmt_ms(asr_segment.get('p95'))} / "
            f"{_fmt_ms(asr_segment.get('max'))}"
        )

        print(
            "Speaker segment mean / p50 / p95 / max: "
            f"{_fmt_ms(speaker_segment.get('mean'))} / "
            f"{_fmt_ms(speaker_segment.get('p50'))} / "
            f"{_fmt_ms(speaker_segment.get('p95'))} / "
            f"{_fmt_ms(speaker_segment.get('max'))}"
        )

        print(
            "Total segment mean / p50 / p95 / max: "
            f"{_fmt_ms(total_segment.get('mean'))} / "
            f"{_fmt_ms(total_segment.get('p50'))} / "
            f"{_fmt_ms(total_segment.get('p95'))} / "
            f"{_fmt_ms(total_segment.get('max'))}"
        )

        print(
            "Segment duration mean / p50 / p95 / max: "
            f"{_fmt_s(segment_duration.get('mean'))} / "
            f"{_fmt_s(segment_duration.get('p50'))} / "
            f"{_fmt_s(segment_duration.get('p95'))} / "
            f"{_fmt_s(segment_duration.get('max'))}"
        )

        print(
            "Long segments >5s / >10s / >15s / >20s: "
            f"{long_segments.get('gt_5s', 0)} / "
            f"{long_segments.get('gt_10s', 0)} / "
            f"{long_segments.get('gt_15s', 0)} / "
            f"{long_segments.get('gt_20s', 0)}"
        )

        print(
            "Segment latency mean / p50 / p95 / max: "
            f"{_fmt_ms(latency.get('mean'))} / "
            f"{_fmt_ms(latency.get('p50'))} / "
            f"{_fmt_ms(latency.get('p95'))} / "
            f"{_fmt_ms(latency.get('max'))}"
        )

        print(
            "Availability delay mean / p50 / p95 / max: "
            f"{_fmt_s(availability_delay.get('mean'))} / "
            f"{_fmt_s(availability_delay.get('p50'))} / "
            f"{_fmt_s(availability_delay.get('p95'))} / "
            f"{_fmt_s(availability_delay.get('max'))}"
        )

        print(
            "Stream lag mean / p50 / p95 / max: "
            f"{_fmt_ms(stream_lag.get('mean'))} / "
            f"{_fmt_ms(stream_lag.get('p50'))} / "
            f"{_fmt_ms(stream_lag.get('p95'))} / "
            f"{_fmt_ms(stream_lag.get('max'))}"
        )


        print(f"First segment latency: {_fmt_ms(timing.get('first_segment_latency_s'))}")

    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m legal_asr_service.metrics.evaluate",
        description="Evaluate ASR WER/CER, speaker attribution and timing metrics.",
    )

    parser.add_argument("--ref", help="Reference transcript path.")
    parser.add_argument("--pred", help="Predicted transcript.json path.")
    parser.add_argument("--timings", default=None, help="timings.json path.")
    parser.add_argument("--name", default="consultation", help="Evaluation name.")

    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "CSV manifest for multiple evaluations. "
            "Required columns: name,ref,pred. Optional column: timings."
        ),
    )

    parser.add_argument("--out", default=None, help="Path to save JSON report.")
    parser.add_argument("--json-only", action="store_true", help="Print JSON only.")

    args = parser.parse_args()

    if args.manifest:
        reports = []

        for row in _load_manifest(args.manifest):
            reports.append(
                evaluate_one(
                    name=row["name"],
                    ref_path=row["ref"],
                    pred_path=row["pred"],
                    timings_path=row.get("timings") or None,
                )
            )

        report = {
            "mode": "manifest",
            "manifest": args.manifest,
            "reports": reports,
            "summary": _aggregate_reports(reports),
        }

        if not args.json_only:
            for item in reports:
                print_one_report(item)

            print("\n" + "#" * 72)
            print("MANIFEST SUMMARY")
            print("#" * 72)
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    else:
        if not args.ref or not args.pred:
            raise SystemExit("Use either --manifest or both --ref and --pred.")

        report = evaluate_one(
            name=args.name,
            ref_path=args.ref,
            pred_path=args.pred,
            timings_path=args.timings,
        )

        if not args.json_only:
            print_one_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if not args.json_only:
            print(f"\nSaved report: {out_path}")

    if args.json_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()