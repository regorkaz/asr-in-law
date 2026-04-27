"""CLI: compute WER, speaker accuracy, and timing summary for a consultation run."""

import argparse
import json
import os
import sys

from metrics.compute import compute_asr_and_speaker, summarize_timings
from metrics.parse import parse_ground_truth, parse_predicted


def _pct(x: float | None) -> str:
    return f"{x * 100:.2f}%" if x is not None else "n/a"


def _ms(x: float | None) -> str:
    return f"{x * 1000:.0f} ms" if x is not None else "n/a"


def evaluate(consultation: str, data_dir: str) -> dict:
    gt_path = os.path.join(data_dir, "input", consultation, "text.txt")
    pred_path = os.path.join(data_dir, "output", f"{consultation}_transcript.txt")
    timings_path = os.path.join(data_dir, "output", f"{consultation}_timings.json")

    report: dict = {"consultation": consultation}

    if not os.path.exists(pred_path):
        print(f"[metrics] No prediction file: {pred_path}", file=sys.stderr)
        return report
    hyp = parse_predicted(pred_path)
    report["n_predicted_segments"] = len(hyp)

    has_gt = os.path.exists(gt_path) and os.path.getsize(gt_path) > 0
    if has_gt:
        ref = parse_ground_truth(gt_path)
        report["n_reference_segments"] = len(ref)
        if ref:
            report["metrics"] = compute_asr_and_speaker(hyp, ref)
        else:
            print(f"[metrics] Ground truth has no parseable segments: {gt_path}", file=sys.stderr)
    else:
        print(f"[metrics] No ground truth at {gt_path} — skipping WER/accuracy", file=sys.stderr)

    if os.path.exists(timings_path):
        with open(timings_path, encoding="utf-8") as f:
            raw = json.load(f)
        report["timing"] = summarize_timings(raw)

    return report


def print_report(report: dict) -> None:
    print("=" * 60)
    print(f"Evaluation: {report.get('consultation')}")
    print("=" * 60)
    print(f"Predicted segments: {report.get('n_predicted_segments', 0)}")
    if "n_reference_segments" in report:
        print(f"Reference segments: {report['n_reference_segments']}")

    m = report.get("metrics")
    if m:
        print("\n--- ASR ---")
        print(f"WER: {_pct(m.get('wer'))}")
        print(f"CER: {_pct(m.get('cer'))}")
        print(f"Ref words / Hyp words: {m.get('ref_words')} / {m.get('hyp_words')}")
        print(
            f"Equal: {m.get('equal', 0)}  "
            f"Subs: {m.get('substitutions', 0)}  "
            f"Dels: {m.get('deletions', 0)}  "
            f"Ins: {m.get('insertions', 0)}"
        )

        print("\n--- Speaker ID ---")
        print(f"Accuracy (word-level, attribution): {_pct(m.get('speaker_accuracy'))}")
        print(
            f"Compared words: {m.get('speaker_compared_words', 0)} "
            f"(correct: {m.get('speaker_correct_words', 0)})"
        )
        for name, s in (m.get("per_speaker") or {}).items():
            print(f"  {name}: acc={_pct(s['accuracy'])} ref_words={s['ref_words']}")

    t = report.get("timing")
    if t:
        print("\n--- Timing ---")
        print(f"Audio duration: {t['audio_duration_s']} s")
        print(f"Wall-clock time: {t['wall_clock_s']} s")
        rtf = t.get("rtf")
        print(f"Overall RTF (wall/audio): {rtf:.3f}" if rtf is not None else "Overall RTF: n/a")
        print(f"Chunks processed: {t['n_chunks']}")
        print(f"ASR mean/max:     {_ms(t['asr_mean_s'])} / {_ms(t['asr_max_s'])}")
        print(f"Speaker mean/max: {_ms(t['speaker_mean_s'])} / {_ms(t['speaker_max_s'])}")
        print(f"End-to-end latency mean/max: {_ms(t['latency_mean_s'])} / {_ms(t['latency_max_s'])}")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consultation", default=os.getenv("CONSULTATION", "consultation1"))
    ap.add_argument("--data-dir", default=os.getenv("DATA_DIR", "./data"))
    ap.add_argument("--json", action="store_true", help="Print report as JSON only")
    args = ap.parse_args()

    report = evaluate(args.consultation, args.data_dir)

    out_dir = os.path.join(args.data_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, f"{args.consultation}_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
        print(f"\nFull report saved: {report_path}")


if __name__ == "__main__":
    main()
