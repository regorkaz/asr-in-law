"""WER, CER, speaker accuracy via word-level alignment."""

import re
from statistics import mean

import jiwer

from metrics.parse import Segment

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str, fold_yo: bool = True) -> str:
    text = text.lower()
    if fold_yo:
        text = text.replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def _to_tagged_words(segments: list[Segment]) -> list[tuple[str, str]]:
    """Flatten segments into a list of (word, speaker) pairs, preserving order."""
    tagged: list[tuple[str, str]] = []
    for seg in segments:
        for w in normalize(seg.text).split():
            if w:
                tagged.append((w, seg.speaker))
    return tagged


def compute_asr_and_speaker(hyp: list[Segment], ref: list[Segment]) -> dict:
    ref_tagged = _to_tagged_words(ref)
    hyp_tagged = _to_tagged_words(hyp)

    result: dict = {
        "ref_words": len(ref_tagged),
        "hyp_words": len(hyp_tagged),
    }

    if not ref_tagged:
        result.update({"wer": None, "cer": None, "speaker_accuracy": None, "per_speaker": {}})
        return result

    ref_str = " ".join(w for w, _ in ref_tagged)
    hyp_str = " ".join(w for w, _ in hyp_tagged)

    out = jiwer.process_words(ref_str, hyp_str)
    cer = jiwer.cer(ref_str, hyp_str)

    correct_speaker = 0
    compared = 0
    subs = dels = ins = eq = 0
    per_speaker: dict[str, dict[str, int]] = {}

    # jiwer.process_words returns alignments as a list[list[AlignmentChunk]] (one per utterance)
    alignment_lists = out.alignments
    chunks = alignment_lists[0] if alignment_lists else []

    for ch in chunks:
        n_ref = ch.ref_end_idx - ch.ref_start_idx
        n_hyp = ch.hyp_end_idx - ch.hyp_start_idx
        if ch.type == "equal":
            eq += n_ref
            for i in range(n_ref):
                r_spk = ref_tagged[ch.ref_start_idx + i][1]
                h_spk = hyp_tagged[ch.hyp_start_idx + i][1]
                stats = per_speaker.setdefault(r_spk, {"total": 0, "correct": 0})
                stats["total"] += 1
                compared += 1
                if r_spk == h_spk:
                    correct_speaker += 1
                    stats["correct"] += 1
        elif ch.type == "substitute":
            subs += n_ref
            for i in range(n_ref):
                r_spk = ref_tagged[ch.ref_start_idx + i][1]
                h_spk = hyp_tagged[ch.hyp_start_idx + i][1]
                stats = per_speaker.setdefault(r_spk, {"total": 0, "correct": 0})
                stats["total"] += 1
                compared += 1
                if r_spk == h_spk:
                    correct_speaker += 1
                    stats["correct"] += 1
        elif ch.type == "delete":
            dels += n_ref
        elif ch.type == "insert":
            ins += n_hyp

    result.update({
        "wer": out.wer,
        "cer": cer,
        "equal": eq,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "speaker_accuracy": correct_speaker / compared if compared else None,
        "speaker_compared_words": compared,
        "speaker_correct_words": correct_speaker,
        "per_speaker": {
            spk: {
                "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
                "ref_words": v["total"],
            }
            for spk, v in per_speaker.items()
        },
    })
    return result


def summarize_timings(timings: dict) -> dict:
    chunks = timings.get("chunks", [])
    if not chunks:
        return {}

    asr_times = [c["asr_processing_s"] for c in chunks if c.get("asr_processing_s") is not None]
    spk_times = [c["speaker_processing_s"] for c in chunks if c.get("speaker_processing_s") is not None]
    e2e = [c["end_to_end_latency_s"] for c in chunks if c.get("end_to_end_latency_s") is not None]
    durations = [c["duration_s"] for c in chunks if c.get("duration_s") is not None]

    audio = timings.get("audio_duration_s") or 0.0
    wall = timings.get("wall_clock_s") or 0.0

    return {
        "audio_duration_s": round(audio, 2),
        "wall_clock_s": round(wall, 2),
        "rtf": (wall / audio) if audio else None,
        "n_chunks": len(chunks),
        "asr_mean_s": mean(asr_times) if asr_times else None,
        "asr_max_s": max(asr_times) if asr_times else None,
        "speaker_mean_s": mean(spk_times) if spk_times else None,
        "speaker_max_s": max(spk_times) if spk_times else None,
        "latency_mean_s": mean(e2e) if e2e else None,
        "latency_max_s": max(e2e) if e2e else None,
        "chunk_duration_mean_s": mean(durations) if durations else None,
    }
