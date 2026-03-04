"""Gateway Simulator — reads MP3, splits on silence (VAD), sends chunks to Redis, collects results."""

import base64
import json
import logging
import os
import sys
import time

import redis
from pydub import AudioSegment
from pydub.silence import split_on_silence

CONSULTATION = os.getenv("CONSULTATION", "consultation1")
DATA_DIR = os.getenv("DATA_DIR", "./data")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
RESULT_TIMEOUT = 120

MIN_SILENCE_LEN = 500
SILENCE_THRESH = -40
KEEP_SILENCE = 200
MIN_CHUNK_LEN_MS = 1000

INPUT_DIR = os.path.join(DATA_DIR, "input", CONSULTATION)
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
LOG_DIR = "logs"


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{CONSULTATION}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def connect_redis() -> redis.Redis:
    logging.info("Connecting to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    logging.info("Redis connection OK")
    return r


def find_input_file() -> str:
    for f in os.listdir(INPUT_DIR):
        if f.endswith(".mp3"):
            return os.path.join(INPUT_DIR, f)
    logging.error("No .mp3 files found in %s", INPUT_DIR)
    sys.exit(1)


def split_audio_on_silence(filepath: str) -> list[tuple[str, bytes]]:
    logging.info("Loading audio: %s", filepath)
    audio = AudioSegment.from_mp3(filepath)
    logging.info("Audio duration: %dms", len(audio))

    raw_chunks = split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
        keep_silence=KEEP_SILENCE,
    )
    logging.info("Raw split produced %d segments", len(raw_chunks))

    if not raw_chunks:
        logging.warning("No silence boundaries found — sending entire audio as one chunk")
        raw_chunks = [audio]

    merged: list[AudioSegment] = []
    for seg in raw_chunks:
        if merged and len(seg) < MIN_CHUNK_LEN_MS:
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)

    if len(merged) > 1 and len(merged[-1]) < MIN_CHUNK_LEN_MS:
        merged[-2] = merged[-2] + merged[-1]
        merged.pop()

    results: list[tuple[str, bytes]] = []
    offset_ms = 0
    for seg in merged:
        end_ms = offset_ms + len(seg)
        start_s = round(offset_ms / 1000, 1)
        end_s = round(end_ms / 1000, 1)
        chunk_id = f"chunk_{start_s}s_{end_s}s"
        wav_bytes = seg.export(format="wav").read()
        results.append((chunk_id, wav_bytes))
        offset_ms = end_ms

    logging.info("Final chunk count: %d", len(results))
    return results


def send_chunks(r: redis.Redis, chunks: list[tuple[str, bytes]]) -> int:
    for chunk_id, chunk_bytes in chunks:
        encoded = base64.b64encode(chunk_bytes).decode("ascii")
        task = json.dumps({"chunk_id": chunk_id, "audio_b64": encoded})
        r.rpush("tasks:asr", task)
        r.rpush("tasks:speaker", task)
    logging.info("Sent %d chunks to workers", len(chunks))
    return len(chunks)


def collect_results(r: redis.Redis, num_chunks: int) -> tuple[list[dict], list[dict]]:
    asr_results: list[dict] = []
    speaker_results: list[dict] = []

    for i in range(num_chunks):
        logging.info("Waiting for ASR result %d/%d...", i + 1, num_chunks)
        resp = r.blpop("results:asr", timeout=RESULT_TIMEOUT)
        if resp is None:
            logging.error("Timeout waiting for ASR result %d", i)
            continue
        asr_results.append(json.loads(resp[1]))

    for i in range(num_chunks):
        logging.info("Waiting for Speaker result %d/%d...", i + 1, num_chunks)
        resp = r.blpop("results:speaker", timeout=RESULT_TIMEOUT)
        if resp is None:
            logging.error("Timeout waiting for Speaker result %d", i)
            continue
        speaker_results.append(json.loads(resp[1]))

    return asr_results, speaker_results


def _chunk_sort_key(chunk_id: str) -> float:
    try:
        parts = chunk_id.replace("chunk_", "").split("s_")
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def write_transcript(asr_results: list[dict], speaker_results: list[dict]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{CONSULTATION}_transcript.txt")

    speaker_map: dict[str, str] = {}
    for sr in speaker_results:
        speaker_map[str(sr.get("chunk_id", ""))] = sr.get("speaker", "Unknown")

    sorted_asr = sorted(asr_results, key=lambda r: _chunk_sort_key(str(r.get("chunk_id", ""))))

    lines: list[str] = []
    for ar in sorted_asr:
        chunk_id = str(ar.get("chunk_id", ""))
        speaker = speaker_map.get(chunk_id, "Unknown")
        text = ar.get("text", "")
        lines.append(f"[{speaker}] ({chunk_id}): {text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logging.info("Transcript written to %s", output_path)


def main() -> None:
    setup_logging()
    logging.info("=== Gateway Simulator (consultation=%s) ===", CONSULTATION)

    r = connect_redis()
    r.flushall()
    logging.info("Redis flushed")

    filepath = find_input_file()
    chunks = split_audio_on_silence(filepath)
    num_sent = send_chunks(r, chunks)
    asr_results, speaker_results = collect_results(r, num_sent)

    write_transcript(asr_results, speaker_results)
    logging.info("=== Done ===")


if __name__ == "__main__":
    main()
