"""Gateway Simulator — streams audio from file through VAD, sends chunks to Redis, collects results."""

import json
import logging
import os
import sys

import redis

from gateway.pipeline import StreamingPipeline
from gateway.sources.file_source import FileAudioSource
from metrics.evaluate import evaluate, print_report

CONSULTATION = os.getenv("CONSULTATION", "consultation1")
DATA_DIR = os.getenv("DATA_DIR", "./data")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REALTIME_FACTOR = float(os.getenv("REALTIME_FACTOR", "0.0"))
RUN_METRICS = os.getenv("RUN_METRICS", "true").lower() == "true"
STREAMING_VAD = os.getenv("STREAMING_VAD", "false").lower() == "true"
LOG_DIR = os.getenv("LOG_DIR", "logs")

# Path to consultation voices as seen by speaker_worker (inside its container).
# speaker_worker mounts ./data at /data, so consultation voices live at /data/input/<consultation>/voices.
SPEAKER_DATA_PREFIX = os.getenv("SPEAKER_DATA_PREFIX", "/data/input")

INPUT_DIR = os.path.join(DATA_DIR, "input", CONSULTATION)
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

AUDIO_FILENAME = "audio.mp3"

REDIS_QUEUES = ("tasks:asr", "tasks:speaker", "results:asr", "results:speaker", "control:speaker")


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
    if not os.path.isdir(INPUT_DIR):
        logging.error("Consultation directory not found: %s", INPUT_DIR)
        sys.exit(1)

    path = os.path.join(INPUT_DIR, AUDIO_FILENAME)
    if not os.path.isfile(path):
        logging.error("Expected %s not found in %s", AUDIO_FILENAME, INPUT_DIR)
        sys.exit(1)
    return path


def request_voices_reload(r: redis.Redis) -> None:
    """Tell speaker_worker to load voices from this consultation's voices dir."""
    voices_dir = f"{SPEAKER_DATA_PREFIX}/{CONSULTATION}/voices"
    msg = {"action": "reload", "enrollment_dir": voices_dir}
    r.rpush("control:speaker", json.dumps(msg))
    logging.info("Sent control:speaker reload -> %s", voices_dir)


def main() -> None:
    setup_logging()
    logging.info("=== Gateway Simulator (consultation=%s) ===", CONSULTATION)

    r = connect_redis()
    r.delete(*REDIS_QUEUES)
    logging.info("Cleared queues: %s", ", ".join(REDIS_QUEUES))

    request_voices_reload(r)

    filepath = find_input_file()
    source = FileAudioSource(filepath, realtime_factor=REALTIME_FACTOR)

    output_path = os.path.join(OUTPUT_DIR, f"{CONSULTATION}_transcript.txt")
    timings_path = os.path.join(OUTPUT_DIR, f"{CONSULTATION}_timings.json")

    pipeline = StreamingPipeline(
        source=source,
        redis_client=r,
        output_path=output_path,
        timings_path=timings_path,
        streaming_vad=STREAMING_VAD,
    )
    pipeline.run()

    if RUN_METRICS:
        try:
            report = evaluate(CONSULTATION, DATA_DIR)
            print_report(report)
            report_path = os.path.join(OUTPUT_DIR, f"{CONSULTATION}_metrics.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logging.info("Metrics report saved: %s", report_path)
        except Exception as e:
            logging.warning("Metrics evaluation failed: %s", e)

    logging.info("=== Done ===")


if __name__ == "__main__":
    main()
