"""Speaker ID Worker — listens to tasks:speaker queue in Redis."""

import base64
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import redis

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENROLLMENT_DIR = os.getenv("ENROLLMENT_DIR", "/data/input/voices")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))

TASK_QUEUE = "tasks:speaker"
RESULT_QUEUE = "results:speaker"

LOG_DIR = os.getenv("LOG_DIR", "logs")

inference_model = None
enrollment_db: dict[str, object] = {}


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, "speaker_worker.log"), encoding="utf-8"),
        ],
    )


def discover_enrollment_files() -> dict[str, str]:
    """Scan ENROLLMENT_DIR for .mp3 files; filename stem becomes speaker label."""
    enrollment_dir = Path(ENROLLMENT_DIR)
    if not enrollment_dir.is_dir():
        logging.warning("Enrollment directory not found: %s", ENROLLMENT_DIR)
        return {}

    files: dict[str, str] = {}
    for p in sorted(enrollment_dir.glob("*.mp3")):
        label = p.stem
        files[label] = str(p)
        logging.info("Discovered enrollment file: %s -> %s", label, p)

    return files


def connect_redis() -> redis.Redis:
    logging.info("Connecting to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    logging.info("Redis connection OK")
    return r


def init_model() -> None:
    global inference_model, enrollment_db

    import torch
    from pyannote.audio import Inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Loading Pyannote embedding model on %s...", device)

    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN

    inference_model = Inference("pyannote/embedding", device=device)
    logging.info("Pyannote embedding model loaded")

    enrollment_files = discover_enrollment_files()
    for label, filepath in enrollment_files.items():
        emb = inference_model(filepath)
        enrollment_db[label] = np.mean(emb.data, axis=0)
        logging.info("Enrolled speaker '%s' from %s", label, filepath)

    if enrollment_db:
        logging.info("Speaker enrollment complete: %s", list(enrollment_db.keys()))
    else:
        logging.warning("No speakers enrolled — all results will be 'Unknown'")


def identify_speaker(audio_bytes: bytes) -> tuple[str, float]:
    from scipy.spatial.distance import cosine

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        raw_emb = inference_model(tmp_path)
        chunk_emb = np.mean(raw_emb.data, axis=0)
    finally:
        os.unlink(tmp_path)

    best_label = "Unknown"
    best_score = -1.0

    for label, ref_emb in enrollment_db.items():
        similarity = 1.0 - cosine(chunk_emb, ref_emb)
        logging.debug("  %s: similarity=%.4f", label, similarity)
        if similarity > best_score:
            best_score = similarity
            best_label = label

    if best_score < SIMILARITY_THRESHOLD:
        return "Unknown", best_score

    return best_label, best_score


def process_mock(task: dict) -> dict:
    time.sleep(0.5)
    chunk_id = task.get("chunk_id", 0)
    mock_speakers = list(discover_enrollment_files().keys()) or ["Lawyer", "Client"]
    speaker = mock_speakers[hash(str(chunk_id)) % len(mock_speakers)]
    return {
        "chunk_id": chunk_id,
        "speaker": speaker,
        "confidence": 0.92,
    }


def process_real(task: dict) -> dict:
    chunk_id = task.get("chunk_id", "unknown")
    audio_b64 = task.get("audio_b64", "")
    audio_bytes = base64.b64decode(audio_b64)

    speaker, confidence = identify_speaker(audio_bytes)
    return {
        "chunk_id": chunk_id,
        "speaker": speaker,
        "confidence": round(float(confidence), 4),
    }


def main() -> None:
    setup_logging()
    logging.info("Speaker Worker starting (MOCK_MODE=%s)", MOCK_MODE)

    if not MOCK_MODE:
        init_model()

    r = connect_redis()
    logging.info("Listening on queue: %s", TASK_QUEUE)

    while True:
        try:
            _, raw = r.blpop(TASK_QUEUE)
            task = json.loads(raw)
            chunk_id = task.get("chunk_id", "unknown")
            logging.info("Received task: chunk_id=%s", chunk_id)

            if MOCK_MODE:
                result = process_mock(task)
            else:
                result = process_real(task)

            r.rpush(RESULT_QUEUE, json.dumps(result))
            logging.info("Result pushed: chunk_id=%s", chunk_id)

        except json.JSONDecodeError as e:
            logging.error("Bad JSON in task: %s", e)
        except redis.ConnectionError as e:
            logging.error("Redis connection lost: %s. Retrying in 3s...", e)
            time.sleep(3)
            r = connect_redis()
        except Exception as e:
            logging.error("Error processing chunk: %s", e)


if __name__ == "__main__":
    main()
