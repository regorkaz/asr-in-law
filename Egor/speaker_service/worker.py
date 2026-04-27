"""Speaker ID Worker — listens to tasks:speaker, identifies via Pyannote embeddings."""

import base64
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
HF_TOKEN = os.getenv("HF_TOKEN", "")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
LOG_DIR = os.getenv("LOG_DIR", "logs")

# Mutable: starts from env, overridden per-run via control:speaker reload message.
current_enrollment_dir: str = os.getenv("ENROLLMENT_DIR", "/data/input/voices")

TASK_QUEUE = "tasks:speaker"
RESULT_QUEUE = "results:speaker"
CONTROL_QUEUE = "control:speaker"

inference_model = None
enrollment_db: dict[str, np.ndarray] = {}


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
    """Scan current enrollment dir for mp3 files; filename stem becomes speaker label."""
    enrollment_dir = Path(current_enrollment_dir)
    if not enrollment_dir.is_dir():
        logging.warning("Enrollment directory not found: %s", current_enrollment_dir)
        return {}

    files: dict[str, str] = {}
    for p in sorted(enrollment_dir.glob("*.mp3")):
        files[p.stem] = str(p)
        logging.info("Discovered enrollment file: %s -> %s", p.stem, p)
    return files


def _build_enrollment_db(files: dict[str, str]) -> dict[str, np.ndarray]:
    db: dict[str, np.ndarray] = {}
    for label, filepath in files.items():
        try:
            emb = inference_model(filepath)
            db[label] = np.mean(emb.data, axis=0)
            logging.info("Enrolled speaker '%s' from %s", label, filepath)
        except Exception as e:
            logging.error("Failed to enroll %s from %s: %s", label, filepath, e)
    return db


def reload_enrollment(new_dir: str | None = None) -> None:
    """Rescan enrollment dir and rebuild speaker DB. Triggered by control:speaker."""
    global enrollment_db, current_enrollment_dir

    if new_dir:
        current_enrollment_dir = new_dir
        logging.info("Enrollment dir switched to: %s", current_enrollment_dir)

    if inference_model is None:
        logging.warning("reload_enrollment called before model init; skipping")
        return

    enrollment_db = _build_enrollment_db(discover_enrollment_files())
    logging.info("Enrollment reloaded: %s", list(enrollment_db.keys()))


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

    enrollment_db = _build_enrollment_db(discover_enrollment_files())
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


def process_task(task: dict) -> dict:
    chunk_id = task.get("chunk_id", "unknown")
    audio_bytes = base64.b64decode(task.get("audio_b64", ""))

    t0 = time.perf_counter()
    speaker, confidence = identify_speaker(audio_bytes)
    return {
        "chunk_id": chunk_id,
        "seq_num": task.get("seq_num"),
        "speaker": speaker,
        "confidence": round(float(confidence), 4),
        "processing_time_s": round(time.perf_counter() - t0, 4),
    }


def main() -> None:
    setup_logging()
    logging.info("Speaker Worker starting")

    init_model()

    r = connect_redis()
    logging.info("Listening on queues: %s, %s", CONTROL_QUEUE, TASK_QUEUE)

    while True:
        try:
            key, raw = r.blpop([CONTROL_QUEUE, TASK_QUEUE])
        except redis.ConnectionError as e:
            logging.error("Redis connection lost: %s. Retrying in 3s...", e)
            time.sleep(3)
            r = connect_redis()
            continue

        if key == CONTROL_QUEUE:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"action": raw}
            action = msg.get("action")
            new_dir = msg.get("enrollment_dir")
            logging.info("Control message: action=%s enrollment_dir=%s", action, new_dir)
            if action == "reload":
                reload_enrollment(new_dir=new_dir)
            continue

        try:
            task = json.loads(raw)
        except json.JSONDecodeError as e:
            logging.error("Bad JSON in task: %s", e)
            continue

        chunk_id = task.get("chunk_id", "unknown")
        seq_num = task.get("seq_num")
        logging.info("Received task: chunk_id=%s", chunk_id)

        try:
            result = process_task(task)
        except Exception as e:
            logging.error("Error processing chunk %s: %s", chunk_id, e)
            result = {
                "chunk_id": chunk_id,
                "seq_num": seq_num,
                "speaker": "Unknown",
                "confidence": 0.0,
                "processing_time_s": 0.0,
                "error": str(e),
            }

        try:
            r.rpush(RESULT_QUEUE, json.dumps(result))
            logging.info("Result pushed: chunk_id=%s", chunk_id)
        except redis.ConnectionError as e:
            logging.error("Redis push failed (%s); reconnecting", e)
            time.sleep(3)
            r = connect_redis()


if __name__ == "__main__":
    main()
