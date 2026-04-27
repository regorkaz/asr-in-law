"""ASR Worker — listens to tasks:asr queue in Redis and runs GigaAM-v3 CTC."""

import base64
import json
import logging
import os
import tempfile
import time

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOG_DIR = os.getenv("LOG_DIR", "logs")

TASK_QUEUE = "tasks:asr"
RESULT_QUEUE = "results:asr"

asr_model = None


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, "asr_worker.log"), encoding="utf-8"),
        ],
    )


def connect_redis() -> redis.Redis:
    logging.info("Connecting to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    logging.info("Redis connection OK")
    return r


def init_model() -> None:
    global asr_model
    import gigaam

    logging.info("Loading GigaAM-v3 E2E CTC model on cpu...")
    asr_model = gigaam.load_model("v3_e2e_ctc", device="cpu", fp16_encoder=False)
    logging.info("ASR model loaded")


def _to_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    for attr in ("text", "transcription"):
        v = getattr(obj, attr, None)
        if isinstance(v, str):
            return v
    return str(obj)


def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        result = asr_model.transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)

    return _to_text(result)


def process_task(task: dict) -> dict:
    chunk_id = task.get("chunk_id", "unknown")
    audio_bytes = base64.b64decode(task.get("audio_b64", ""))

    t0 = time.perf_counter()
    text = transcribe_audio(audio_bytes)
    return {
        "chunk_id": chunk_id,
        "seq_num": task.get("seq_num"),
        "text": text,
        "language": "ru",
        "processing_time_s": round(time.perf_counter() - t0, 4),
    }


def main() -> None:
    setup_logging()
    logging.info("ASR Worker starting")

    init_model()

    r = connect_redis()
    logging.info("Listening on queue: %s", TASK_QUEUE)

    while True:
        try:
            _, raw = r.blpop(TASK_QUEUE)
        except redis.ConnectionError as e:
            logging.error("Redis connection lost: %s. Retrying in 3s...", e)
            time.sleep(3)
            r = connect_redis()
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
                "text": "",
                "language": "ru",
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
