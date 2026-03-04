"""ASR Worker — listens to tasks:asr queue in Redis."""

import base64
import json
import logging
import os
import tempfile
import time

import redis

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

TASK_QUEUE = "tasks:asr"
RESULT_QUEUE = "results:asr"

LOG_DIR = os.getenv("LOG_DIR", "logs")

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


def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        text = asr_model.transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)

    return text


def process_mock(task: dict) -> dict:
    time.sleep(0.5)
    return {
        "chunk_id": task.get("chunk_id", "unknown"),
        "text": f"Mock transcription for chunk {task.get('chunk_id', '?')}",
        "language": "ru",
        "confidence": 0.95,
    }


def process_real(task: dict) -> dict:
    chunk_id = task.get("chunk_id", "unknown")
    audio_b64 = task.get("audio_b64", "")
    audio_bytes = base64.b64decode(audio_b64)

    text = transcribe_audio(audio_bytes)
    return {
        "chunk_id": chunk_id,
        "text": text,
        "language": "ru",
    }


def main() -> None:
    setup_logging()
    logging.info("ASR Worker starting (MOCK_MODE=%s)", MOCK_MODE)

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
