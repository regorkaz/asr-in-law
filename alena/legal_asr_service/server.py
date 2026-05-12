from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from typing import Any
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from pathlib import Path

from .audio_utils import is_supported_audio_path
from .config import SETTINGS
from .schemas import SessionCreateRequest
from .session import SessionManager


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

manager = SessionManager(output_dir=SETTINGS.output_dir)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Preloading ASR/speaker models...")
    await asyncio.to_thread(manager.preload_models)
    logger.info("Models are ready.")
    yield
    logger.info("ASR service shutdown.")
app = FastAPI(
    title='Legal Real-Time ASR Service',
    version='0.1.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/")
def root():
    return RedirectResponse(url="/ui")

app.mount(
    "/ui",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="ui",
)
@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}

@app.post('/sessions')
def create_session(req: SessionCreateRequest | None = None) -> dict[str, Any]:
    record = manager.create_session(req)
    return record.create_response().model_dump()

@app.get('/sessions/{session_id}')
def get_session(session_id: str) -> dict[str, Any]:
    try:
        record = manager.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail='Session not found')
    return {
        'session_id': record.session_id,
        'created_at': record.created_at,
        'title': record.title,
        'lawyer_enrolled': record.lawyer_enrolled,
        'speaker_similarity_threshold': record.speaker_similarity_threshold,
        'segments_count': len(record.segments),
        'transcript_path': str(record.transcript_path),
    }

@app.post('/sessions/{session_id}/enroll')
async def enroll_lawyer(session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        record = manager.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail='Session not found')
    upload_dir = record.session_dir / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    if not is_supported_audio_path(file_path):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_path.suffix}. "
                   "Supported formats: wav, flac, ogg, mp3, m4a, aac, mp4, webm, wma."
        )
    file_path.write_bytes(await file.read())
    return record.enroll_lawyer_from_file(str(file_path)).model_dump()

@app.get('/sessions/{session_id}/transcript')
def get_transcript(session_id: str) -> dict[str, Any]:
    try:
        record = manager.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail='Session not found')
    path = record.transcript_path
    if not path.exists():
        raise HTTPException(status_code=404, detail='Transcript not found')
    return json.loads(path.read_text(encoding='utf-8'))

@app.get('/sessions/{session_id}/transcript-file')
def download_transcript_file(session_id: str):
    try:
        record = manager.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail='Session not found')
    path = record.transcript_path
    if not path.exists():
        raise HTTPException(status_code=404, detail='Transcript not found')
    return FileResponse(path)

@app.post('/sessions/{session_id}/finalize')
def finalize_session(session_id: str) -> dict[str, Any]:
    try:
        record = manager.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail='Session not found')
    return record.finalize().model_dump()

@app.websocket('/ws/audio/{session_id}')
async def ws_audio(session_id: str, websocket: WebSocket):
    await websocket.accept()
    try:
        record = manager.get(session_id)
    except KeyError:
        await websocket.send_json({
            'event': 'error',
            'session_id': session_id,
            'payload': {'message': 'Session not found'}
        })
        await websocket.close()
        return

    finalized = False

    await websocket.send_json({
        'event': 'info',
        'session_id': session_id,
        'payload': {
            'message': 'Send raw PCM16 mono audio frames at 16 kHz. Use a client that transmits 20 ms frames.',
            'sample_rate': SETTINGS.sample_rate,
            'vad_frame_ms': SETTINGS.vad_frame_ms,
            'asr_chunk_ms': SETTINGS.asr_chunk_ms,
        },
    })

    try:
        while True:
            try:
                message = await websocket.receive()
            except RuntimeError:
                break

            if message.get('type') == 'websocket.disconnect':
                break

            if message.get("bytes") is not None:
                data = message["bytes"]

                try:
                    t0 = time.perf_counter()
                    events = await asyncio.to_thread(record.process_audio_bytes, data)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    if elapsed_ms > 500:
                        logger.warning(
                            "Slow audio processing: session=%s bytes=%d elapsed_ms=%.1f",
                            session_id,
                            len(data),
                            elapsed_ms,
                        )

                except Exception as exc:
                    logger.error(
                        "Audio processing failed: session=%s error=%s\n%s",
                        session_id,
                        repr(exc),
                        traceback.format_exc(),
                    )
                    await websocket.send_json({
                        "event": "error",
                        "session_id": session_id,
                        "payload": {
                            "message": "Audio processing failed",
                            "error": repr(exc),
                        },
                    })
                    await websocket.close(code=1011)
                    return

                if events["segments"] or events["speaker_turns"]:
                    await websocket.send_json({
                        "event": "partial",
                        "session_id": session_id,
                        "payload": events,
                    })
                continue

            if message.get('text') is not None:
                try:
                    obj = json.loads(message['text'])
                except json.JSONDecodeError:
                    await websocket.send_json({
                        'event': 'error',
                        'session_id': session_id,
                        'payload': {'message': 'Invalid JSON control message'}
                    })
                    continue

                if obj.get('type') == 'finalize':
                    #transcript = record.finalize()
                    transcript = await asyncio.to_thread(record.finalize)
                    finalized = True
                    await websocket.send_json({
                        'event': 'final',
                        'session_id': session_id,
                        'payload': transcript.model_dump()
                    })
                    await websocket.close()
                    return

    except WebSocketDisconnect:
        pass
    finally:
        if not finalized:
            try:
                await asyncio.to_thread(record.finalize)
            except Exception:
                logger.exception("Failed to finalize session=%s after websocket close", session_id)
