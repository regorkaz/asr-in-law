from __future__ import annotations
import json
from typing import Any
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import SETTINGS
from .schemas import SessionCreateRequest
from .session import SessionManager

app = FastAPI(title='Legal Real-Time ASR Service', version='0.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

manager = SessionManager(output_dir=SETTINGS.output_dir)

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
        await websocket.send_json({'event': 'error', 'session_id': session_id, 'payload': {'message': 'Session not found'}})
        await websocket.close()
        return

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
            message = await websocket.receive()
            if message.get('bytes') is not None:
                events = record.process_audio_bytes(message['bytes'])
                await websocket.send_json({'event': 'partial', 'session_id': session_id, 'payload': events})
                continue
            if message.get('text') is not None:
                try:
                    obj = json.loads(message['text'])
                except json.JSONDecodeError:
                    await websocket.send_json({'event': 'error', 'session_id': session_id, 'payload': {'message': 'Invalid JSON control message'}})
                    continue
                if obj.get('type') == 'finalize':
                    transcript = record.finalize()
                    await websocket.send_json({'event': 'final', 'session_id': session_id, 'payload': transcript.model_dump()})
                    await websocket.close()
                    return
    except WebSocketDisconnect:
        pass
    finally:
        try:
            record.finalize()
        except Exception:
            pass
