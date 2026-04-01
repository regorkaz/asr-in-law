from __future__ import annotations
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import websocket

from .audio_utils import load_audio_file, float32_to_int16
from .config import SETTINGS


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='legal_asr_service')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('serve')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)

    p = sub.add_parser('create-session')
    p.add_argument('--api', default='http://localhost:8000')
    p.add_argument('--title', default=None)
    p.add_argument('--threshold', type=float, default=None)

    p = sub.add_parser('enroll')
    p.add_argument('--api', default='http://localhost:8000')
    p.add_argument('--session-id', required=True)
    p.add_argument('--file', required=True)

    p = sub.add_parser('record-enroll')
    p.add_argument('--api', default='http://localhost:8000')
    p.add_argument('--session-id', required=True)
    p.add_argument('--seconds', type=float, default=45.0)
    p.add_argument('--output', default='lawyer_enrollment.wav')

    p = sub.add_parser('stream-file')
    p.add_argument('--api', default='http://localhost:8000')
    p.add_argument('--session-id', required=True)
    p.add_argument('--file', required=True)
    p.add_argument('--block-ms', type=int, default=20)

    p = sub.add_parser('stream-mic')
    p.add_argument('--api', default='http://localhost:8000')
    p.add_argument('--session-id', required=True)
    p.add_argument('--block-ms', type=int, default=20)

    p = sub.add_parser('demo')
    p.add_argument('--api', default='http://localhost:8000')
    p.add_argument('--lawyer-file', required=True)
    p.add_argument('--consultation-file', required=True)
    p.add_argument('--title', default='Legal consultation demo')
    p.add_argument('--threshold', type=float, default=None)

    return parser


def api_post(api: str, path: str, **kwargs):
    url = api.rstrip('/') + path
    resp = requests.post(url, timeout=600, **kwargs)
    resp.raise_for_status()
    return resp.json()


def api_get(api: str, path: str):
    url = api.rstrip('/') + path
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    return resp.json()


def cmd_create_session(args) -> None:
    data = api_post(args.api, '/sessions', json={'title': args.title, 'speaker_similarity_threshold': args.threshold})
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_enroll(args) -> None:
    file_path = Path(args.file)
    with file_path.open('rb') as f:
        files = {'file': (file_path.name, f, 'application/octet-stream')}
        data = api_post(args.api, f'/sessions/{args.session_id}/enroll', files=files)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_record_enroll(args) -> None:
    sr = SETTINGS.sample_rate
    print(f'Recording {args.seconds:.1f} seconds at {sr} Hz. Speak clearly.')
    audio = sd.rec(int(args.seconds * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = audio.squeeze()
    sf.write(args.output, audio, sr)
    print(f'Saved enrollment audio to {args.output}')
    cmd_enroll(argparse.Namespace(api=args.api, session_id=args.session_id, file=args.output))


def _stream_pcm16(ws, audio: np.ndarray, sr: int, block_ms: int) -> None:
    block_samples = int(sr * block_ms / 1000)
    block_samples = max(1, block_samples)
    pcm = float32_to_int16(audio).tobytes()
    frame_bytes = block_samples * 2
    for idx in range(0, len(pcm), frame_bytes):
        frame = pcm[idx: idx + frame_bytes]
        if not frame:
            break
        ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
        time.sleep(block_ms / 1000.0)


def _ws_url(api: str, session_id: str) -> str:
    return api.replace('http://', 'ws://').replace('https://', 'wss://').rstrip('/') + f'/ws/audio/{session_id}'


def cmd_stream_file(args) -> None:
    audio, sr = load_audio_file(args.file, target_sr=SETTINGS.sample_rate)
    ws = websocket.WebSocket()
    ws.connect(_ws_url(args.api, args.session_id))
    try:
        _stream_pcm16(ws, audio, sr, args.block_ms)
        ws.send(json.dumps({'type': 'finalize'}))
        while True:
            try:
                msg = ws.recv()
            except Exception:
                break
            if not msg:
                break
            print(msg)
    finally:
        ws.close()


def cmd_stream_mic(args) -> None:
    sr = SETTINGS.sample_rate
    block_samples = int(sr * args.block_ms / 1000)
    ws = websocket.WebSocket()
    ws.connect(_ws_url(args.api, args.session_id))
    q: 'queue.Queue[bytes | None]' = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    def sender():
        while True:
            data = q.get()
            if data is None:
                return
            ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
            try:
                ws.settimeout(0.01)
                while True:
                    msg = ws.recv()
                    if not msg:
                        break
                    print(msg)
            except Exception:
                pass

    t = threading.Thread(target=sender, daemon=True)
    t.start()
    print('Speak into the microphone. Press Ctrl+C to stop.')
    try:
        with sd.RawInputStream(samplerate=sr, channels=1, dtype='int16', blocksize=block_samples, callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ws.send(json.dumps({'type': 'finalize'}))
        except Exception:
            pass
        q.put(None)
        t.join(timeout=1.0)
        ws.close()


def cmd_demo(args) -> None:
    session = api_post(args.api, '/sessions', json={'title': args.title, 'speaker_similarity_threshold': args.threshold})
    session_id = session['session_id']
    print('Created session:', session_id)

    with open(args.lawyer_file, 'rb') as f:
        files = {'file': (Path(args.lawyer_file).name, f, 'application/octet-stream')}
        enroll = api_post(args.api, f'/sessions/{session_id}/enroll', files=files)
    print('Enrolled:', enroll)

    audio, sr = load_audio_file(args.consultation_file, target_sr=SETTINGS.sample_rate)
    ws = websocket.WebSocket()
    ws.connect(_ws_url(args.api, session_id))
    try:
        _stream_pcm16(ws, audio, sr, 20)
        ws.send(json.dumps({'type': 'finalize'}))
        while True:
            try:
                msg = ws.recv()
            except Exception:
                break
            if not msg:
                break
            print(msg)
    finally:
        ws.close()

    transcript = api_get(args.api, f'/sessions/{session_id}/transcript')
    print('\nFinal transcript:')
    print(json.dumps(transcript, ensure_ascii=False, indent=2))



def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.command == 'serve':
        import uvicorn
        uvicorn.run('legal_asr_service.server:app', host=args.host, port=args.port, reload=False)
    elif args.command == 'create-session':
        cmd_create_session(args)
    elif args.command == 'enroll':
        cmd_enroll(args)
    elif args.command == 'record-enroll':
        cmd_record_enroll(args)
    elif args.command == 'stream-file':
        cmd_stream_file(args)
    elif args.command == 'stream-mic':
        cmd_stream_mic(args)
    elif args.command == 'demo':
        cmd_demo(args)
    else:
        parser.error('Unknown command')

if __name__ == '__main__':
    main()
