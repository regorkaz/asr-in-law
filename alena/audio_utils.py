from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PCM16_MAX = np.iinfo(np.int16).max

def bytes_to_int16_pcm(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.empty((0,), dtype=np.int16)
    return np.frombuffer(audio_bytes, dtype=np.int16)

def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return audio
    return (audio.astype(np.float32) / float(PCM16_MAX)).astype(np.float32)

def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * float(PCM16_MAX)).astype(np.int16)

def ensure_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=1) if audio.shape[0] > audio.shape[1] else np.mean(audio, axis=0)
    raise ValueError(f'Unsupported audio shape: {audio.shape}')

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    if audio.size == 0:
        return audio.astype(np.float32)
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(audio.astype(np.float32), up, down).astype(np.float32)

def load_audio_file(path: str | Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype='float32', always_2d=False)
    audio = ensure_mono(audio)
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr

def write_wav_file(path: str | Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)

def frame_bytes_to_float32(frame: bytes) -> np.ndarray:
    return int16_to_float32(bytes_to_int16_pcm(frame))

def chunk_audio(audio: np.ndarray, chunk_samples: int) -> Iterator[np.ndarray]:
    audio = np.asarray(audio, dtype=np.float32)
    for start in range(0, len(audio), chunk_samples):
        yield audio[start:start + chunk_samples]

def audio_duration_seconds(num_samples: int, sr: int) -> float:
    return float(num_samples) / float(sr)
