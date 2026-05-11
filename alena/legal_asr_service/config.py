from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass(slots=True)
class Settings:
    sample_rate: int = 16_000
    asr_sample_rate: int = 8_000

    vad_frame_ms: int = 20
    asr_chunk_ms: int = 300
    vad_aggressiveness: int = _env_int("LEGAL_ASR_VAD_AGGRESSIVENESS", 2)

    silence_padding_ms: int = _env_int("LEGAL_ASR_SILENCE_PADDING_MS", 500)
    min_speech_ms: int = _env_int("LEGAL_ASR_MIN_SPEECH_MS", 300)

    enable_speaker_id: bool = _env_bool("LEGAL_ASR_ENABLE_SPEAKER_ID", True)
    speaker_guess_update_ms: int = _env_int("LEGAL_ASR_SPEAKER_GUESS_UPDATE_MS", 2_000)
    speaker_embedding_window_ms: int = _env_int("LEGAL_ASR_SPEAKER_EMBEDDING_WINDOW_MS", 3_000)
    max_speaker_turn_ms: int = _env_int("LEGAL_ASR_MAX_SPEAKER_TURN_MS", 10_000)
    speaker_similarity_threshold: float = _env_float("LEGAL_ASR_SPEAKER_THRESHOLD", 0.35)

    tone_decoder_type: str = os.getenv("LEGAL_ASR_TONE_DECODER", "beam_search")

    # T-one splitter settings
    tone_max_phrase_duration_ms: int = _env_int("LEGAL_ASR_TONE_MAX_PHRASE_DURATION_MS", 30000)
    tone_min_silence_duration_ms: int = _env_int("LEGAL_ASR_TONE_MIN_SILENCE_DURATION_MS", 600)
    tone_silence_threshold: float = _env_float("LEGAL_ASR_TONE_SILENCE_THRESHOLD", 0.9)

    persist_interval_sec: float = _env_float("LEGAL_ASR_PERSIST_INTERVAL_SEC", 5.0)

    output_dir: Path = field(default_factory=lambda: Path(os.getenv('LEGAL_ASR_OUTPUT_DIR', 'output')))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('LEGAL_ASR_DATA_DIR', 'data')))

SETTINGS = Settings()
