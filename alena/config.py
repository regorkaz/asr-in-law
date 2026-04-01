from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass(slots=True)
class Settings:
    sample_rate: int = 16_000
    vad_frame_ms: int = 20
    asr_chunk_ms: int = 300
    vad_aggressiveness: int = 2
    silence_padding_ms: int = 500
    min_speech_ms: int = 300
    speaker_guess_update_ms: int = 600
    speaker_similarity_threshold: float = 0.72
    output_dir: Path = field(default_factory=lambda: Path(os.getenv('LEGAL_ASR_OUTPUT_DIR', 'output')))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('LEGAL_ASR_DATA_DIR', 'data')))

SETTINGS = Settings()
