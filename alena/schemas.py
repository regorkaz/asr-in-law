from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

SpeakerLabel = Literal['LAWYER', 'CLIENT', 'UNKNOWN']

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class SessionCreateRequest(BaseModel):
    title: Optional[str] = None
    speaker_similarity_threshold: float | None = None

class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: str
    sample_rate: int
    asr_chunk_ms: int
    vad_frame_ms: int
    speaker_similarity_threshold: float
    output_dir: str

class EnrollmentResponse(BaseModel):
    session_id: str
    enrolled: bool
    lawyer_profile_seconds: float
    similarity_threshold: float
    message: str

class Segment(BaseModel):
    segment_id: int
    start_time: float
    end_time: float
    speaker: SpeakerLabel = 'UNKNOWN'
    speaker_confidence: float | None = None
    speaker_similarity: float | None = None
    text: str
    asr_confidence: float | None = None
    raw_phrase: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class TranscriptDocument(BaseModel):
    session_id: str
    created_at: str
    finalized_at: str | None = None
    title: str | None = None
    sample_rate: int
    source_mode: str = 'stream'
    speaker_similarity_threshold: float
    lawyer_enrolled: bool = False
    segments: list[Segment] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

class StreamEvent(BaseModel):
    event: Literal['partial', 'segment', 'info', 'final', 'error']
    session_id: str
    timestamp: str = Field(default_factory=utc_now_iso)
    payload: dict[str, Any] = Field(default_factory=dict)
