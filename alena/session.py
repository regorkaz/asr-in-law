from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import uuid

import numpy as np

from .config import SETTINGS
from .schemas import TranscriptDocument, Segment, SessionCreateRequest, SessionCreateResponse, EnrollmentResponse, utc_now_iso
from .speaker_id import LawyerSpeakerMatcher
from .streaming import StreamingSessionProcessor, StreamingSpeakerTracker
from .tone_asr import ToneStreamingASR


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    created_at: str
    title: str | None = None
    sample_rate: int = SETTINGS.sample_rate
    speaker_similarity_threshold: float = SETTINGS.speaker_similarity_threshold
    source_mode: str = 'stream'
    lawyer_enrolled: bool = False
    output_dir: Path = field(default_factory=lambda: SETTINGS.output_dir)
    metadata: dict[str, Any] = field(default_factory=dict)

    asr: ToneStreamingASR = field(default_factory=ToneStreamingASR)
    matcher: LawyerSpeakerMatcher = field(init=False)
    processor: StreamingSessionProcessor = field(init=False)
    segments: list[Segment] = field(default_factory=list)
    session_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.matcher = LawyerSpeakerMatcher(threshold=self.speaker_similarity_threshold)
        self.processor = StreamingSessionProcessor(
            session_id=self.session_id,
            asr=self.asr,
            speaker_tracker=StreamingSpeakerTracker(matcher=self.matcher, sample_rate=self.sample_rate),
            sample_rate=self.sample_rate,
        )
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._persist_state()
        self._persist_transcript(finalized=False)

    def create_response(self) -> SessionCreateResponse:
        return SessionCreateResponse(
            session_id=self.session_id,
            created_at=self.created_at,
            sample_rate=self.sample_rate,
            asr_chunk_ms=SETTINGS.asr_chunk_ms,
            vad_frame_ms=SETTINGS.vad_frame_ms,
            speaker_similarity_threshold=self.speaker_similarity_threshold,
            output_dir=str(self.session_dir),
        )

    def enroll_lawyer_from_file(self, file_path: str) -> EnrollmentResponse:
        profile = self.matcher.enroll_from_file(file_path, target_sr=self.sample_rate)
        self.lawyer_enrolled = True
        self._persist_state()
        self._persist_transcript(finalized=False)
        return EnrollmentResponse(
            session_id=self.session_id,
            enrolled=True,
            lawyer_profile_seconds=profile.seconds,
            similarity_threshold=self.speaker_similarity_threshold,
            message=f'Lawyer profile enrolled from file: {file_path}',
        )

    def enroll_lawyer_from_audio(self, audio: np.ndarray, sample_rate: int, source_path: str | None = None) -> EnrollmentResponse:
        profile = self.matcher.enroll_from_audio(audio, sample_rate, source_path=source_path)
        self.lawyer_enrolled = True
        self._persist_state()
        self._persist_transcript(finalized=False)
        return EnrollmentResponse(
            session_id=self.session_id,
            enrolled=True,
            lawyer_profile_seconds=profile.seconds,
            similarity_threshold=self.speaker_similarity_threshold,
            message='Lawyer profile enrolled from audio buffer.',
        )

    def process_audio_bytes(self, audio_bytes: bytes) -> dict[str, Any]:
        events = self.processor.feed_bytes(audio_bytes)
        for item in events['segments']:
            segment = Segment(**item)
            self.segments.append(segment)
            self._append_jsonl('segments.jsonl', segment.model_dump())
        self._persist_state()
        self._persist_transcript(finalized=False)
        return events

    def finalize(self) -> TranscriptDocument:
        events = self.processor.finalize()
        for item in events['segments']:
            segment = Segment(**item)
            self.segments.append(segment)
            self._append_jsonl('segments.jsonl', segment.model_dump())
        self._persist_transcript(finalized=True)
        return self.get_transcript(finalized=True)

    def get_transcript(self, finalized: bool = False) -> TranscriptDocument:
        return TranscriptDocument(
            session_id=self.session_id,
            created_at=self.created_at,
            finalized_at=now_iso() if finalized else None,
            title=self.title,
            sample_rate=self.sample_rate,
            source_mode=self.source_mode,
            speaker_similarity_threshold=self.speaker_similarity_threshold,
            lawyer_enrolled=self.lawyer_enrolled,
            segments=self.segments,
            metadata=self.metadata,
        )

    def _append_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.session_dir / filename
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _persist_state(self) -> None:
        state = {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'title': self.title,
            'sample_rate': self.sample_rate,
            'speaker_similarity_threshold': self.speaker_similarity_threshold,
            'lawyer_enrolled': self.lawyer_enrolled,
            'source_mode': self.source_mode,
            'metadata': self.metadata,
            'segments_count': len(self.segments),
        }
        (self.session_dir / 'session_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

    def _persist_transcript(self, finalized: bool = False) -> None:
        doc = self.get_transcript(finalized=finalized)
        (self.session_dir / 'transcript.json').write_text(json.dumps(doc.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')

    @property
    def transcript_path(self) -> Path:
        return self.session_dir / 'transcript.json'


class SessionManager:
    def __init__(self, output_dir: Path = SETTINGS.output_dir) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, SessionRecord] = {}

    def create_session(self, req: SessionCreateRequest | None = None) -> SessionRecord:
        req = req or SessionCreateRequest()
        session_id = uuid.uuid4().hex[:12]
        threshold = req.speaker_similarity_threshold or SETTINGS.speaker_similarity_threshold
        record = SessionRecord(
            session_id=session_id,
            created_at=now_iso(),
            title=req.title,
            speaker_similarity_threshold=threshold,
            output_dir=self.output_dir,
        )
        self.sessions[session_id] = record
        return record

    def get(self, session_id: str) -> SessionRecord:
        if session_id in self.sessions:
            return self.sessions[session_id]
        session_dir = self.output_dir / session_id
        if not session_dir.exists():
            raise KeyError(session_id)
        state_path = session_dir / 'session_state.json'
        data = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
        record = SessionRecord(
            session_id=session_id,
            created_at=data.get('created_at', now_iso()),
            title=data.get('title'),
            sample_rate=data.get('sample_rate', SETTINGS.sample_rate),
            speaker_similarity_threshold=data.get('speaker_similarity_threshold', SETTINGS.speaker_similarity_threshold),
            source_mode=data.get('source_mode', 'stream'),
            lawyer_enrolled=data.get('lawyer_enrolled', False),
            output_dir=self.output_dir,
            metadata=data.get('metadata', {}),
        )
        transcript_path = session_dir / 'transcript.json'
        if transcript_path.exists():
            transcript_data = json.loads(transcript_path.read_text(encoding='utf-8'))
            record.segments = [Segment(**item) for item in transcript_data.get('segments', [])]
        self.sessions[session_id] = record
        return record
