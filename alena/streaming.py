from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import webrtcvad

from .audio_utils import bytes_to_int16_pcm, int16_to_float32
from .config import SETTINGS
from .speaker_id import LawyerSpeakerMatcher, SpeakerDecision
from .tone_asr import ToneStreamingASR, NormalizedPhrase


@dataclass(slots=True)
class SpeakerTurn:
    start_time: float
    end_time: float
    label: str
    confidence: float
    similarity: float
    audio: np.ndarray
    sample_rate: int


@dataclass(slots=True)
class ActiveTurn:
    start_frame_index: int
    speech_frames: list[bytes] = field(default_factory=list)
    speech_frame_count: int = 0
    silence_frame_count: int = 0
    current_guess: SpeakerDecision | None = None


class StreamingSpeakerTracker:
    def __init__(
        self,
        matcher: LawyerSpeakerMatcher,
        sample_rate: int = SETTINGS.sample_rate,
        frame_ms: int = SETTINGS.vad_frame_ms,
        silence_padding_ms: int = SETTINGS.silence_padding_ms,
        min_speech_ms: int = SETTINGS.min_speech_ms,
        guess_update_ms: int = SETTINGS.speaker_guess_update_ms,
    ) -> None:
        self.matcher = matcher
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.vad = webrtcvad.Vad(SETTINGS.vad_aggressiveness)
        self.silence_padding_frames = max(1, silence_padding_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.guess_update_frames = max(1, guess_update_ms // frame_ms)
        self.current: ActiveTurn | None = None
        self.finished_turns: list[SpeakerTurn] = []
        self.frame_index = 0

    def _speaker_guess(self, speech_frames: list[bytes]) -> SpeakerDecision:
        if not speech_frames:
            return SpeakerDecision(label='UNKNOWN', similarity=0.0, confidence=0.0)
        audio = bytes_to_int16_pcm(b''.join(speech_frames)).astype(np.int16)
        audio_f32 = int16_to_float32(audio)
        return self.matcher.compare(audio_f32, self.sample_rate)

    def process_frame(self, frame: bytes) -> SpeakerTurn | None:
        is_speech = self.vad.is_speech(frame, self.sample_rate)
        self.frame_index += 1

        if is_speech:
            if self.current is None:
                self.current = ActiveTurn(start_frame_index=self.frame_index)
            self.current.speech_frames.append(frame)
            self.current.speech_frame_count += 1
            self.current.silence_frame_count = 0
            if self.current.speech_frame_count >= self.min_speech_frames and self.current.speech_frame_count % self.guess_update_frames == 0:
                self.current.current_guess = self._speaker_guess(self.current.speech_frames)
            return None

        if self.current is None:
            return None

        self.current.silence_frame_count += 1
        if self.current.silence_frame_count >= self.silence_padding_frames:
            speech_frames = self.current.speech_frames[:]
            start_time = (self.current.start_frame_index - 1) * self.frame_ms / 1000.0
            end_time = self.frame_index * self.frame_ms / 1000.0
            guess = self.current.current_guess or self._speaker_guess(speech_frames)
            audio = bytes_to_int16_pcm(b''.join(speech_frames)).astype(np.int16)
            turn = SpeakerTurn(
                start_time=start_time,
                end_time=end_time,
                label=guess.label if guess.label in {'LAWYER', 'CLIENT'} else 'UNKNOWN',
                confidence=guess.confidence,
                similarity=guess.similarity,
                audio=audio,
                sample_rate=self.sample_rate,
            )
            self.finished_turns.append(turn)
            self.current = None
            return turn

        return None

    def flush(self) -> SpeakerTurn | None:
        if self.current is None or not self.current.speech_frames:
            self.current = None
            return None
        speech_frames = self.current.speech_frames[:]
        start_time = (self.current.start_frame_index - 1) * self.frame_ms / 1000.0
        end_time = self.frame_index * self.frame_ms / 1000.0
        guess = self.current.current_guess or self._speaker_guess(speech_frames)
        audio = bytes_to_int16_pcm(b''.join(speech_frames)).astype(np.int16)
        turn = SpeakerTurn(
            start_time=start_time,
            end_time=end_time,
            label=guess.label if guess.label in {'LAWYER', 'CLIENT'} else 'UNKNOWN',
            confidence=guess.confidence,
            similarity=guess.similarity,
            audio=audio,
            sample_rate=self.sample_rate,
        )
        self.finished_turns.append(turn)
        self.current = None
        return turn

    def label_for_time(self, time_sec: float) -> tuple[str, float, float]:
        for turn in reversed(self.finished_turns):
            if turn.start_time <= time_sec <= turn.end_time:
                return turn.label, turn.confidence, turn.similarity
        if self.current is not None:
            current_start = (self.current.start_frame_index - 1) * self.frame_ms / 1000.0
            current_end = self.frame_index * self.frame_ms / 1000.0
            if current_start <= time_sec <= current_end:
                guess = self.current.current_guess or SpeakerDecision(label='UNKNOWN', similarity=0.0, confidence=0.0)
                return guess.label, guess.confidence, guess.similarity
        return 'UNKNOWN', 0.0, 0.0


class StreamingSessionProcessor:
    def __init__(
        self,
        session_id: str,
        asr: ToneStreamingASR,
        speaker_tracker: StreamingSpeakerTracker,
        sample_rate: int = SETTINGS.sample_rate,
        asr_chunk_ms: int = SETTINGS.asr_chunk_ms,
    ) -> None:
        self.session_id = session_id
        self.asr = asr
        self.speaker_tracker = speaker_tracker
        self.sample_rate = sample_rate
        self.asr_chunk_ms = asr_chunk_ms
        self.frame_samples = int(sample_rate * SETTINGS.vad_frame_ms / 1000)
        self.asr_chunk_frames = max(1, asr_chunk_ms // SETTINGS.vad_frame_ms)
        self._frame_buffer = bytearray()
        self._asr_frames: list[bytes] = []
        self.asr_state: Any = None
        self.segment_counter = 0
        self.seen_phrase_keys: set[tuple] = set()

    def _make_segment_key(self, phrase: NormalizedPhrase) -> tuple:
        return (round(phrase.start_time, 2), round(phrase.end_time, 2), phrase.text)

    def _build_segment(self, phrase: NormalizedPhrase, assigned_by: str) -> dict[str, Any]:
        mid_time = (phrase.start_time + phrase.end_time) / 2.0
        speaker, speaker_conf, speaker_sim = self.speaker_tracker.label_for_time(mid_time)
        self.segment_counter += 1
        return {
            'segment_id': self.segment_counter,
            'start_time': phrase.start_time,
            'end_time': phrase.end_time,
            'speaker': speaker,
            'speaker_confidence': speaker_conf,
            'speaker_similarity': speaker_sim,
            'text': phrase.text,
            'asr_confidence': phrase.asr_confidence,
            'raw_phrase': phrase.raw,
            'metadata': {'assigned_by': assigned_by, 'mid_time': mid_time},
        }

    def feed_bytes(self, data: bytes) -> dict[str, list[dict[str, Any]]]:
        self._frame_buffer.extend(data)
        events: dict[str, list[dict[str, Any]]] = {'speaker_turns': [], 'segments': []}
        frame_size_bytes = self.frame_samples * 2
        while len(self._frame_buffer) >= frame_size_bytes:
            frame = bytes(self._frame_buffer[:frame_size_bytes])
            del self._frame_buffer[:frame_size_bytes]
            turn = self.speaker_tracker.process_frame(frame)
            self._asr_frames.append(frame)

            if len(self._asr_frames) >= self.asr_chunk_frames:
                chunk = b''.join(self._asr_frames[: self.asr_chunk_frames])
                del self._asr_frames[: self.asr_chunk_frames]
                chunk_f32 = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                phrases, self.asr_state = self.asr.forward_chunk(chunk_f32, self.asr_state)
                for phrase in phrases:
                    key = self._make_segment_key(phrase)
                    if key in self.seen_phrase_keys:
                        continue
                    self.seen_phrase_keys.add(key)
                    events['segments'].append(self._build_segment(phrase, 'overlap_match'))

            if turn is not None:
                events['speaker_turns'].append({
                    'start_time': turn.start_time,
                    'end_time': turn.end_time,
                    'speaker': turn.label,
                    'confidence': turn.confidence,
                    'similarity': turn.similarity,
                })
        return events

    def finalize(self) -> dict[str, list[dict[str, Any]]]:
        events: dict[str, list[dict[str, Any]]] = {'speaker_turns': [], 'segments': []}
        turn = self.speaker_tracker.flush()
        if turn is not None:
            events['speaker_turns'].append({
                'start_time': turn.start_time,
                'end_time': turn.end_time,
                'speaker': turn.label,
                'confidence': turn.confidence,
                'similarity': turn.similarity,
            })

        if self._asr_frames:
            chunk = b''.join(self._asr_frames)
            self._asr_frames.clear()
            chunk_f32 = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            phrases, self.asr_state = self.asr.forward_chunk(chunk_f32, self.asr_state)
            for phrase in phrases:
                key = self._make_segment_key(phrase)
                if key in self.seen_phrase_keys:
                    continue
                self.seen_phrase_keys.add(key)
                events['segments'].append(self._build_segment(phrase, 'finalization'))

        remaining, self.asr_state = self.asr.finalize(self.asr_state)
        for phrase in remaining:
            key = self._make_segment_key(phrase)
            if key in self.seen_phrase_keys:
                continue
            self.seen_phrase_keys.add(key)
            events['segments'].append(self._build_segment(phrase, 'finalize'))

        return events
