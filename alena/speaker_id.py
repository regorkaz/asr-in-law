from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

from .audio_utils import audio_duration_seconds, load_audio_file


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _to_embedding_vector(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return np.squeeze(arr).astype(np.float32)


@dataclass(slots=True)
class SpeakerProfile:
    embedding: np.ndarray
    sample_rate: int
    seconds: float
    source_path: str | None = None


@dataclass(slots=True)
class SpeakerDecision:
    label: str
    similarity: float
    confidence: float


class SpeakerEmbeddingExtractor:
    def __init__(self) -> None:
        self._classifier = None

    def load(self) -> None:
        if self._classifier is not None:
            return
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except Exception:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
        self._classifier = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb')

    @property
    def classifier(self):
        self.load()
        return self._classifier

    def encode_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.load()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        wav = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.no_grad():
            emb = self.classifier.encode_batch(wav)
        return _to_embedding_vector(emb)


class LawyerSpeakerMatcher:
    def __init__(self, threshold: float = 0.72) -> None:
        self.threshold = threshold
        self.profile: SpeakerProfile | None = None
        self.extractor = SpeakerEmbeddingExtractor()

    def enroll_from_audio(self, audio: np.ndarray, sample_rate: int, source_path: str | None = None) -> SpeakerProfile:
        embedding = self.extractor.encode_audio(audio, sample_rate)
        profile = SpeakerProfile(
            embedding=embedding,
            sample_rate=sample_rate,
            seconds=audio_duration_seconds(len(audio), sample_rate),
            source_path=source_path,
        )
        self.profile = profile
        return profile

    def enroll_from_file(self, path: str, target_sr: int = 16_000) -> SpeakerProfile:
        audio, sr = load_audio_file(path, target_sr=target_sr)
        return self.enroll_from_audio(audio, sr, source_path=path)

    def compare(self, audio: np.ndarray, sample_rate: int) -> SpeakerDecision:
        if self.profile is None:
            return SpeakerDecision(label='UNKNOWN', similarity=0.0, confidence=0.0)
        emb = self.extractor.encode_audio(audio, sample_rate)
        sim = _cosine_similarity(emb, self.profile.embedding)
        label = 'LAWYER' if sim >= self.threshold else 'CLIENT'
        if label == 'LAWYER':
            conf = min(1.0, max(0.0, (sim - self.threshold) / max(1e-6, 1.0 - self.threshold)))
        else:
            conf = min(1.0, max(0.0, (self.threshold - sim) / max(1e-6, self.threshold)))
        return SpeakerDecision(label=label, similarity=sim, confidence=conf)
