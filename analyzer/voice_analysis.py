"""
Voice analysis using Librosa (features). Recording via PyAudio.
Runs fully offline; no cloud APIs.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import librosa


# Default recording duration (seconds)
DEFAULT_RECORD_SECONDS = 20.0


class VoiceAnalyzer:
    """Record audio and extract features for confidence scoring. All offline."""

    def __init__(self, sample_rate: int = 16000) -> None:
        """
        Args:
            sample_rate: Sample rate for recording and analysis (16000 is reliable).
        """
        self.sample_rate = sample_rate
        self._initialized = False

    def initialize(self) -> None:
        """Lazy init (no heavy setup required)."""
        self._initialized = True

    def record_from_microphone(
        self, duration_seconds: float = DEFAULT_RECORD_SECONDS
    ) -> Optional[np.ndarray]:
        """
        Record audio from the default microphone using PyAudio.

        Args:
            duration_seconds: Recording length in seconds (default 20).

        Returns:
            Audio as float32 in [-1, 1], shape (n_samples,), or None on error.
        """
        try:
            import pyaudio
        except ImportError:
            return None

        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        total_frames = int(self.sample_rate * duration_seconds / chunk)
        frames = []

        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=format,
                channels=channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=chunk,
            )
            for _ in range(total_frames):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception:
            return None

        if not frames:
            return None
        audio_int = np.concatenate(frames)
        audio_float = audio_int.astype(np.float32) / 32768.0
        return audio_float

    def load_audio(self, path: Path) -> Optional[np.ndarray]:
        """
        Load audio file into a numpy array (mono, self.sample_rate).

        Args:
            path: Path to WAV or other format supported by librosa.

        Returns:
            Audio samples (n_samples,) float32, or None on error.
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            y, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
            return y.astype(np.float32)
        except Exception:
            return None

    def extract_features(self, audio: np.ndarray) -> dict:
        """
        Extract speech duration, silence ratio, and basic pitch variation using Librosa.

        Args:
            audio: 1D float array at self.sample_rate.

        Returns:
            Dict with: speech_duration_sec, silence_ratio, pitch_variation_std, voice_score.
        """
        if audio is None or len(audio) == 0:
            return self._empty_features()

        sr = self.sample_rate
        total_duration_sec = len(audio) / sr

        # RMS per frame to separate speech vs silence
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = np.percentile(rms, 20)  # below 20th percentile = silence
        if np.isnan(threshold) or threshold <= 0:
            threshold = rms.mean() * 0.1 if rms.size else 1e-6

        speech_frames = np.sum(rms >= threshold)
        total_frames = len(rms)
        silence_ratio = 1.0 - (speech_frames / total_frames) if total_frames else 0.0
        frame_duration_sec = hop_length / sr
        speech_duration_sec = float(speech_frames * frame_duration_sec)

        # Basic pitch variation (f0 std) using pyin
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=75, fmax=400, sr=sr, frame_length=frame_length, hop_length=hop_length
        )
        voiced = np.isfinite(f0) & (f0 > 0)
        if np.any(voiced):
            pitch_variation_std = float(np.nanstd(f0[voiced]))
        else:
            pitch_variation_std = 0.0

        voice_score = self._compute_voice_score(
            total_duration_sec=total_duration_sec,
            speech_duration_sec=speech_duration_sec,
            silence_ratio=silence_ratio,
            pitch_variation_std=pitch_variation_std,
        )

        return {
            "speech_duration_sec": speech_duration_sec,
            "silence_ratio": silence_ratio,
            "pitch_variation_std": pitch_variation_std,
            "voice_score": voice_score,
            "total_duration_sec": total_duration_sec,
        }

    def _compute_voice_score(
        self,
        total_duration_sec: float,
        speech_duration_sec: float,
        silence_ratio: float,
        pitch_variation_std: float,
    ) -> float:
        """
        Simple 0-1 score: reward more speech, less silence, some pitch variation.
        """
        if total_duration_sec <= 0:
            return 0.0
        speech_ratio = speech_duration_sec / total_duration_sec
        # More speech = better (cap at 0.9 so 100% speech isn't required)
        speech_component = min(1.0, speech_ratio / 0.7) * 0.5
        # Less silence = better
        silence_component = (1.0 - silence_ratio) * 0.35
        # Some pitch variation = natural (std 20-100 Hz is good)
        pitch_component = min(1.0, pitch_variation_std / 80.0) * 0.15
        score = speech_component + silence_component + pitch_component
        return float(np.clip(score, 0.0, 1.0))

    def _empty_features(self) -> dict:
        return {
            "speech_duration_sec": 0.0,
            "silence_ratio": 1.0,
            "pitch_variation_std": 0.0,
            "voice_score": 0.0,
            "total_duration_sec": 0.0,
        }

    def record_and_analyze(
        self, duration_seconds: float = DEFAULT_RECORD_SECONDS
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Record for the given duration and extract features in one call.

        Args:
            duration_seconds: Recording length (default 20).

        Returns:
            (audio_array or None, features_dict). features_dict always has voice_score.
        """
        audio = self.record_from_microphone(duration_seconds)
        features = self.extract_features(audio) if audio is not None else self._empty_features()
        return audio, features

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe speech to text (optional; requires offline recognizer for full edge).
        Not implemented here to keep the module offline and simple.
        """
        return ""
