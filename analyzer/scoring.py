"""
Confidence scoring: combine face/emotion and voice signals into a single score.
Runs locally; no cloud APIs.
"""

from dataclasses import dataclass
from typing import List, Optional

# Weights for the simple confidence formula
EYE_WEIGHT = 0.4
EMOTION_WEIGHT = 0.3
VOICE_WEIGHT = 0.3


def calculate_confidence(
    eye_score: float,
    emotion_score: float,
    voice_score: float,
) -> float:
    """
    Combined confidence from eye contact, emotion, and voice (0-100).

    Formula:
        confidence = 0.4 * eye_score + 0.3 * emotion_score + 0.3 * voice_score
        final_score = confidence normalized to 0-100.

    Args:
        eye_score: 0-1 (from face_detection).
        emotion_score: 0-1 (from emotion_detection).
        voice_score: 0-1 (from voice_analysis).

    Returns:
        final_score: Float in [0, 100].
    """
    e = max(0.0, min(1.0, float(eye_score)))
    m = max(0.0, min(1.0, float(emotion_score)))
    v = max(0.0, min(1.0, float(voice_score)))
    confidence = EYE_WEIGHT * e + EMOTION_WEIGHT * m + VOICE_WEIGHT * v
    final_score = confidence * 100.0
    return round(final_score, 2)


@dataclass
class FrameResult:
    """Result for a single video frame (face + emotion)."""
    face_detected: bool
    emotion: str
    emotion_confidence: float
    # Optional: bounding box or ROI for debugging
    box: Optional[tuple] = None


@dataclass
class VoiceResult:
    """Result of voice analysis for a segment."""
    transcript: str
    pitch_mean: float
    energy_mean: float
    pause_ratio: float
    speech_rate: float
    features: dict


class ConfidenceScorer:
    """Aggregates frame and voice results into an overall confidence score."""

    def __init__(
        self,
        emotion_weights: Optional[dict] = None,
        positive_emotions: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            emotion_weights: Map emotion -> weight (e.g. happy=1.0, neutral=0.7, fear=0.2).
            positive_emotions: Emotions that boost confidence; rest use weights or penalize.
        """
        self.emotion_weights = emotion_weights or {
            "happy": 1.0,
            "neutral": 0.8,
            "surprise": 0.6,
            "sad": 0.4,
            "angry": 0.2,
            "fear": 0.2,
            "disgust": 0.1,
        }
        self.positive_emotions = positive_emotions or ["happy", "neutral", "surprise"]

    def score_emotion_frames(self, frame_results: List[FrameResult]) -> float:
        """
        Compute a 0–1 score from a list of frame-level emotion results.

        Args:
            frame_results: Results per frame (e.g. over last N seconds).

        Returns:
            Score in [0, 1]; higher = more confident.
        """
        if not frame_results:
            return 0.0
        # TODO: average weight[emotion] * emotion_confidence; normalize to [0,1]
        return 0.0

    def score_voice(self, voice_result: VoiceResult) -> float:
        """
        Compute a 0–1 score from voice features (pitch stability, energy, pauses).

        Args:
            voice_result: Output of VoiceAnalyzer.extract_features + transcript.

        Returns:
            Score in [0, 1]; higher = more confident.
        """
        # TODO: combine pitch_mean, energy_mean, pause_ratio, speech_rate into scalar
        return 0.0

    def combined_score(
        self,
        frame_results: List[FrameResult],
        voice_result: Optional[VoiceResult] = None,
        face_weight: float = 0.6,
        voice_weight: float = 0.4,
    ) -> float:
        """
        Combined confidence score from face/emotion and optional voice.

        Args:
            frame_results: Per-frame emotion results.
            voice_result: Optional voice analysis result.
            face_weight: Weight for face/emotion score.
            voice_weight: Weight for voice score (ignored if voice_result is None).

        Returns:
            Score in [0, 1].
        """
        face_s = self.score_emotion_frames(frame_results)
        if voice_result is None:
            return face_s
        voice_s = self.score_voice(voice_result)
        return face_weight * face_s + voice_weight * voice_s
