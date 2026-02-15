"""
Emotion detection using a pretrained TensorFlow/Keras .h5 model.
Runs locally; no cloud APIs.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import cv2


# Order must match model output (common FER / FER2013 order)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Map predicted emotion -> interview confidence score (0-1)
EMOTION_TO_SCORE = {
    "happy": 0.9,
    "neutral": 0.7,
    "surprise": 0.6,
    "sad": 0.4,
    "angry": 0.3,
    "fear": 0.3,
    "disgust": 0.3,
}


class EmotionDetector:
    """Loads a Keras .h5 model and predicts emotion from face images."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Args:
            model_path: Path to .h5 file. If None, uses project models/emotion_model.h5.
        """
        base = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else base / "models" / "emotion_model.h5"
        self._model = None
        self._input_shape: Optional[Tuple[int, ...]] = None

    def load_model(self) -> bool:
        """
        Load the Keras model from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.model_path.exists():
            return False
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(str(self.model_path))
            # input_shape without batch: (H, W, C)
            self._input_shape = tuple(self._model.input.shape[1:])
            return True
        except Exception:
            return False

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Expected input shape (e.g. (48, 48, 1) for grayscale)."""
        if self._input_shape is None:
            self.load_model()
        return self._input_shape or (48, 48, 1)

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop: resize to model input, normalize to [0, 1], add batch dim.

        Args:
            face_image: Face ROI — grayscale (H, W), (H, W, 1), or BGR (H, W, 3).

        Returns:
            Batch of shape (1, *input_shape), float32 in [0, 1].
        """
        target = self.input_shape  # (H, W, C)
        h, w = target[0], target[1]
        channels = target[2] if len(target) == 3 else 1

        if face_image.ndim == 2:
            face_image = np.expand_dims(face_image, axis=-1)
        elif face_image.ndim == 3 and face_image.shape[-1] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = np.expand_dims(face_image, axis=-1)

        if face_image.shape[:2] != (h, w):
            face_image = cv2.resize(face_image, (w, h), interpolation=cv2.INTER_LINEAR)

        if face_image.dtype != np.float32:
            face_image = face_image.astype(np.float32)
        if face_image.max() > 1.0:
            face_image = face_image / 255.0

        if channels == 1 and face_image.shape[-1] != 1:
            face_image = np.mean(face_image, axis=-1, keepdims=True)
        batch = np.expand_dims(face_image, axis=0)
        return batch

    def predict_emotion(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Predict dominant emotion and return a confidence score (0-1) for interview context.

        Args:
            face_image: Raw face ROI (H, W) or (H, W, 1); will be preprocessed.

        Returns:
            (emotion_label, emotion_score) e.g. ("happy", 0.9).
            emotion_score is from EMOTION_TO_SCORE, not raw softmax prob.
        """
        if self._model is None and not self.load_model():
            return EMOTION_LABELS[-1], EMOTION_TO_SCORE["neutral"]

        batch = self.preprocess(face_image)
        probs = self._model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else EMOTION_LABELS[-1]
        score = EMOTION_TO_SCORE.get(label, 0.5)
        return label, float(score)

    def predict_emotions(self, face_image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Return all emotions with probabilities, sorted by probability descending.

        Args:
            face_image: Face ROI (preprocessed or raw).

        Returns:
            List of (label, probability) sorted by probability.
        """
        if self._model is None and not self.load_model():
            return [(EMOTION_LABELS[-1], 0.0)]

        batch = self.preprocess(face_image)
        probs = self._model.predict(batch, verbose=0)[0]
        pairs = [(EMOTION_LABELS[i], float(probs[i])) for i in range(min(len(EMOTION_LABELS), len(probs)))]
        pairs.sort(key=lambda x: -x[1])
        return pairs
