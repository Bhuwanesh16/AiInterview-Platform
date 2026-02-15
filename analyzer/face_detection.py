"""
Face detection using OpenCV + MediaPipe Face Mesh.
Runs locally; no cloud APIs.
"""

from typing import Optional, Tuple, List
import numpy as np
import cv2
import mediapipe as mp


# MediaPipe Face Mesh landmark indices
NOSE_TIP = 1
LEFT_IRIS = 468
RIGHT_IRIS = 473


class FaceDetector:
    """Detects faces with MediaPipe Face Mesh and computes eye contact score."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        """
        Initialize MediaPipe Face Mesh and OpenCV.

        Args:
            min_detection_confidence: Minimum confidence for face detection [0, 1].
        """
        self.min_detection_confidence = min_detection_confidence
        self._face_mesh = None
        self._initialized = False

    def initialize(self) -> None:
        """Lazy init of MediaPipe Face Mesh."""
        if self._initialized:
            return
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # needed for iris (468, 473)
            min_detection_confidence=self.min_detection_confidence,
        )
        self._initialized = True

    def process_frame(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Run face mesh on a BGR frame and compute eye contact score.

        Args:
            frame: BGR image (H, W, 3), e.g. from cv2.VideoCapture.

        Returns:
            eye_score: Float in [0, 1]. 0 = no face or looking away, 1 = looking at camera.
            processed_frame: BGR frame with optional landmark drawing (same size as input).
        """
        self.initialize()
        h, w = frame.shape[:2]
        processed = frame.copy()

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        eye_score = 0.0
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            eye_score = self._eye_contact_score(landmarks, w, h)
            # Draw landmarks on processed frame (optional)
            self._draw_landmarks(processed, landmarks, w, h)

        return eye_score, processed

    def _eye_contact_score(self, landmarks, width: int, height: int) -> float:
        """
        Simple eye contact score: how close the gaze center is to the frame center.
        Uses iris centers if available (refine_landmarks=True), else nose tip.
        """
        num_landmarks = len(landmarks.landmark)
        frame_cx = width / 2.0
        frame_cy = height / 2.0
        max_dist = np.sqrt(width ** 2 + height ** 2) / 2.0
        if max_dist <= 0:
            return 0.0

        if num_landmarks > RIGHT_IRIS:
            # Use midpoint of both irises as "gaze center"
            left = landmarks.landmark[LEFT_IRIS]
            right = landmarks.landmark[RIGHT_IRIS]
            gaze_x = (left.x + right.x) / 2.0
            gaze_y = (left.y + right.y) / 2.0
        else:
            # Fallback: nose tip
            nose = landmarks.landmark[NOSE_TIP]
            gaze_x = nose.x
            gaze_y = nose.y

        px = gaze_x * width
        py = gaze_y * height
        distance = np.sqrt((px - frame_cx) ** 2 + (py - frame_cy) ** 2)
        # Clamp: 0 when far, 1 when at center
        score = max(0.0, 1.0 - distance / max_dist)
        return float(score)

    def _draw_landmarks(
        self, frame: np.ndarray, landmarks, width: int, height: int
    ) -> None:
        """Draw face mesh points on frame (for visualization)."""
        for lm in landmarks.landmark:
            x = int(lm.x * width)
            y = int(lm.y * height)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect face bounding boxes in a BGR frame using Face Mesh results.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            List of (x, y, w, h) bounding boxes for each face.
        """
        self.initialize()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        boxes = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [lm.x * w for lm in face_landmarks.landmark]
                ys = [lm.y * h for lm in face_landmarks.landmark]
                x_min = int(max(0, min(xs) - 0.05 * w))
                y_min = int(max(0, min(ys) - 0.05 * h))
                x_max = int(min(w, max(xs) + 0.05 * w))
                y_max = int(min(h, max(ys) + 0.05 * h))
                boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
        return boxes

    def extract_face_roi(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        target_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Crop and resize face region for emotion model input.

        Args:
            frame: Full BGR frame.
            box: (x, y, w, h) face bounding box.
            target_size: (height, width) for the model, e.g. (48, 48).

        Returns:
            Grayscale face patch of shape (H, W, 1), or None if invalid.
        """
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return None
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return None
        th, tw = target_size
        resized = cv2.resize(roi, (tw, th))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return np.expand_dims(gray, axis=-1).astype(np.float32) / 255.0

    def get_webcam_frame(self, cap: cv2.VideoCapture) -> Optional[Tuple[float, np.ndarray]]:
        """
        Read one frame from an OpenCV VideoCapture and process it.

        Args:
            cap: Open cv2.VideoCapture(0) (or other device).

        Returns:
            (eye_score, processed_frame) or None if no frame read.
        """
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return self.process_frame(frame)

    @staticmethod
    def open_webcam(device: int = 0) -> cv2.VideoCapture:
        """Create and return an OpenCV VideoCapture for webcam. Caller must release it."""
        return cv2.VideoCapture(device)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
        self._face_mesh = None
        self._initialized = False
