"""
Edge-Based AI Interview Confidence Analyzer – Streamlit app.
Integrates face, emotion, voice, and scoring. Runs fully locally.
"""

import time
import streamlit as st
import cv2
import numpy as np

from analyzer.face_detection import FaceDetector
from analyzer.emotion_detection import EmotionDetector
from analyzer.voice_analysis import VoiceAnalyzer
from analyzer.scoring import calculate_confidence


def init_session_state() -> None:
    if "interview_running" not in st.session_state:
        st.session_state.interview_running = False
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "face_detector" not in st.session_state:
        st.session_state.face_detector = None
    if "emotion_detector" not in st.session_state:
        st.session_state.emotion_detector = None
    if "voice_analyzer" not in st.session_state:
        st.session_state.voice_analyzer = None
    if "voice_score" not in st.session_state:
        st.session_state.voice_score = 0.0
    if "last_eye_score" not in st.session_state:
        st.session_state.last_eye_score = 0.0
    if "last_emotion_label" not in st.session_state:
        st.session_state.last_emotion_label = "—"
    if "last_emotion_score" not in st.session_state:
        st.session_state.last_emotion_score = 0.0
    if "last_final_score" not in st.session_state:
        st.session_state.last_final_score = 0.0


def start_interview() -> None:
    init_session_state()
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.face_detector = FaceDetector()
    st.session_state.emotion_detector = EmotionDetector()
    st.session_state.voice_analyzer = VoiceAnalyzer()
    st.session_state.interview_running = True


def stop_interview() -> None:
    if st.session_state.get("cap") is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    if st.session_state.get("face_detector") is not None:
        st.session_state.face_detector.close()
        st.session_state.face_detector = None
    st.session_state.interview_running = False


def process_one_frame() -> None:
    cap = st.session_state.cap
    face_detector = st.session_state.face_detector
    emotion_detector = st.session_state.emotion_detector
    if cap is None or face_detector is None or emotion_detector is None:
        return

    ret, frame = cap.read()
    if not ret or frame is None:
        return

    eye_score, processed_frame = face_detector.process_frame(frame)
    boxes = face_detector.detect_faces(frame)
    emotion_label, emotion_score = "neutral", 0.5
    if boxes:
        face_roi = face_detector.extract_face_roi(
            frame, boxes[0], emotion_detector.input_shape[:2]
        )
        if face_roi is not None:
            emotion_label, emotion_score = emotion_detector.predict_emotion(face_roi)

    voice_score = st.session_state.voice_score
    final_score = calculate_confidence(eye_score, emotion_score, voice_score)

    st.session_state.last_eye_score = eye_score
    st.session_state.last_emotion_label = emotion_label
    st.session_state.last_emotion_score = emotion_score
    st.session_state.last_final_score = final_score
    st.session_state.last_processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(
        page_title="AI Interview Confidence Analyzer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    st.title("🎯 Edge-Based AI Interview Confidence Analyzer")
    st.caption("Runs 100% locally — no cloud APIs.")

    # ——— Sidebar ———
    with st.sidebar:
        st.subheader("Interview")
        if not st.session_state.interview_running:
            if st.button("▶ Start Interview", type="primary", use_container_width=True):
                start_interview()
                st.rerun()
        else:
            if st.button("⏹ Stop Interview", type="secondary", use_container_width=True):
                stop_interview()
                st.rerun()

        st.divider()
        st.subheader("Voice")
        if st.button("🎤 Record voice (20 s)", use_container_width=True):
            analyzer = st.session_state.voice_analyzer or VoiceAnalyzer()
            st.session_state.voice_analyzer = analyzer
            with st.spinner("Recording 20 seconds… speak now."):
                _, features = analyzer.record_and_analyze(duration_seconds=20.0)
                st.session_state.voice_score = features.get("voice_score", 0.0)
            st.success(f"Voice score: {st.session_state.voice_score:.2f}")
            st.rerun()

        if st.session_state.voice_score > 0:
            st.metric("Current voice score", f"{st.session_state.voice_score:.2f}")

        st.divider()
        st.caption("Weights: Eye 40% · Emotion 30% · Voice 30%")

    # ——— Main area ———
    if st.session_state.interview_running:
        process_one_frame()
        last_frame = st.session_state.get("last_processed_frame")
        if last_frame is not None:
            col_cam, col_metrics = st.columns([2, 1])
            with col_cam:
                st.image(last_frame, channels="RGB", use_container_width=True)
            with col_metrics:
                st.metric("Emotion", st.session_state.last_emotion_label)
                st.metric("Eye score", f"{st.session_state.last_eye_score:.2f}")
                st.metric("Voice score", f"{st.session_state.voice_score:.2f}")
                st.metric("Final confidence", f"{st.session_state.last_final_score:.1f} / 100")
                st.progress(st.session_state.last_final_score / 100.0)
        else:
            st.info("Starting camera…")
        time.sleep(0.1)
        st.rerun()
    else:
        st.info("Click **Start Interview** in the sidebar to begin.")
        # Show placeholder metrics so layout is clear
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Emotion", "—")
        with col2:
            st.metric("Eye score", "—")
        with col3:
            st.metric("Voice score", "—")
        with col4:
            st.metric("Final confidence", "—")
        st.progress(0.0)

    with st.expander("About"):
        st.markdown("""
        - **Webcam:** OpenCV + MediaPipe Face Mesh (eye contact score).
        - **Emotion:** Pretrained Keras `.h5` in `models/`.
        - **Voice:** PyAudio + Librosa (record 20 s for voice score).
        - **Score:** `0.4×eye + 0.3×emotion + 0.3×voice` → 0–100.
        """)


if __name__ == "__main__":
    main()
