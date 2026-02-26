"""
Edge-Based AI Interview Confidence Analyzer – Real-Time WebRTC Version
Continuous streaming with no fixed time limits.
"""

import time
import threading
from typing import List, Optional
import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from analyzer.face_detection import FaceDetector
from analyzer.emotion_detection import EmotionDetector
from analyzer.voice_analysis import VoiceAnalyzer
from analyzer.scoring import calculate_confidence
from analyzer.question_generator import generate_questions

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Interview Pro - Real-Time",
    page_icon="🎯",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
.main-header { font-size: 42px; font-weight: 800; color: #1E293B; margin-bottom: 20px; }
.status-badge { padding: 5px 15px; border-radius: 20px; font-size: 14px; font-weight: 600; }
.recording-on { background-color: #FEE2E2; color: #DC2626; border: 1px solid #FECACA; }
.recording-off { background-color: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }
.metric-container { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# VIDEO PROCESSOR
# ---------------------------------------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._face_detector = FaceDetector()
        self._emotion_detector = EmotionDetector()
        self._recording = False
        self._lock = threading.Lock()
        self.emotion_scores = []
        self.eye_scores = []
        self.emotion_labels = []

    def update_recording_state(self, state: bool):
        with self._lock:
            if state and not self._recording:
                # Reset buffers on start
                self.emotion_scores = []
                self.eye_scores = []
                self.emotion_labels = []
            self._recording = state

    def get_results(self):
        with self._lock:
            return {
                "avg_emotion": np.mean(self.emotion_scores) if self.emotion_scores else 0.0,
                "avg_eye": np.mean(self.eye_scores) if self.eye_scores else 0.0,
                "labels": list(self.emotion_labels)
            }

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        with self._lock:
            rec = self._recording

        if rec:
            eye_score, processed = self._face_detector.process_frame(img)
            boxes = self._face_detector.detect_faces(img)
            
            emotion_label, emotion_score = "neutral", 0.5
            if boxes:
                roi = self._face_detector.extract_face_roi(img, boxes[0], self._emotion_detector.input_shape[:2])
                if roi is not None:
                    emotion_label, emotion_score = self._emotion_detector.predict_emotion(roi)
            
            with self._lock:
                self.emotion_scores.append(emotion_score)
                self.eye_scores.append(eye_score)
                self.emotion_labels.append(emotion_label)
            
            # HUD overlay
            cv2.putText(processed, f"REC | {emotion_label.upper()}", (30, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            img = processed
        else:
            cv2.putText(img, "STANDBY", (30, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (150, 150, 150), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
def init_state():
    if "stage" not in st.session_state: st.session_state.stage = "setup"
    if "questions" not in st.session_state: st.session_state.questions = []
    if "idx" not in st.session_state: st.session_state.idx = 0
    if "results" not in st.session_state: st.session_state.results = []
    if "recording" not in st.session_state: st.session_state.recording = False
    if "voice_analyzer" not in st.session_state: st.session_state.voice_analyzer = VoiceAnalyzer()
    if "voice_score" not in st.session_state: st.session_state.voice_score = 0.0

init_state()

# ---------------------------------------------------
# UI STAGES
# ---------------------------------------------------

def show_setup():
    st.markdown('<div class="main-header">🚀 Interview Intelligence Setup</div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input("🎯 Target Job Role", "Software Engineer")
            exp = st.selectbox("📈 Seniority Level", ["Fresher", "Junior", "Mid", "Senior", "Lead"])
        with col2:
            skills = st.text_input("🛠 Core Skills", "Python, React, System Design")
            rtype = st.selectbox("🔄 Round Type", ["Technical", "HR", "Behavioral"])
        
        count = st.slider("❓ Number of Questions", 1, 10, 3)
        
        if st.button("Initialize AI Interview Engine", type="primary", use_container_width=True):
            with st.spinner("Synthesizing custom questions..."):
                data = generate_questions(role, exp, skills, rtype, count=count)
                st.session_state.questions = data.get("questions", [])
                st.session_state.stage = "interview"
                st.session_state.idx = 0
                st.rerun()

def show_interview():
    q = st.session_state.questions[st.session_state.idx]
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown(f"### Question {st.session_state.idx+1} of {len(st.session_state.questions)}")
        st.info(f"**{q['question']}**")
        
        # WebRTC Implementation
        ctx = webrtc_streamer(
            key="interview-stream",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False}, # Separate audio handling
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        if ctx.video_processor:
            ctx.video_processor.update_recording_state(st.session_state.recording)
            
        col_btn1, col_btn2 = st.columns(2)
        
        if not st.session_state.recording:
            if col_btn1.button("🎤 Start Answer", type="primary", use_container_width=True):
                st.session_state.recording = True
                # Start voice in background (long duration, manual stop)
                def voice_thread():
                    _, feat = st.session_state.voice_analyzer.record_and_analyze(duration_seconds=300.0, continuous=True)
                    st.session_state.voice_score = feat.get("voice_score", 0.0)
                
                threading.Thread(target=voice_thread).start()
                st.rerun()
        else:
            if col_btn1.button("⏹ Stop & Analyze", type="secondary", use_container_width=True):
                st.session_state.recording = False
                st.session_state.voice_analyzer.stop_recording()
                
                if ctx.video_processor:
                    res = ctx.video_processor.get_results()
                    final = calculate_confidence(res["avg_eye"], res["avg_emotion"], st.session_state.voice_score)
                    
                    st.session_state.results.append({
                        "q": q["question"],
                        "score": final,
                        "emotion": max(set(res["labels"]), key=res["labels"].count) if res["labels"] else "neutral",
                        "eye": res["avg_eye"],
                        "voice": st.session_state.voice_score
                    })
                    
                    st.session_state.idx += 1
                    if st.session_state.idx >= len(st.session_state.questions):
                        st.session_state.stage = "results"
                st.rerun()

    with col_right:
        st.markdown("#### Real-Time Metrics")
        rec_status = "RECORDING" if st.session_state.recording else "STANDBY"
        status_class = "recording-on" if st.session_state.recording else "recording-off"
        st.markdown(f'<span class="status-badge {status_class}">{rec_status}</span>', unsafe_allow_html=True)
        
        st.divider()
        if st.session_state.recording:
            st.warning("Confidence models are active. Maintain eye contact and speak clearly.")
        else:
            st.info("Prepare your thoughts, then click 'Start Answer'.")

def show_results():
    st.markdown('<div class="main-header">📊 Intelligent Performance Analytics</div>', unsafe_allow_html=True)
    
    avg = sum(r["score"] for r in st.session_state.results) / len(st.session_state.results)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Aggregate Confidence", f"{avg:.1f}%")
        st.progress(avg / 100)
    
    with col2:
        if avg > 75: st.success("Exceptional! Your delivery shows high professional maturity.")
        elif avg > 50: st.info("Solid performance. Minor adjustments in eye contact could boost score.")
        else: st.warning("Consider practicing your delivery to appear more confident.")

    st.divider()
    
    for i, r in enumerate(st.session_state.results):
        with st.expander(f"Question {i+1} Detailed Breakdown"):
            st.write(f"**Question:** {r['q']}")
            c_a, c_b, c_c, c_d = st.columns(4)
            c_a.metric("Final", f"{r['score']:.1f}%")
            c_b.metric("Dominant Emotion", r['emotion'])
            c_c.metric("Eye Contact", f"{r['eye']:.2f}")
            c_d.metric("Voice Tone", f"{r['voice']:.2f}")

    if st.button("🔁 Start New Session", use_container_width=True):
        st.session_state.stage = "setup"
        st.session_state.results = []
        st.rerun()

# ---------------------------------------------------
# ROUTER
# ---------------------------------------------------
if st.session_state.stage == "setup":
    show_setup()
elif st.session_state.stage == "interview":
    show_interview()
elif st.session_state.stage == "results":
    show_results()