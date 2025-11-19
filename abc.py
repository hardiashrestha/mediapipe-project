import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="VisionVerse - The AI Detection Universe",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS with animations and modern design
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Landing Page Styles */
    .landing-container {
        text-align: center;
        padding: 30px 20px;
        min-height: 100vh;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .logo-container {
        margin: 20px auto;
        width: 200px;
        height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: float 3s ease-in-out infinite, rotate 10s linear infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(5deg); }
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .emoji-logo {
        font-size: 8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        width: 100%;
        height: 100%;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 
            0 0 40px rgba(102, 126, 234, 0.6),
            0 0 80px rgba(118, 75, 162, 0.5),
            0 0 120px rgba(240, 147, 251, 0.4);
        animation: glow 2s ease-in-out infinite alternate, pulse 3s ease-in-out infinite;
        border: 4px solid transparent;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) padding-box,
                    linear-gradient(45deg, #667eea, #764ba2, #f093fb) border-box;
        position: relative;
        overflow: hidden;
    }
    
    .emoji-logo::before {
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    @keyframes glow {
        from {
            box-shadow: 
                0 0 30px rgba(102, 126, 234, 0.7),
                0 0 60px rgba(118, 75, 162, 0.5),
                0 0 90px rgba(240, 147, 251, 0.4);
        }
        to {
            box-shadow: 
                0 0 50px rgba(102, 126, 234, 0.9),
                0 0 100px rgba(118, 75, 162, 0.7),
                0 0 150px rgba(240, 147, 251, 0.5);
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .landing-title {
        font-size: 8rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 30px 0 20px 0;
        animation: titlePulse 3s ease-in-out infinite, colorShift 5s ease-in-out infinite;
        text-shadow: 2px 2px 20px rgba(102, 126, 234, 0.3);
        letter-spacing: 3px;
    }
    
    @keyframes titlePulse {
        0%, 100% { 
            opacity: 1; 
            transform: scale(1);
        }
        50% { 
            opacity: 0.85; 
            transform: scale(1.02);
        }
    }
    
    @keyframes colorShift {
        0%, 100% { filter: hue-rotate(0deg) brightness(1); }
        50% { filter: hue-rotate(15deg) brightness(1.1); }
    }
    
    .landing-subtitle {
        font-size: 3rem;
        color: #555;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 600;
        animation: fadeInUp 1s ease-out, subtitleGlow 4s ease-in-out infinite;
        letter-spacing: 2px;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes subtitleGlow {
        0%, 100% { text-shadow: 0 0 10px rgba(102, 126, 234, 0.3); }
        50% { text-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
    }
    
    .feature-stats {
        text-align: center;
        font-size: 1.4rem;
        color: #666;
        margin-bottom: 50px;
        animation: fadeIn 2s ease-out;
        font-weight: 500;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 25px;
        max-width: 1200px;
        margin: 50px auto;
        padding: 20px;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 35px;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.4s ease;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        min-height: 200px;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .feature-card:hover::before {
        left: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.5);
    }
    
    /* Small emoji logo for model page */
    .small-emoji-logo {
        position: fixed;
        top: 20px;
        right: 20px;
        width: 80px;
        height: 80px;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 
            0 0 20px rgba(102, 126, 234, 0.4),
            0 0 40px rgba(118, 75, 162, 0.3);
        animation: logoGlow 2s ease-in-out infinite alternate;
        z-index: 1000;
    }
    
    @keyframes logoGlow {
        from {
            box-shadow: 
                0 0 15px rgba(102, 126, 234, 0.5),
                0 0 30px rgba(118, 75, 162, 0.3);
        }
        to {
            box-shadow: 
                0 0 25px rgba(102, 126, 234, 0.7),
                0 0 50px rgba(118, 75, 162, 0.5);
        }
    }
    
    /* Main Header with Animation */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1.5rem;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(20deg); }
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-top: -20px;
        margin-bottom: 30px;
        font-weight: 500;
    }
    
    .info-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 20px;
        margin: 25px 0;
        border: 3px solid #e0e0e0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .info-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    
    .gestures-list {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
        margin-top: 20px;
    }
    
    .gesture-item {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px 25px;
        border-radius: 15px;
        text-align: center;
        min-width: 130px;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .gesture-item:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.6);
    }
    
    .gesture-icon {
        font-size: 3rem;
        margin-bottom: 8px;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .gesture-name {
        font-size: 1.1rem;
        font-weight: bold;
        margin: 0;
        letter-spacing: 0.5px;
    }
    
    .detected-gesture-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 25px 0;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.4);
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .hand-result-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 5px;
        text-align: center;
        min-width: 180px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        display: inline-block;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        margin: 10px 0;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 10px 0;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .status-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .badge-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .badge-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .footer {
        text-align: center;
        padding: 30px;
        background: linear-gradient(90deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin-top: 40px;
    }
    
    .footer-text {
        font-size: 1.1rem;
        color: #444;
        font-weight: 600;
    }
    
    .feature-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Initialize MediaPipe
mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

@st.cache_resource
def get_model_path(model_name):
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, 'models', model_name)
    return os.path.normpath(model_path)

@st.cache_resource
def load_gesture_recognizer_multi():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    model_path = get_model_path('gesture_recognizer.task')
    base_opt = python.BaseOptions(model_asset_path=model_path)
    opt = vision.GestureRecognizerOptions(
        base_options=base_opt,
        num_hands=5,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return vision.GestureRecognizer.create_from_options(opt)

@st.cache_resource
def load_object_detector_image():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    model_path = get_model_path('efficientdet_lite0.tflite')
    base_opt = python.BaseOptions(model_asset_path=model_path)
    opt = vision.ObjectDetectorOptions(base_options=base_opt, max_results=5, score_threshold=0.5)
    return vision.ObjectDetector.create_from_options(opt)

@st.cache_resource
def load_audio_classifier():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import audio
    model_path = get_model_path('yamnet.tflite')
    base_opt = python.BaseOptions(model_asset_path=model_path)
    opt = audio.AudioClassifierOptions(base_options=base_opt, max_results=5)
    return audio.AudioClassifier.create_from_options(opt)

GESTURE_MAP = {
    "Thumb_Up": ("Thumbs Up", "👍"),
    "Thumb_Down": ("Thumbs Down", "👎"),
    "Victory": ("Peace", "✌️"),
    "Open_Palm": ("Open Hand", "🤚"),
    "Closed_Fist": ("Fist", "✊"),
    "Pointing_Up": ("Pointing Up", "☝️"),
    "ILoveYou": ("I Love You", "🤟")
}

HAND_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)
]

ALL_MODELS = {
    "🔍 Face Detection": {"icon": "👤", "desc": "Detect faces with 6 key landmarks", "category": "Vision AI"},
    "😊 Face Mesh (468 Landmarks)": {"icon": "😊", "desc": "Ultra-detailed 468-point facial mapping", "category": "Vision AI"},
    "🧍 Pose Detection (33 Points)": {"icon": "🧍", "desc": "Full-body pose tracking with 33 landmarks", "category": "Body Tracking"},
    "✋ Gesture Recognition": {"icon": "✋", "desc": "Multi-hand gesture detection (up to 5 hands)", "category": "Body Tracking"},
    "📦 Object Detection": {"icon": "📦", "desc": "Real-time object recognition (90+ types)", "category": "Object Detection"},
    "🎵 Audio Classification": {"icon": "🎵", "desc": "Audio event detection (521 types)", "category": "Audio AI"}
}

# ========== LANDING PAGE ==========
if st.session_state.page == 'landing':
    
    st.markdown("""
    <div class="logo-container">
        <div class="emoji-logo">👁️</div>
    </div>
    <h1 class="landing-title">VisionVerse by Shrestha & Ankit</h1>
    <p class="landing-subtitle">The AI Detection Universe</p>
    <p class="feature-stats">
        🚀 6 AI-Powered Models • ⚡ Real-Time Processing • 🎯 Multi-Modal Detection
    </p>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    for idx, (model_name, model_data) in enumerate(ALL_MODELS.items()):
        with cols[idx % 3]:
            if st.button(f"{model_data['icon']} {model_name}", key=f"btn_{model_name}", use_container_width=True):
                st.session_state.page = 'model'
                st.session_state.selected_model = model_name
                st.rerun()
            st.markdown(f'<div style="text-align:center; padding:10px; color:#666; font-size:0.85rem;">{model_data["desc"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer" style="margin-top:80px;"><p class="footer-text">✨ Built with <b>MediaPipe</b> & <b>Streamlit</b><br>💎 Advanced Computer Vision & Audio AI Platform</p></div>', unsafe_allow_html=True)

# ========== MODEL PAGE ==========
elif st.session_state.page == 'model':
    header_cols = st.columns([1, 5, 1])
    with header_cols[0]:
        if st.button("← Home", key="back_btn"):
            st.session_state.page = 'landing'
            st.rerun()
    with header_cols[2]:
        st.markdown('<div class="small-emoji-logo">👁️</div>', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">👁️ VisionVerse by Shrestha & Ankit</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Multi-Modal AI Detection • Real-Time Processing</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## 🎨 All Detection Models")
        feature = st.selectbox("Choose Model:", list(ALL_MODELS.keys()),
            index=list(ALL_MODELS.keys()).index(st.session_state.selected_model) if st.session_state.selected_model else 0)
        st.session_state.selected_model = feature
        
        st.markdown("---\n### 📋 Categories")
        vision_models = [k for k, v in ALL_MODELS.items() if v['category'] == 'Vision AI']
        body_models = [k for k, v in ALL_MODELS.items() if v['category'] == 'Body Tracking']
        object_models = [k for k, v in ALL_MODELS.items() if v['category'] == 'Object Detection']
        audio_models = [k for k, v in ALL_MODELS.items() if v['category'] == 'Audio AI']
        
        st.markdown("**👁️ Vision AI**")
        for model in vision_models: st.markdown(f"- {model}")
        st.markdown("**🧍 Body Tracking**")
        for model in body_models: st.markdown(f"- {model}")
        st.markdown("**📦 Object Detection**")
        for model in object_models: st.markdown(f"- {model}")
        st.markdown("**🎵 Audio AI**")
        for model in audio_models: st.markdown(f"- {model}")
        
        st.markdown("---\n### ⚡ Capabilities")
        st.markdown('<span class="feature-badge">Real-Time</span><span class="feature-badge">High Accuracy</span><span class="feature-badge">Multi-Hand</span>', unsafe_allow_html=True)
        st.markdown(f"---\n### 📊 System Info")
        st.info(f"**MediaPipe** v{mp.__version__}  \n**Status:** 🟢 Active")
    
    # FACE DETECTION
    if "Face Detection" in feature:
        st.markdown("## 👤 Face Detection Engine\n<p style='font-size:1.1rem; color:#666'>Advanced facial detection with 6 key landmarks • Real-time processing</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown('<div class="stat-card"><div class="stat-number">6</div><div class="stat-label">Key Landmarks</div></div>', unsafe_allow_html=True)
        with col2: st.markdown('<div class="stat-card"><div class="stat-number">30ms</div><div class="stat-label">Processing Time</div></div>', unsafe_allow_html=True)
        with col3: st.markdown('<div class="stat-card"><div class="stat-number">95%</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
        
        input_mode = st.radio("📥 Select Input Mode:", ["📷 Upload Image", "🎥 Live Webcam"], key="face_input", horizontal=True)
        
        if "Upload" in input_mode:
            file = st.file_uploader("📤 Drop your image here", type=["jpg", "jpeg", "png"], key="face_file")
            if file:
                img = Image.open(file)
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                with st.spinner("🔍 Analyzing facial features..."):
                    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
                        result = detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                        if result.detections:
                            st.markdown(f'<div class="status-badge badge-success">✅ Detected {len(result.detections)} face(s)</div>', unsafe_allow_html=True)
                            for detection in result.detections:
                                mp_drawing.draw_detection(img_bgr, detection)
                        else:
                            st.warning("❌ No faces detected")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📥 Original Image**\n<div class='image-container'>", unsafe_allow_html=True)
                    st.image(img, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown("**📤 Detection Result**\n<div class='image-container'>", unsafe_allow_html=True)
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("### 🎥 Live Webcam Detection")
            webcam_run = st.checkbox("▶️ START WEBCAM", key="face_webcam_run")
            frame_display, status_display = st.empty(), st.empty()
            if webcam_run:
                cap = cv2.VideoCapture(0)
                status_display.markdown('<div class="status-badge badge-success">🟢 Webcam Active</div>', unsafe_allow_html=True)
                with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
                    while webcam_run:
                        ret, frame = cap.read()
                        if not ret: break
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = detector.process(rgb)
                        if result.detections:
                            for detection in result.detections:
                                mp_drawing.draw_detection(frame, detection)
                        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                cap.release()
    
    # FACE MESH
    elif "Face Mesh" in feature:
        st.markdown("## 😊 Face Mesh - 468 Precision Landmarks\n<p style='font-size:1.1rem; color:#666'>Ultra-detailed facial mapping • 3D landmark tracking • Expression analysis</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown('<div class="stat-card"><div class="stat-number">468</div><div class="stat-label">Face Landmarks</div></div>', unsafe_allow_html=True)
        with col2: st.markdown('<div class="stat-card"><div class="stat-number">3D</div><div class="stat-label">Tracking</div></div>', unsafe_allow_html=True)
        with col3: st.markdown('<div class="stat-card"><div class="stat-number">Real-Time</div><div class="stat-label">Processing</div></div>', unsafe_allow_html=True)
        
        input_mode = st.radio("📥 Select Input Mode:", ["📷 Upload Image", "🎥 Live Webcam"], key="facemesh_input", horizontal=True)
        
        if "Upload" in input_mode:
            file = st.file_uploader("📤 Drop your image here", type=["jpg", "jpeg", "png"], key="facemesh_file")
            if file:
                img = Image.open(file)
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                with st.spinner("😊 Mapping 468 facial landmarks..."):
                    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
                        result = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                        if result.multi_face_landmarks:
                            st.markdown(f'<div class="status-badge badge-success">✅ Mapped {len(result.multi_face_landmarks)} face(s) with 468 landmarks</div>', unsafe_allow_html=True)
                            for face_landmarks in result.multi_face_landmarks:
                                mp_drawing.draw_landmarks(image=img_bgr, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                                mp_drawing.draw_landmarks(image=img_bgr, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                                mp_drawing.draw_landmarks(image=img_bgr, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_IRISES,
                                    landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                        else:
                            st.warning("❌ No faces detected")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📥 Original Image**\n<div class='image-container'>", unsafe_allow_html=True)
                    st.image(img, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown("**📤 Face Mesh Result**\n<div class='image-container'>", unsafe_allow_html=True)
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("### 🎥 Live Face Mesh Tracking")
            webcam_run = st.checkbox("▶️ START WEBCAM", key="facemesh_webcam_run")
            frame_display, status_display = st.empty(), st.empty()
            if webcam_run:
                cap = cv2.VideoCapture(0)
                status_display.markdown('<div class="status-badge badge-success">🟢 Webcam Active - Tracking 468 points</div>', unsafe_allow_html=True)
                with mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                    while webcam_run:
                        ret, frame = cap.read()
                        if not ret: break
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = face_mesh.process(rgb)
                        if result.multi_face_landmarks:
                            for face_landmarks in result.multi_face_landmarks:
                                mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                                mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                cap.release()
    
    # POSE DETECTION
    elif "Pose" in feature:
        st.markdown("## 🧍 Pose Detection - 33 Body Landmarks\n<p style='font-size:1.1rem; color:#666'>Full-body tracking • 3D pose estimation • Real-time motion capture</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown('<div class="stat-card"><div class="stat-number">33</div><div class="stat-label">Body Points</div></div>', unsafe_allow_html=True)
        with col2: st.markdown('<div class="stat-card"><div class="stat-number">3D</div><div class="stat-label">Coordinates</div></div>', unsafe_allow_html=True)
        with col3: st.markdown('<div class="stat-card"><div class="stat-number">30 FPS</div><div class="stat-label">Real-Time</div></div>', unsafe_allow_html=True)
        
        with st.expander("📖 Body Landmarks Guide"):
            st.markdown('<div class="info-card">**Upper Body:** Nose, Eyes, Ears, Mouth, Shoulders, Elbows, Wrists, Hands<br>**Torso:** Chest, Shoulders, Hips<br>**Lower Body:** Hips, Knees, Ankles, Heels, Feet</div>', unsafe_allow_html=True)
        
        input_mode = st.radio("📥 Select Input Mode:", ["📷 Upload Image", "🎥 Live Webcam"], key="pose_input", horizontal=True)
        
        if "Upload" in input_mode:
            file = st.file_uploader("📤 Drop your image here", type=["jpg", "jpeg", "png"], key="pose_file")
            if file:
                img = Image.open(file)
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                with st.spinner("🧍 Analyzing body pose..."):
                    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5) as pose:
                        result = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                        if result.pose_landmarks:
                            st.markdown('<div class="status-badge badge-success">✅ Detected pose with 33 landmarks</div>', unsafe_allow_html=True)
                            mp_drawing.draw_landmarks(img_bgr, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                            if result.segmentation_mask is not None:
                                condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.1
                                bg_image = np.zeros(img_bgr.shape, dtype=np.uint8)
                                bg_image[:] = (192, 192, 192)
                                annotated_image = np.where(condition, img_bgr, bg_image)
                            else:
                                annotated_image = img_bgr
                        else:
                            st.warning("❌ No pose detected")
                            annotated_image = img_bgr
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📥 Original Image**\n<div class='image-container'>", unsafe_allow_html=True)
                    st.image(img, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown("**📤 Pose Detection Result**\n<div class='image-container'>", unsafe_allow_html=True)
                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("### 🎥 Live Pose Tracking")
            webcam_run = st.checkbox("▶️ START WEBCAM", key="pose_webcam_run")
            frame_display, status_display = st.empty(), st.empty()
            if webcam_run:
                cap = cv2.VideoCapture(0)
                status_display.markdown('<div class="status-badge badge-success">🟢 Webcam Active - Full body tracking</div>', unsafe_allow_html=True)
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while webcam_run:
                        ret, frame = cap.read()
                        if not ret: break
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = pose.process(rgb)
                        if result.pose_landmarks:
                            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                cap.release()
    
    # GESTURE RECOGNITION (5 HANDS)
    elif "Gesture" in feature:
        st.markdown("## ✋ Multi-Hand Gesture Recognition AI\n<p style='font-size:1.1rem; color:#666'>**Up to 5 hands simultaneously** • 7 gesture types • High accuracy (70% threshold)</p>", unsafe_allow_html=True)
        
        st.markdown("""
        <aside class="info-section">
            <div class="info-card">
                <h2 style="text-align:center; color:#667eea">🎯 Recognized Gestures</h2>
                <div class="gestures-list">
                    <div class="gesture-item"><div class="gesture-icon">✌️</div><p class="gesture-name">Peace</p></div>
                    <div class="gesture-item"><div class="gesture-icon">🤚</div><p class="gesture-name">Open Hand</p></div>
                    <div class="gesture-item"><div class="gesture-icon">👍</div><p class="gesture-name">Thumbs Up</p></div>
                    <div class="gesture-item"><div class="gesture-icon">✊</div><p class="gesture-name">Fist</p></div>
                    <div class="gesture-item"><div class="gesture-icon">👎</div><p class="gesture-name">Thumbs Down</p></div>
                    <div class="gesture-item"><div class="gesture-icon">☝️</div><p class="gesture-name">Pointing Up</p></div>
                    <div class="gesture-item"><div class="gesture-icon">🤟</div><p class="gesture-name">I Love You</p></div>
                </div>
            </div>
        </aside>
        """, unsafe_allow_html=True)
        
        input_mode = st.radio("📥 Select Input Mode:", ["📷 Upload Image", "🎥 Live Webcam"], key="gesture_input", horizontal=True)
        
        if "Upload" in input_mode:
            file = st.file_uploader("📤 Drop your image here (show up to 5 hands)", type=["jpg", "jpeg", "png"], key="gesture_file")
            if file:
                img = Image.open(file)
                img_np = np.array(img)
                try:
                    with st.spinner("✋ Recognizing gestures from multiple hands..."):
                        recognizer = load_gesture_recognizer_multi()
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
                        result = recognizer.recognize(mp_img)
                        if result.gestures:
                            st.markdown(f'<div class="status-badge badge-success">✅ Detected {len(result.gestures)} hand(s)</div>', unsafe_allow_html=True)
                            results_html = '<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; margin: 20px 0;">'
                            for idx, (gesture, handedness) in enumerate(zip(result.gestures, result.handedness)):
                                gesture_name = gesture[0].category_name
                                gesture_score = gesture[0].score
                                hand_side = handedness[0].category_name
                                if gesture_score >= 0.7:
                                    if gesture_name in GESTURE_MAP:
                                        display_name, emoji = GESTURE_MAP[gesture_name]
                                    else:
                                        display_name = gesture_name.replace('_', ' ')
                                        emoji = "🤲"
                                    results_html += f'<div class="hand-result-box"><h3 style="margin:0 0 10px 0">Hand #{idx + 1}</h3><div style="font-size: 3.5rem; margin: 10px 0">{emoji}</div><div style="font-size: 1.4rem; font-weight: bold; margin: 8px 0">{display_name}</div><div style="font-size: 1rem; margin-top: 8px">{hand_side} Hand</div><div style="font-size: 0.9rem; margin-top: 5px; opacity: 0.9">Confidence: {gesture_score:.0%}</div></div>'
                            results_html += '</div>'
                            st.markdown(results_html, unsafe_allow_html=True)
                        else:
                            st.warning("❌ No hands detected")
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(img, use_column_width=True, caption="Uploaded Image")
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.markdown("### 🎥 Live Multi-Hand Gesture Recognition")
            st.info("👋 **Show up to 5 hands** - Each hand will be detected and tracked individually!")
            webcam_run = st.checkbox("▶️ START WEBCAM", key="gesture_webcam_run")
            col1, col2 = st.columns([2, 1])
            with col1: frame_display = st.empty()
            with col2: gesture_display = st.empty()
            status_display = st.empty()
            if webcam_run:
                try:
                    from mediapipe.tasks import python
                    from mediapipe.tasks.python import vision
                    model_path = get_model_path('gesture_recognizer.task')
                    base_opt = python.BaseOptions(model_asset_path=model_path)
                    opt = vision.GestureRecognizerOptions(base_options=base_opt, running_mode=vision.RunningMode.VIDEO,
                        num_hands=5, min_hand_detection_confidence=0.7, min_hand_presence_confidence=0.7, min_tracking_confidence=0.7)
                    recognizer = vision.GestureRecognizer.create_from_options(opt)
                    cap = cv2.VideoCapture(0)
                    frame_count = 0
                    status_display.markdown('<div class="status-badge badge-success">🟢 Webcam Active - Multi-hand mode (up to 5 hands)</div>', unsafe_allow_html=True)
                    while webcam_run:
                        ret, frame = cap.read()
                        if not ret: break
                        frame_count += 1
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        result = recognizer.recognize_for_video(mp_img, frame_count)
                        if result.hand_landmarks:
                            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                                color = HAND_COLORS[hand_idx % 5]
                                h, w, c = frame.shape
                                for landmark in hand_landmarks:
                                    x, y = int(landmark.x * w), int(landmark.y * h)
                                    cv2.circle(frame, (x, y), 5, color, -1)
                                if len(hand_landmarks) > 0:
                                    x = int(hand_landmarks[0].x * w)
                                    y = int(hand_landmarks[0].y * h)
                                    cv2.putText(frame, f"#{hand_idx + 1}", (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        if result.gestures:
                            gesture_html = f"<div style='padding:10px'><h3 style='color:#667eea'>🎯 {len(result.gestures)} Hand(s) Detected</h3>"
                            for idx, (gesture, handedness) in enumerate(zip(result.gestures, result.handedness)):
                                gesture_name = gesture[0].category_name
                                gesture_score = gesture[0].score
                                hand_side = handedness[0].category_name
                                if gesture_score >= 0.7:
                                    if gesture_name in GESTURE_MAP:
                                        display_name, emoji = GESTURE_MAP[gesture_name]
                                    else:
                                        display_name = gesture_name.replace('_', ' ')
                                        emoji = "🤲"
                                    gesture_html += f'<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 12px; border-radius: 10px; margin: 8px 0;"><div style="font-size: 1rem"><b>Hand #{idx + 1}</b> ({hand_side})</div><div style="font-size: 2.5rem; margin: 5px 0">{emoji}</div><div style="font-size: 1.2rem; font-weight: bold">{display_name}</div><div style="font-size: 0.85rem; margin-top: 5px">Confidence: {gesture_score:.0%}</div></div>'
                            gesture_html += '</div>'
                            gesture_display.markdown(gesture_html, unsafe_allow_html=True)
                        else:
                            gesture_display.info("👋 Show your hands (up to 5)")
                        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                    cap.release()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # OBJECT DETECTION
    elif "Object" in feature:
        st.markdown("## 📦 Object Detection Engine\n<p style='font-size:1.1rem; color:#666'>90+ object categories • Real-time detection • Bounding box visualization</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown('<div class="stat-card"><div class="stat-number">90+</div><div class="stat-label">Object Types</div></div>', unsafe_allow_html=True)
        with col2: st.markdown('<div class="stat-card"><div class="stat-number">5</div><div class="stat-label">Max Detections</div></div>', unsafe_allow_html=True)
        with col3: st.markdown('<div class="stat-card"><div class="stat-number">50%</div><div class="stat-label">Confidence</div></div>', unsafe_allow_html=True)
        
        input_mode = st.radio("📥 Select Input Mode:", ["📷 Upload Image", "🎥 Live Webcam"], key="object_input", horizontal=True)
        
        if "Upload" in input_mode:
            file = st.file_uploader("📤 Drop your image here", type=["jpg", "jpeg", "png"], key="object_file")
            if file:
                img = Image.open(file)
                img_np = np.array(img)
                try:
                    with st.spinner("📦 Detecting objects..."):
                        detector = load_object_detector_image()
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
                        result = detector.detect(mp_img)
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        if result.detections:
                            st.markdown(f'<div class="status-badge badge-success">✅ Detected {len(result.detections)} object(s)</div>', unsafe_allow_html=True)
                            detected_html = '<div style="margin:20px 0">'
                            for detection in result.detections:
                                bbox = detection.bounding_box
                                cv2.rectangle(img_bgr, (bbox.origin_x, bbox.origin_y), (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (0, 255, 0), 3)
                                label = f"{detection.categories[0].category_name}: {detection.categories[0].score:.2f}"
                                cv2.putText(img_bgr, label, (bbox.origin_x, bbox.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                obj_name = detection.categories[0].category_name
                                obj_conf = detection.categories[0].score
                                detected_html += f'<span class="feature-badge">🏷️ {obj_name} ({obj_conf:.0%})</span>'
                            detected_html += '</div>'
                            st.markdown(detected_html, unsafe_allow_html=True)
                        else:
                            st.warning("❌ No objects detected")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**📥 Original Image**\n<div class='image-container'>", unsafe_allow_html=True)
                            st.image(img, use_column_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown("**📤 Detection Result**\n<div class='image-container'>", unsafe_allow_html=True)
                            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.markdown("### 🎥 Live Object Detection")
            webcam_run = st.checkbox("▶️ START WEBCAM", key="object_webcam_run")
            frame_display, objects_display, status_display = st.empty(), st.empty(), st.empty()
            if webcam_run:
                try:
                    from mediapipe.tasks import python
                    from mediapipe.tasks.python import vision
                    model_path = get_model_path('efficientdet_lite0.tflite')
                    base_opt = python.BaseOptions(model_asset_path=model_path)
                    opt = vision.ObjectDetectorOptions(base_options=base_opt, max_results=5, score_threshold=0.5, running_mode=vision.RunningMode.VIDEO)
                    detector = vision.ObjectDetector.create_from_options(opt)
                    cap = cv2.VideoCapture(0)
                    frame_count = 0
                    status_display.markdown('<div class="status-badge badge-success">🟢 Webcam Active</div>', unsafe_allow_html=True)
                    while webcam_run:
                        ret, frame = cap.read()
                        if not ret: break
                        frame_count += 1
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        result = detector.detect_for_video(mp_img, frame_count)
                        if result.detections:
                            detected_objects = []
                            for detection in result.detections:
                                bbox = detection.bounding_box
                                cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y), (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (0, 255, 0), 3)
                                obj_name = detection.categories[0].category_name
                                detected_objects.append(obj_name)
                                cv2.putText(frame, obj_name, (bbox.origin_x, bbox.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            objects_display.markdown(f'<div class="status-badge badge-info">🎯 Detected: {", ".join(detected_objects)}</div>', unsafe_allow_html=True)
                        else:
                            objects_display.info("🔍 Searching...")
                        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                    cap.release()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # AUDIO CLASSIFICATION
    elif "Audio" in feature:
        st.markdown("## 🎵 Audio Classification AI\n<p style='font-size:1.1rem; color:#666'>521 audio event types • Music & speech detection • Environmental sound recognition</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown('<div class="stat-card"><div class="stat-number">521</div><div class="stat-label">Audio Types</div></div>', unsafe_allow_html=True)
        with col2: st.markdown('<div class="stat-card"><div class="stat-number">16kHz</div><div class="stat-label">Sample Rate</div></div>', unsafe_allow_html=True)
        with col3: st.markdown('<div class="stat-card"><div class="stat-number">Top 5</div><div class="stat-label">Results</div></div>', unsafe_allow_html=True)
        
        input_mode = st.radio("📥 Select Input Mode:", ["📁 Upload Audio File", "🎤 Live Microphone"], key="audio_input", horizontal=True)
        
        if "Upload" in input_mode:
            file = st.file_uploader("📤 Drop your .wav file here", type=["wav"], key="audio_file")
            if file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                st.audio(file)
                try:
                    from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
                    import wave
                    with st.spinner("🎵 Analyzing audio..."):
                        classifier = load_audio_classifier()
                        with wave.open(tmp_path, 'rb') as wav:
                            rate = wav.getframerate()
                            frames = wav.readframes(wav.getnframes())
                            audio_arr = np.frombuffer(frames, dtype=np.int16)
                        audio_float = audio_arr.astype(np.float32) / 32768.0
                        audio_obj = audio_data_module.AudioData.create_from_array(audio_float, rate)
                        classification_result_list = classifier.classify(audio_obj)
                        if classification_result_list:
                            st.markdown('<div class="status-badge badge-success">✅ Classification Complete!</div>', unsafe_allow_html=True)
                            for result in classification_result_list:
                                for classification in result.classifications:
                                    for idx, category in enumerate(classification.categories[:5]):
                                        conf = category.score * 100
                                        st.markdown(f'<div class="info-card"><b style="font-size:1.2rem">{idx+1}. {category.category_name}</b><div style="background:linear-gradient(90deg, #667eea 0%, #764ba2 100%); width:{conf}%; height:18px; border-radius:10px; margin-top:8px"></div><small style="color:#666; font-weight:600">{conf:.1f}% confidence</small></div>', unsafe_allow_html=True)
                        else:
                            st.warning("❌ No classification results")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(tmp_path): os.remove(tmp_path)
        else:
            st.markdown("### 🎤 Live Microphone Classification")
            mic_run = st.checkbox("▶️ START MICROPHONE", key="audio_mic_run")
            results_display, status_display = st.empty(), st.empty()
            if mic_run:
                try:
                    import pyaudio
                    from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
                    classifier = load_audio_classifier()
                    p = pyaudio.PyAudio()
                    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=15600)
                    status_display.markdown('<div class="status-badge badge-success">🟢 Microphone Active</div>', unsafe_allow_html=True)
                    while mic_run:
                        data = stream.read(15600, exception_on_overflow=False)
                        audio_arr = np.frombuffer(data, dtype=np.int16)
                        audio_float = audio_arr.astype(np.float32) / 32768.0
                        audio_obj = audio_data_module.AudioData.create_from_array(audio_float, 16000)
                        classification_result_list = classifier.classify(audio_obj)
                        if classification_result_list:
                            with results_display.container():
                                st.markdown("### 🎯 Live Classification:")
                                for result in classification_result_list:
                                    for classification in result.classifications:
                                        for idx, cat in enumerate(classification.categories[:5]):
                                            conf = cat.score * 100
                                            bar = "█" * int(conf / 5)
                                            st.markdown(f"**{idx+1}. {cat.category_name}**  \n`{bar}` {conf:.1f}%")
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                except ImportError:
                    st.error("❌ PyAudio not installed! Run: `pip install pyaudio`")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Footer
    st.markdown(f'<br><br><div class="footer"><p class="footer-text">✨ Built with <b>MediaPipe</b> & <b>Streamlit</b><br>🚀 VisionVerse - The AI Detection Universe</p><p style="margin-top:15px; color:#888">Powered by Google MediaPipe v{mp.__version__} • Advanced Computer Vision & Audio AI</p></div>', unsafe_allow_html=True)
