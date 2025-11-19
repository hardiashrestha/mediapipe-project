import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(
    page_title="MediaPipe Detection App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced UI matching your design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    
    .info-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        border: 2px solid #e0e0e0;
    }
    
    .info-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .gestures-list {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        justify-content: center;
        margin-top: 15px;
    }
    
    .gesture-item {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        text-align: center;
        min-width: 120px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .gesture-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    .gesture-icon {
        font-size: 2.5rem;
        margin-bottom: 5px;
    }
    
    .gesture-name {
        font-size: 1rem;
        font-weight: bold;
        margin: 0;
    }
    
    .detected-gesture-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎯 MediaPipe Multi-Detection App</h1>', unsafe_allow_html=True)

# Initialize MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# FIXED: Cache model loading
@st.cache_resource
def get_model_path(model_name):
    """Returns normalized absolute path - CACHED"""
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, 'models', model_name)
    return os.path.normpath(model_path)

@st.cache_resource
def load_gesture_recognizer():
    """Load and cache gesture recognizer"""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    model_path = get_model_path('gesture_recognizer.task')
    base_opt = python.BaseOptions(model_asset_path=model_path)
    opt = vision.GestureRecognizerOptions(base_options=base_opt)
    return vision.GestureRecognizer.create_from_options(opt)

@st.cache_resource
def load_object_detector_image():
    """Load object detector for images"""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    model_path = get_model_path('efficientdet_lite0.tflite')
    base_opt = python.BaseOptions(model_asset_path=model_path)
    opt = vision.ObjectDetectorOptions(base_options=base_opt, max_results=5, score_threshold=0.5)
    return vision.ObjectDetector.create_from_options(opt)

@st.cache_resource
def load_audio_classifier():
    """Load and cache audio classifier"""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import audio
    
    model_path = get_model_path('yamnet.tflite')
    base_opt = python.BaseOptions(model_asset_path=model_path)
    opt = audio.AudioClassifierOptions(base_options=base_opt, max_results=5)
    return audio.AudioClassifier.create_from_options(opt)

# Gesture mapping - Matching your code snippet
GESTURE_MAP = {
    (0, 1, 0, 0, 0): ("Peace", "✌️"),
    (1, 1, 1, 1, 1): ("Open_Palm", "🤚"),
    (0, 0, 0, 0, 0): ("Thumb_Up", "👍"),
    (1, 0, 0, 0, 0): ("Closed_Fist", "✊"),
    "Thumb_Up": ("Thumbs Up", "👍"),
    "Thumb_Down": ("Thumbs Down", "👎"),
    "Victory": ("Peace", "✌️"),
    "Open_Palm": ("Open Hand", "🤚"),
    "Closed_Fist": ("Fist", "✊"),
    "Pointing_Up": ("Pointing Up", "☝️"),
    "ILoveYou": ("I Love You", "🤟")
}

# Sidebar
with st.sidebar:
    st.title("🎨 Navigation")
    feature = st.selectbox(
        "Choose Feature:",
        ["🔍 Face Detection", "✋ Gesture Recognition", "📦 Object Detection", "🎵 Audio Classification"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Features")
    st.markdown("✅ Real-time Detection")
    st.markdown("✅ Multi-model Support")
    st.markdown("✅ High Accuracy")
    st.markdown("---")
    st.info(f"**MediaPipe v{mp.__version__}**")

# ==================== FACE DETECTION ====================
if "Face Detection" in feature:
    st.header("👤 Face Detection")
    st.markdown("Detect faces with 6 key facial landmarks in real-time")
    
    input_mode = st.radio("Select Input:", ["📷 Upload Image", "🎥 Live Webcam"], key="face_input", horizontal=True)
    
    if "Upload" in input_mode:
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="face_file")
        if file:
            img = Image.open(file)
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            with st.spinner("🔍 Detecting faces..."):
                with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
                    result = detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    if result.detections:
                        st.success(f"✅ Detected {len(result.detections)} face(s)")
                        for detection in result.detections:
                            mp_drawing.draw_detection(img_bgr, detection)
                    else:
                        st.warning("❌ No faces detected")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📥 Original Image**")
                st.image(img, width=400)
            with col2:
                st.markdown("**📤 Detection Result**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=400)
    
    else:
        st.markdown("**🎥 Live Webcam Face Detection**")
        webcam_run = st.checkbox("▶️ START WEBCAM", key="face_webcam_run")
        frame_display = st.empty()
        status_display = st.empty()
        
        if webcam_run:
            cap = cv2.VideoCapture(0)
            status_display.success("✅ Webcam Active - Detecting faces...")
            
            with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
                while webcam_run:
                    ret, frame = cap.read()
                    if not ret:
                        status_display.error("❌ Cannot access webcam")
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = detector.process(rgb)
                    if result.detections:
                        for detection in result.detections:
                            mp_drawing.draw_detection(frame, detection)
                    frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            cap.release()

# ==================== GESTURE RECOGNITION ====================
elif "Gesture" in feature:
    st.header("✋ Hand Gesture Recognition")
    st.markdown("Recognize hand gestures in real-time with AI")
    
    # Display Recognized Gestures Section - Matching your HTML design
    st.markdown("""
    <aside class="info-section">
        <div class="info-card">
            <h2>Recognized Gestures</h2>
            <div class="gestures-list">
                <div class="gesture-item">
                    <div class="gesture-icon">✌️</div>
                    <p class="gesture-name">Peace</p>
                </div>
                <div class="gesture-item">
                    <div class="gesture-icon">🤚</div>
                    <p class="gesture-name">Open Hand</p>
                </div>
                <div class="gesture-item">
                    <div class="gesture-icon">👍</div>
                    <p class="gesture-name">Thumbs Up</p>
                </div>
                <div class="gesture-item">
                    <div class="gesture-icon">✊</div>
                    <p class="gesture-name">Fist</p>
                </div>
                <div class="gesture-item">
                    <div class="gesture-icon">👎</div>
                    <p class="gesture-name">Thumbs Down</p>
                </div>
                <div class="gesture-item">
                    <div class="gesture-icon">☝️</div>
                    <p class="gesture-name">Pointing Up</p>
                </div>
                <div class="gesture-item">
                    <div class="gesture-icon">🤟</div>
                    <p class="gesture-name">I Love You</p>
                </div>
            </div>
        </div>
    </aside>
    """, unsafe_allow_html=True)
    
    input_mode = st.radio("Select Input:", ["📷 Upload Image", "🎥 Live Webcam"], key="gesture_input", horizontal=True)
    
    if "Upload" in input_mode:
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="gesture_file")
        if file:
            img = Image.open(file)
            img_np = np.array(img)
            
            try:
                with st.spinner("✋ Recognizing gestures..."):
                    recognizer = load_gesture_recognizer()
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
                    result = recognizer.recognize(mp_img)
                    
                    if result.gestures:
                        st.success(f"✅ Detected {len(result.gestures)} hand(s)")
                        
                        # Display detected gestures with your design
                        for idx, (gesture, handedness) in enumerate(zip(result.gestures, result.handedness)):
                            gesture_name = gesture[0].category_name
                            gesture_score = gesture[0].score
                            hand_side = handedness[0].category_name
                            
                            # Get emoji and display name
                            if gesture_name in GESTURE_MAP:
                                display_name, emoji = GESTURE_MAP[gesture_name]
                            else:
                                display_name = gesture_name.replace('_', ' ')
                                emoji = "🤲"
                            
                            st.markdown(f"""
                            <div class="detected-gesture-box">
                                <div style="font-size: 4rem; margin-bottom: 10px">{emoji}</div>
                                <div style="font-size: 2rem">{display_name}</div>
                                <div style="font-size: 1rem; margin-top: 10px">{hand_side} Hand | Confidence: {gesture_score:.0%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("❌ No hands detected - Try showing your hand clearly")
                
                st.image(img, width=600, caption="Uploaded Image")
            except Exception as e:
                st.error(f"Error: {e}")
    
    else:
        st.markdown("**🎥 Live Webcam Gesture Recognition**")
        webcam_run = st.checkbox("▶️ START WEBCAM", key="gesture_webcam_run")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            frame_display = st.empty()
        with col2:
            gesture_display = st.empty()
        
        status_display = st.empty()
        
        if webcam_run:
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                # Load model for video mode
                model_path = get_model_path('gesture_recognizer.task')
                base_opt = python.BaseOptions(model_asset_path=model_path)
                opt = vision.GestureRecognizerOptions(
                    base_options=base_opt,
                    running_mode=vision.RunningMode.VIDEO
                )
                recognizer = vision.GestureRecognizer.create_from_options(opt)
                
                cap = cv2.VideoCapture(0)
                frame_count = 0
                status_display.success("✅ Webcam Active - Show your gestures!")
                
                while webcam_run:
                    ret, frame = cap.read()
                    if not ret:
                        status_display.error("❌ Cannot access webcam")
                        break
                    
                    frame_count += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to MediaPipe image
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    
                    # Recognize gestures
                    result = recognizer.recognize_for_video(mp_img, frame_count)
                    
                    # Draw hand landmarks
                    if result.hand_landmarks:
                        for hand_landmarks in result.hand_landmarks:
                            # Draw landmarks on frame
                            h, w, c = frame.shape
                            for landmark in hand_landmarks:
                                x = int(landmark.x * w)
                                y = int(landmark.y * h)
                                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    
                    # Display detected gestures
                    if result.gestures:
                        for gesture, handedness in zip(result.gestures, result.handedness):
                            gesture_name = gesture[0].category_name
                            gesture_score = gesture[0].score
                            hand_side = handedness[0].category_name
                            
                            if gesture_name in GESTURE_MAP:
                                display_name, emoji = GESTURE_MAP[gesture_name]
                            else:
                                display_name = gesture_name.replace('_', ' ')
                                emoji = "🤲"
                            
                            gesture_display.markdown(f"""
                            <div class="detected-gesture-box">
                                <div style="font-size: 3rem">{emoji}</div>
                                <div style="font-size: 1.5rem">{display_name}</div>
                                <div style="font-size: 0.9rem; margin-top: 5px">{hand_side} | {gesture_score:.0%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        gesture_display.info("👋 Show your hand to detect gestures")
                    
                    frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                
                cap.release()
            except Exception as e:
                st.error(f"Error: {e}")

# ==================== OBJECT DETECTION ====================
elif "Object" in feature:
    st.header("📦 Object Detection")
    st.markdown("Detect 90+ common objects in real-time")
    
    input_mode = st.radio("Select Input:", ["📷 Upload Image", "🎥 Live Webcam"], key="object_input", horizontal=True)
    
    if "Upload" in input_mode:
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="object_file")
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
                        st.success(f"✅ Detected {len(result.detections)} object(s)")
                        
                        detected_badges = ""
                        for detection in result.detections:
                            bbox = detection.bounding_box
                            cv2.rectangle(img_bgr, 
                                        (bbox.origin_x, bbox.origin_y),
                                        (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                                        (0, 255, 0), 3)
                            label = f"{detection.categories[0].category_name}: {detection.categories[0].score:.2f}"
                            cv2.putText(img_bgr, label, (bbox.origin_x, bbox.origin_y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            obj_name = detection.categories[0].category_name
                            obj_conf = detection.categories[0].score
                            detected_badges += f"🏷️ **{obj_name}** ({obj_conf:.0%}) | "
                        
                        st.markdown(detected_badges)
                    else:
                        st.warning("❌ No objects detected")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📥 Original Image**")
                        st.image(img, width=400)
                    with col2:
                        st.markdown("**📤 Detection Result**")
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=400)
            except Exception as e:
                st.error(f"Error: {e}")
    
    else:
        st.markdown("**🎥 Live Webcam Object Detection**")
        webcam_run = st.checkbox("▶️ START WEBCAM", key="object_webcam_run")
        
        frame_display = st.empty()
        objects_display = st.empty()
        status_display = st.empty()
        
        if webcam_run:
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                model_path = get_model_path('efficientdet_lite0.tflite')
                base_opt = python.BaseOptions(model_asset_path=model_path)
                opt = vision.ObjectDetectorOptions(
                    base_options=base_opt,
                    max_results=5,
                    score_threshold=0.5,
                    running_mode=vision.RunningMode.VIDEO
                )
                detector = vision.ObjectDetector.create_from_options(opt)
                
                cap = cv2.VideoCapture(0)
                frame_count = 0
                status_display.success("✅ Webcam Active - Point at objects!")
                
                while webcam_run:
                    ret, frame = cap.read()
                    if not ret:
                        status_display.error("❌ Cannot access webcam")
                        break
                    
                    frame_count += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect_for_video(mp_img, frame_count)
                    
                    detected_objects = []
                    if result.detections:
                        for detection in result.detections:
                            bbox = detection.bounding_box
                            cv2.rectangle(frame,
                                        (bbox.origin_x, bbox.origin_y),
                                        (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                                        (0, 255, 0), 3)
                            obj_name = detection.categories[0].category_name
                            detected_objects.append(obj_name)
                            cv2.putText(frame, obj_name, (bbox.origin_x, bbox.origin_y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        objects_display.success(f"🎯 **Detected:** {', '.join(detected_objects)}")
                    else:
                        objects_display.info("🔍 Searching for objects...")
                    
                    frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                
                cap.release()
            except Exception as e:
                st.error(f"Error: {e}")

# ==================== AUDIO CLASSIFICATION ====================
elif "Audio" in feature:
    st.header("🎵 Audio Classification")
    st.markdown("Classify 521 audio events including music, speech, and environmental sounds")
    
    input_mode = st.radio("Select Input:", ["📁 Upload Audio File", "🎤 Live Microphone"], key="audio_input", horizontal=True)
    
    if "Upload" in input_mode:
        file = st.file_uploader("Upload Audio (.wav)", type=["wav"], key="audio_file")
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            st.audio(file)
            
            try:
                from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
                import wave
                
                with st.spinner("🎵 Classifying audio..."):
                    classifier = load_audio_classifier()
                    
                    with wave.open(tmp_path, 'rb') as wav:
                        rate = wav.getframerate()
                        frames = wav.readframes(wav.getnframes())
                        audio_arr = np.frombuffer(frames, dtype=np.int16)
                    
                    audio_float = audio_arr.astype(np.float32) / 32768.0
                    audio_obj = audio_data_module.AudioData.create_from_array(audio_float, rate)
                    
                    classification_result_list = classifier.classify(audio_obj)
                    
                    if classification_result_list:
                        st.success("✅ Classification Complete!")
                        
                        for result in classification_result_list:
                            for classification in result.classifications:
                                for idx, category in enumerate(classification.categories[:5]):
                                    conf = category.score * 100
                                    st.markdown(f"""
                                    <div class="info-card">
                                        <b>{idx+1}. {category.category_name}</b>
                                        <div style="background:linear-gradient(90deg, #667eea 0%, #764ba2 100%); width:{conf}%; height:15px; border-radius:5px; margin-top:5px"></div>
                                        <small>{conf:.1f}% confidence</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.warning("❌ No classification results")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    else:
        st.markdown("**🎤 Live Microphone Audio Classification**")
        mic_run = st.checkbox("▶️ START MICROPHONE", key="audio_mic_run")
        results_display = st.empty()
        status_display = st.empty()
        
        if mic_run:
            try:
                import pyaudio
                from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
                
                CHUNK = 15600
                FORMAT = pyaudio.paInt16
                CHANNELS = 1
                RATE = 16000
                
                classifier = load_audio_classifier()
                
                p = pyaudio.PyAudio()
                stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                              input=True, frames_per_buffer=CHUNK)
                
                status_display.success("🎤 Microphone Active - Make some sounds!")
                
                while mic_run:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_arr = np.frombuffer(data, dtype=np.int16)
                    audio_float = audio_arr.astype(np.float32) / 32768.0
                    audio_obj = audio_data_module.AudioData.create_from_array(audio_float, RATE)
                    
                    classification_result_list = classifier.classify(audio_obj)
                    
                    if classification_result_list:
                        with results_display.container():
                            st.markdown("### 🎯 Live Classification Results:")
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
                st.error("❌ PyAudio not installed!")
                st.code("pip install pyaudio")
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666'>
    <p>✨ Built with MediaPipe & Streamlit | 🚀 All Models Cached for Performance</p>
</div>
""", unsafe_allow_html=True)
