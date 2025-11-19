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
    
    .hand-result-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        display: inline-block;
        min-width: 180px;
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
mp_drawing_styles = mp.solutions.drawing_styles


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

# Hand color mapping for visualization (BGR format for OpenCV)
HAND_COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255)   # Magenta
]

# Simple gesture detection based on finger states
def detect_simple_gesture(hand_landmarks):
    """Detect basic gestures from hand landmarks"""
    # Get landmark positions
    landmarks = hand_landmarks.landmark
    
    # Check if fingers are extended (comparing finger tip with finger base)
    thumb_up = landmarks[4].y < landmarks[3].y
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    
    # Count extended fingers
    fingers_up = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])
    
    # Detect gestures
    if fingers_up == 0:
        return "Fist", "✊"
    elif fingers_up == 5:
        return "Open Hand", "🤚"
    elif fingers_up == 1 and index_up:
        return "Pointing Up", "☝️"
    elif fingers_up == 2 and index_up and middle_up:
        return "Peace", "✌️"
    elif fingers_up == 1 and thumb_up:
        return "Thumbs Up", "👍"
    elif fingers_up == 3 and index_up and middle_up and pinky_up:
        return "I Love You", "🤟"
    else:
        return f"{fingers_up} Fingers", "🤲"


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
    st.markdown("✅ **Multi-Hand Detection (Up to 5 hands)**")
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
    st.markdown("🚀 **Detect and recognize up to 5 hands simultaneously with individual gesture detection!**")
    
    # Display Recognized Gestures Section
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
        file = st.file_uploader("Upload Image (showing up to 5 hands)", type=["jpg", "jpeg", "png"], key="gesture_file")
        if file:
            img = Image.open(file)
            img_np = np.array(img)
            
            try:
                with st.spinner("✋ Detecting multiple hands..."):
                    # Use MediaPipe Hands with max 5 hands
                    with mp_hands.Hands(
                        static_image_mode=True,
                        max_num_hands=5,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    ) as hands:
                        results = hands.process(img_np)
                        
                        if results.multi_hand_landmarks:
                            num_hands = len(results.multi_hand_landmarks)
                            st.success(f"✅ Detected {num_hands} hand(s)!")
                            
                            # Draw on image
                            img_annotated = img_np.copy()
                            
                            # Display results for each hand
                            st.markdown("### 🎯 Individual Hand Results:")
                            results_html = '<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 15px;">'
                            
                            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                                # Draw landmarks with unique color
                                color = HAND_COLORS[idx % 5]
                                mp_drawing.draw_landmarks(
                                    img_annotated,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style()
                                )
                                
                                # Detect gesture
                                gesture_name, gesture_emoji = detect_simple_gesture(hand_landmarks)
                                hand_side = handedness.classification[0].label
                                confidence = handedness.classification[0].score
                                
                                # Add hand number label on image
                                h, w, c = img_annotated.shape
                                wrist = hand_landmarks.landmark[0]
                                x, y = int(wrist.x * w), int(wrist.y * h)
                                cv2.putText(img_annotated, f"Hand {idx+1}", (x-30, y-20),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                
                                # Create result card for this hand
                                results_html += f"""
                                <div class="hand-result-box">
                                    <h3 style="margin:0 0 10px 0">Hand #{idx + 1}</h3>
                                    <div style="font-size: 3.5rem; margin: 10px 0">{gesture_emoji}</div>
                                    <div style="font-size: 1.4rem; font-weight: bold; margin: 8px 0">{gesture_name}</div>
                                    <div style="font-size: 1rem; margin-top: 8px">{hand_side} Hand</div>
                                    <div style="font-size: 0.9rem; margin-top: 5px; opacity: 0.9">Confidence: {confidence:.0%}</div>
                                </div>
                                """
                            
                            results_html += '</div>'
                            st.markdown(results_html, unsafe_allow_html=True)
                            
                            # Show images
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**📥 Original Image**")
                                st.image(img, use_column_width=True)
                            with col2:
                                st.markdown("**📤 Detection Result**")
                                st.image(img_annotated, use_column_width=True)
                        else:
                            st.warning("❌ No hands detected - Try showing your hands clearly")
                            st.image(img, use_column_width=True)
                            
            except Exception as e:
                st.error(f"Error: {e}")
    
    else:
        st.markdown("**🎥 Live Webcam - Multi-Hand Gesture Recognition**")
        st.info("👋 Show up to 5 hands for simultaneous detection!")
        
        webcam_run = st.checkbox("▶️ START WEBCAM", key="gesture_webcam_run")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            frame_display = st.empty()
        with col2:
            gesture_display = st.empty()
        
        status_display = st.empty()
        
        if webcam_run:
            try:
                cap = cv2.VideoCapture(0)
                status_display.success("✅ Webcam Active - Show up to 5 hands!")
                
                with mp_hands.Hands(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=5
                ) as hands:
                    
                    while webcam_run:
                        ret, frame = cap.read()
                        if not ret:
                            status_display.error("❌ Cannot access webcam")
                            break
                        
                        # Flip for mirror view
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(rgb)
                        
                        if results.multi_hand_landmarks:
                            num_hands = len(results.multi_hand_landmarks)
                            
                            # Build results HTML
                            results_html = f'<div style="padding:10px"><h3 style="color:#667eea">🎯 {num_hands} Hand(s) Detected</h3>'
                            
                            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                                # Draw landmarks with unique color
                                color = HAND_COLORS[idx % 5]
                                
                                # Draw connections
                                mp_drawing.draw_landmarks(
                                    frame,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style()
                                )
                                
                                # Add hand number label
                                h, w, c = frame.shape
                                wrist = hand_landmarks.landmark[0]
                                x, y = int(wrist.x * w), int(wrist.y * h)
                                cv2.putText(frame, f"#{idx+1}", (x-20, y-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                
                                # Detect gesture
                                gesture_name, gesture_emoji = detect_simple_gesture(hand_landmarks)
                                hand_side = handedness.classification[0].label
                                confidence = handedness.classification[0].score
                                
                                # Add to results HTML
                                results_html += f"""
                                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                            color: white; padding: 12px; border-radius: 10px; margin: 8px 0;">
                                    <div style="font-size: 1rem"><b>Hand #{idx + 1}</b> ({hand_side})</div>
                                    <div style="font-size: 2.5rem; margin: 5px 0">{gesture_emoji}</div>
                                    <div style="font-size: 1.2rem; font-weight: bold">{gesture_name}</div>
                                    <div style="font-size: 0.85rem; margin-top: 5px">Confidence: {confidence:.0%}</div>
                                </div>
                                """
                            
                            results_html += '</div>'
                            gesture_display.markdown(results_html, unsafe_allow_html=True)
                        else:
                            gesture_display.info("👋 Show your hands (up to 5) to detect gestures")
                        
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
    <p>✨ Built with MediaPipe & Streamlit | 🚀 Multi-Hand Detection (Up to 5 Hands Simultaneously)</p>
</div>
""", unsafe_allow_html=True)
