import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(page_title="MediaPipe App", layout="wide")
st.title("🎯 MediaPipe Detection App")

# Sidebar
st.sidebar.title("Select Feature")
feature = st.sidebar.selectbox(
    "Choose:",
    ["Face Detection", "Gesture Recognition", "Object Detection", "Audio Classification"]
)

# Initialize MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Get model path - FIXED FOR WINDOWS
def get_model_path(model_name):
    """Returns normalized absolute path"""
    current_dir = os.path.dirname(os.path.realpath(__file__)) if '__file__' in globals() else os.getcwd()
    model_path = os.path.join(current_dir, 'models', model_name)
    normalized_path = os.path.normpath(model_path)
    return normalized_path

# ==================== FACE DETECTION ====================
if feature == "Face Detection":
    st.header("👤 Face Detection")
    
    mode = st.radio("Input Type:", ["Image", "Webcam"], key="face_mode")
    
    if mode == "Image":
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="face_upload")
        if file:
            img = Image.open(file)
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
                result = detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                if result.detections:
                    st.success(f"✅ Detected {len(result.detections)} face(s)")
                    for detection in result.detections:
                        mp_drawing.draw_detection(img_bgr, detection)
                else:
                    st.warning("No faces detected")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original**")
                st.image(img, width=400)
            with col2:
                st.write("**Result**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=400)
    
    elif mode == "Webcam":
        st.write("**Webcam Mode**")
        start = st.checkbox("▶️ Start Webcam", key="face_start")
        frame_placeholder = st.empty()
        
        if start:
            cap = cv2.VideoCapture(0)
            with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
                while start:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = detector.process(rgb)
                    if result.detections:
                        for detection in result.detections:
                            mp_drawing.draw_detection(frame, detection)
                    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

# ==================== GESTURE RECOGNITION ====================
elif feature == "Gesture Recognition":
    st.header("✋ Gesture Recognition")
    
    mode = st.radio("Input Type:", ["Image", "Webcam"], key="gesture_mode")
    
    if mode == "Image":
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="gesture_upload")
        if file:
            img = Image.open(file)
            img_np = np.array(img)
            
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                model_path = get_model_path('gesture_recognizer.task')
                
                base_opt = python.BaseOptions(model_asset_path=model_path)
                opt = vision.GestureRecognizerOptions(base_options=base_opt)
                recognizer = vision.GestureRecognizer.create_from_options(opt)
                
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
                result = recognizer.recognize(mp_img)
                
                if result.gestures:
                    st.success(f"✅ Detected {len(result.gestures)} hand(s)")
                    for idx, gesture in enumerate(result.gestures):
                        st.write(f"**Gesture {idx+1}:** {gesture[0].category_name} ({gesture[0].score:.2f})")
                else:
                    st.warning("No hands detected")
                
                st.image(img, width=600)
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif mode == "Webcam":
        st.write("**Webcam Mode**")
        start = st.checkbox("▶️ Start Webcam", key="gesture_start")
        frame_placeholder = st.empty()
        
        if start:
            cap = cv2.VideoCapture(0)
            with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                while start:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)
                    if result.multi_hand_landmarks:
                        for hand in result.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

# ==================== OBJECT DETECTION ====================
elif feature == "Object Detection":
    st.header("📦 Object Detection")
    
    # FIXED: Changed key to avoid conflicts
    mode = st.radio("Input Type:", ["Image", "Webcam"], key="object_mode")
    
    if mode == "Image":
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="object_upload")
        if file:
            img = Image.open(file)
            img_np = np.array(img)
            
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                model_path = get_model_path('efficientdet_lite0.tflite')
                
                if not os.path.exists(model_path):
                    st.error(f"❌ Model file not found at: {model_path}")
                else:
                    base_opt = python.BaseOptions(model_asset_path=model_path)
                    opt = vision.ObjectDetectorOptions(base_options=base_opt, max_results=5, score_threshold=0.5)
                    detector = vision.ObjectDetector.create_from_options(opt)
                    
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
                    result = detector.detect(mp_img)
                    
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    
                    if result.detections:
                        st.success(f"✅ Detected {len(result.detections)} object(s)")
                        for detection in result.detections:
                            bbox = detection.bounding_box
                            cv2.rectangle(img_bgr, 
                                        (bbox.origin_x, bbox.origin_y),
                                        (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                                        (0, 255, 0), 2)
                            label = f"{detection.categories[0].category_name}: {detection.categories[0].score:.2f}"
                            cv2.putText(img_bgr, label, (bbox.origin_x, bbox.origin_y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            st.write(f"**{detection.categories[0].category_name}** - {detection.categories[0].score:.2f}")
                    else:
                        st.warning("No objects detected")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original**")
                        st.image(img, width=400)
                    with col2:
                        st.write("**Result**")
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=400)
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif mode == "Webcam":
        # *** WEBCAM FOR OBJECT DETECTION - NOW VISIBLE ***
        st.write("**🎥 Webcam Mode - Real-Time Object Detection**")
        st.info("Check the box below to start detecting objects")
        
        start = st.checkbox("▶️ Start Webcam", key="object_start")
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        
        if start:
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                model_path = get_model_path('efficientdet_lite0.tflite')
                
                if not os.path.exists(model_path):
                    st.error(f"❌ Model file not found at: {model_path}")
                else:
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
                    
                    info_placeholder.success("✅ Webcam started! Point at objects...")
                    
                    while start:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        
                        result = detector.detect_for_video(mp_img, frame_count)
                        
                        objects = []
                        if result.detections:
                            for detection in result.detections:
                                bbox = detection.bounding_box
                                cv2.rectangle(frame,
                                            (bbox.origin_x, bbox.origin_y),
                                            (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                                            (0, 255, 0), 3)
                                label = f"{detection.categories[0].category_name}"
                                objects.append(label)
                                cv2.putText(frame, label, (bbox.origin_x, bbox.origin_y - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            info_placeholder.success(f"✅ Detected: {', '.join(objects)}")
                        else:
                            info_placeholder.info("🔍 Looking for objects...")
                        
                        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    cap.release()
            except Exception as e:
                st.error(f"Error: {e}")

# ==================== AUDIO CLASSIFICATION ====================
elif feature == "Audio Classification":
    st.header("🎵 Audio Classification")
    
    mode = st.radio("Input Type:", ["Audio File", "Microphone"], key="audio_mode")
    
    if mode == "Audio File":
        file = st.file_uploader("Upload Audio (.wav)", type=["wav"], key="audio_upload")
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            st.audio(file)
            
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import audio
                from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
                import wave
                
                model_path = get_model_path('yamnet.tflite')
                
                if not os.path.exists(model_path):
                    st.error(f"❌ Model file not found at: {model_path}")
                else:
                    base_opt = python.BaseOptions(model_asset_path=model_path)
                    opt = audio.AudioClassifierOptions(base_options=base_opt, max_results=5)
                    classifier = audio.AudioClassifier.create_from_options(opt)
                    
                    with wave.open(tmp_path, 'rb') as wav:
                        rate = wav.getframerate()
                        frames = wav.readframes(wav.getnframes())
                        audio_arr = np.frombuffer(frames, dtype=np.int16)
                    
                    audio_float = audio_arr.astype(np.float32) / 32768.0
                    audio_obj = audio_data_module.AudioData.create_from_array(audio_float, rate)
                    
                    # Classify audio
                    result = classifier.classify(audio_obj)
                    
                    # FIXED: Correct way to access results
                    if result and len(result) > 0:
                        st.success("✅ Classification Results:")
                        
                        # Result is a list of AudioClassifierResult objects
                        classification_list = result[0]
                        
                        # Get top 5 categories
                        for idx, category in enumerate(classification_list.classifications[0].categories[:5]):
                            st.write(f"**{idx+1}. {category.category_name}** - {category.score:.4f}")
                    else:
                        st.warning("No results")
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    elif mode == "Microphone":
        # *** MICROPHONE FOR AUDIO ***
        st.write("**🎤 Microphone Mode - Real-Time Audio Classification**")
        st.info("Check the box below to start listening")
        
        start = st.checkbox("▶️ Start Microphone", key="audio_start")
        results_placeholder = st.empty()
        
        if start:
            try:
                import pyaudio
                from mediapipe.tasks import python
                from mediapipe.tasks.python import audio
                from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
                
                CHUNK = 15600
                FORMAT = pyaudio.paInt16
                CHANNELS = 1
                RATE = 16000
                
                model_path = get_model_path('yamnet.tflite')
                
                if not os.path.exists(model_path):
                    st.error(f"❌ Model file not found at: {model_path}")
                else:
                    base_opt = python.BaseOptions(model_asset_path=model_path)
                    opt = audio.AudioClassifierOptions(base_options=base_opt, max_results=5)
                    classifier = audio.AudioClassifier.create_from_options(opt)
                    
                    p = pyaudio.PyAudio()
                    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                  input=True, frames_per_buffer=CHUNK)
                    
                    st.success("🎤 Listening... Speak or make sounds!")
                    
                    while start:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        audio_arr = np.frombuffer(data, dtype=np.int16)
                        audio_float = audio_arr.astype(np.float32) / 32768.0
                        audio_obj = audio_data_module.AudioData.create_from_array(audio_float, RATE)
                        
                        result = classifier.classify(audio_obj)
                        
                        # FIXED: Correct way to access real-time results
                        if result and len(result) > 0:
                            classification_list = result[0]
                            
                            with results_placeholder.container():
                                st.markdown("### 🎯 Live Results:")
                                for idx, cat in enumerate(classification_list.classifications[0].categories[:5]):
                                    conf = cat.score * 100
                                    bar = "█" * int(conf / 5)
                                    st.markdown(f"**{idx+1}. {cat.category_name}**  \n`{bar}` {conf:.1f}%")
                    
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                
            except ImportError:
                st.error("❌ Install PyAudio: pip install pyaudio")
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built with MediaPipe")
