import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Streamlit page configuration
st.set_page_config(page_title="MediaPipe Multi-Feature App", layout="wide")
st.title("🎯 MediaPipe Multi-Feature Detection App")
st.markdown("**Face Detection | Gesture Recognition | Audio Classification | Object Detection**")

# Sidebar for feature selection
st.sidebar.title("Choose Detection Type")
feature = st.sidebar.selectbox(
    "Select Feature:",
    ["Face Detection", "Gesture Recognition", "Object Detection", "Audio Classification"]
)

# ==================== FACE DETECTION ====================
if feature == "Face Detection":
    st.header("👤 Face Detection")
    st.write("Upload an image or use webcam to detect faces")
    
    input_type = st.radio("Choose input type:", ["Upload Image", "Use Webcam"])
    
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert RGB to BGR for MediaPipe
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Face detection
            with mp_face_detection.FaceDetection(
                model_selection=1, 
                min_detection_confidence=0.5
            ) as face_detection:
                results = face_detection.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                
                # Draw face detections
                if results.detections:
                    st.success(f"✅ Detected {len(results.detections)} face(s)")
                    for detection in results.detections:
                        mp_drawing.draw_detection(image_bgr, detection)
                else:
                    st.warning("❌ No faces detected")
            
            # Convert back to RGB for display
            output_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader("Detection Result")
                st.image(output_image, use_container_width=True)
    
    else:  # Webcam option
        st.info("🎥 Click 'Start' to use webcam")
        run_webcam = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])
        
        if run_webcam:
            cap = cv2.VideoCapture(0)
            
            with mp_face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=0.5
            ) as face_detection:
                
                while run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Cannot access webcam")
                        break
                    
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process
                    results = face_detection.process(image_rgb)
                    
                    # Draw detections
                    if results.detections:
                        for detection in results.detections:
                            mp_drawing.draw_detection(frame, detection)
                    
                    # Display
                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            cap.release()

# ==================== GESTURE RECOGNITION ====================
elif feature == "Gesture Recognition":
    st.header("✋ Hand Gesture Recognition")
    st.write("Recognizes: Thumbs Up, Thumbs Down, Victory, Open Palm, Closed Fist, Pointing Up, ILoveYou")
    
    input_type = st.radio("Choose input type:", ["Upload Image", "Use Webcam"])
    
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Import gesture recognizer
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Create gesture recognizer
            base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
            options = vision.GestureRecognizerOptions(base_options=base_options)
            recognizer = vision.GestureRecognizer.create_from_options(options)
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            
            # Recognize gestures
            recognition_result = recognizer.recognize(mp_image)
            
            # Display results
            if recognition_result.gestures:
                st.success(f"✅ Detected {len(recognition_result.gestures)} hand(s)")
                
                # Draw hand landmarks
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                for idx, (gesture, handedness) in enumerate(zip(recognition_result.gestures, recognition_result.handedness)):
                    gesture_name = gesture[0].category_name
                    gesture_score = gesture[0].score
                    hand_side = handedness[0].category_name
                    
                    st.write(f"**Hand {idx+1}:** {hand_side} hand - Gesture: **{gesture_name}** (Confidence: {gesture_score:.2f})")
                    
                    # Draw landmarks
                    if recognition_result.hand_landmarks:
                        hand_landmarks = recognition_result.hand_landmarks[idx]
                        
                        # Convert normalized coordinates to pixel coordinates
                        h, w, c = image_bgr.shape
                        for landmark in hand_landmarks:
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(image_bgr, (x, y), 5, (0, 255, 0), -1)
                
                output_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)
                with col2:
                    st.subheader("Detection Result")
                    st.image(output_image, use_container_width=True)
            else:
                st.warning("❌ No hands detected")
                st.image(image, use_container_width=True)
    
    else:  # Webcam
        st.info("🎥 Click 'Start' to use webcam")
        run_webcam = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])
        gesture_text = st.empty()
        
        if run_webcam:
            cap = cv2.VideoCapture(0)
            
            with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as hands:
                
                while run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Cannot access webcam")
                        break
                    
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process
                    results = hands.process(image_rgb)
                    
                    # Draw hand landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS
                            )
                        gesture_text.success(f"✅ Detected {len(results.multi_hand_landmarks)} hand(s)")
                    else:
                        gesture_text.info("No hands detected")
                    
                    # Display
                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            cap.release()

# ==================== OBJECT DETECTION ====================
elif feature == "Object Detection":
    st.header("📦 Object Detection")
    st.write("Detects common objects like person, car, dog, cat, etc.")
    
    input_type = st.radio("Choose input type:", ["Upload Image", "Use Webcam"])
    
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Import object detector
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Create object detector
            base_options = python.BaseOptions(model_asset_path='models/efficientdet_lite0.tflite')
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                max_results=5,
                score_threshold=0.5
            )
            detector = vision.ObjectDetector.create_from_options(options)
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            
            # Detect objects
            detection_result = detector.detect(mp_image)
            
            # Draw results
            if detection_result.detections:
                st.success(f"✅ Detected {len(detection_result.detections)} object(s)")
                
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                for detection in detection_result.detections:
                    # Get bounding box
                    bbox = detection.bounding_box
                    start_point = (int(bbox.origin_x), int(bbox.origin_y))
                    end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
                    
                    # Draw rectangle
                    cv2.rectangle(image_bgr, start_point, end_point, (0, 255, 0), 2)
                    
                    # Get label
                    category = detection.categories[0]
                    label = f"{category.category_name}: {category.score:.2f}"
                    
                    # Draw label
                    cv2.putText(image_bgr, label, (start_point[0], start_point[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    st.write(f"**{category.category_name}** - Confidence: {category.score:.2f}")
                
                output_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)
                with col2:
                    st.subheader("Detection Result")
                    st.image(output_image, use_container_width=True)
            else:
                st.warning("❌ No objects detected")
                st.image(image, use_container_width=True)
    
    else:  # Webcam
        st.info("🎥 Webcam mode for object detection - Upload image for now")

# ==================== AUDIO CLASSIFICATION ====================
elif feature == "Audio Classification":
    st.header("🎵 Audio Classification")
    st.write("Classifies sounds like music, speech, dog bark, car horn, etc.")
    
    st.info("📁 Upload an audio file (.wav format)")
    
    uploaded_audio = st.file_uploader("Choose an audio file...", type=["wav"])
    
    if uploaded_audio is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_audio.read())
            tmp_file_path = tmp_file.name
        
        st.audio(uploaded_audio, format='audio/wav')
        
        try:
            # Import audio classifier
            from mediapipe.tasks import python
            from mediapipe.tasks.python import audio
            
            # Create audio classifier
            base_options = python.BaseOptions(model_asset_path='models/yamnet.tflite')
            options = audio.AudioClassifierOptions(
                base_options=base_options,
                max_results=5
            )
            classifier = audio.AudioClassifier.create_from_options(options)
            
            # Load audio file
            from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
            
            # Read audio file
            import wave
            with wave.open(tmp_file_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                num_channels = wav_file.getnchannels()
                audio_frames = wav_file.readframes(wav_file.getnframes())
                audio_array = np.frombuffer(audio_frames, dtype=np.int16)
            
            # Convert to float32
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Create AudioData object
            audio_data_obj = audio_data_module.AudioData.create_from_array(
                audio_float,
                sample_rate
            )
            
            # Classify
            classification_result = classifier.classify(audio_data_obj)
            
            # Display results
            if classification_result.classifications:
                st.success("✅ Audio Classification Results:")
                
                for idx, classification in enumerate(classification_result.classifications[0].categories[:5]):
                    st.write(f"**{idx+1}. {classification.category_name}** - Confidence: {classification.score:.4f}")
            else:
                st.warning("❌ No classification results")
        
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            st.info("Make sure the audio file is in WAV format with 16kHz sample rate")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with MediaPipe & Streamlit")
