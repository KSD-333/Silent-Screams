"""
Silent Screams - Streamlit GUI Application
-------------------------------------------
Main application interface for live monitoring and video analysis.

Features:
- Live camera monitoring with real-time alerts
- Video upload and analysis
- Configurable settings (threshold, window length, sound, etc.)
- Alert logging and visualization
- Privacy-first local processing
"""

import os
import time
import cv2
import numpy as np
import streamlit as st
from datetime import datetime
from PIL import Image
import tempfile

# Import custom modules
from keypoint_extractor import KeypointExtractor
from models import load_model, create_mock_model
from inference import RealtimeMonitor, VideoProcessor
from sound_utils import play_alert_sound, generate_beep_sound


# Page configuration
st.set_page_config(
    page_title="Silent Screams - Behavior Monitoring",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'alert_log' not in st.session_state:
        st.session_state.alert_log = []
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    if 'total_frames' not in st.session_state:
        st.session_state.total_frames = 0
    if 'total_alerts' not in st.session_state:
        st.session_state.total_alerts = 0


def load_model_cached(model_path: str):
    """Load model with caching."""
    try:
        if not os.path.exists(model_path):
            st.warning(f"Model file not found: {model_path}")
            st.info("Creating mock model for demo purposes...")
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Create and save mock model
            model = create_mock_model(save_path=model_path)
            st.success("Mock model created successfully!")
            return model
        
        model = load_model(model_path)
        st.success(f"Model loaded: {model_path}")
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def render_sidebar():
    """Render sidebar with settings."""
    st.sidebar.title("Settings")
    
    # Model settings (using trained model)
    model_path = "models/distress_detector_v2.h5"
    
    # Auto-load model if not loaded
    if not st.session_state.model_loaded:
        model = load_model_cached(model_path)
        if model is not None:
            st.session_state.model = model
            st.session_state.model_loaded = True
    
    # Fixed values for balanced detection (hidden from UI)
    threshold = 0.76  # Optimal threshold for balanced performance
    window_length = 30  # Must match model training (30 frames)
    frame_rate = 30  # Maximum FPS for fastest real-time response
    
    # Audio settings
    st.sidebar.subheader("Audio Settings")
    sound_file = st.sidebar.text_input(
        "Alert Sound File",
        value="alarm.wav",
        help="Path to custom alert sound (WAV format)"
    )
    
    mute_audio = st.sidebar.checkbox(
        "Mute Audio Alerts",
        value=False,
        help="Disable alert sounds"
    )
    
    st.sidebar.markdown("---")
    
    # System Status
    st.sidebar.subheader("System Status")
    status_text = "Ready" if st.session_state.model_loaded else "No Model"
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Status", status_text)
    with col2:
        st.metric("Alerts", st.session_state.total_alerts)
    
    st.sidebar.metric("Frames Processed", st.session_state.total_frames)
    
    return {
        'model_path': model_path,
        'threshold': threshold,
        'window_length': int(window_length),
        'frame_rate': int(frame_rate),
        'sound_file': sound_file,
        'mute_audio': mute_audio
    }


def render_alert_banner():
    """Render alert banner."""
    st.markdown(
        """
        <div style="
            background-color: #ff4444;
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin: 20px 0;
            border: 2px solid #cc0000;
        ">
            ABNORMAL BEHAVIOR DETECTED
        </div>
        """,
        unsafe_allow_html=True
    )


def add_alert_to_log(confidence: float):
    """Add alert to log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.alert_log.append({
        'timestamp': timestamp,
        'confidence': confidence
    })
    st.session_state.total_alerts += 1
    
    # Keep only last 50 alerts
    if len(st.session_state.alert_log) > 50:
        st.session_state.alert_log = st.session_state.alert_log[-50:]


def render_alert_log():
    """Render alert log."""
    if len(st.session_state.alert_log) > 0:
        st.subheader("Recent Alerts")
        
        # Display in reverse chronological order
        for alert in reversed(st.session_state.alert_log[-10:]):
            st.text(f"Alert: {alert['timestamp']} - Confidence: {alert['confidence']:.2%}")
    else:
        st.info("No alerts yet")


def live_monitoring_mode(settings: dict):
    """Live camera monitoring mode."""
    st.header("Live Camera Monitoring")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first using the sidebar settings.")
        return
    
    # Control buttons
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("Start Monitoring", disabled=st.session_state.monitoring_active):
            st.session_state.monitoring_active = True
            st.rerun()
        
        if st.button("Stop Monitoring", disabled=not st.session_state.monitoring_active):
            st.session_state.monitoring_active = False
            st.rerun()
    
    if not st.session_state.monitoring_active:
        st.info("Click 'Start Monitoring' to begin live camera feed analysis.")
        return
    
    # Create placeholders
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    alert_placeholder = st.empty()
    log_placeholder = st.empty()
    
    # Initialize components
    extractor = KeypointExtractor()
    monitor = RealtimeMonitor(
        model=st.session_state.model,
        window_length=settings['window_length'],
        threshold=settings['threshold'],
        inference_stride=3,  # Faster inference for instant detection
        cooldown_seconds=0.5  # Shorter cooldown for hospital use
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check camera permissions.")
        st.session_state.monitoring_active = False
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, settings['frame_rate'])
    
    frame_count = 0
    last_inference_time = time.time()
    
    try:
        while st.session_state.monitoring_active:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Failed to read from webcam")
                break
            
            frame_count += 1
            st.session_state.total_frames += 1
            
            # Process frame
            annotated_frame, keypoints, confidence, detected = extractor.process_webcam_frame(frame)
            
            # Add to monitor buffer
            result = monitor.add_frame(keypoints)
            
            # Display frame
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update status
            if result is not None:
                pred_class, prob = result
                
                status_text = f"**Frame:** {frame_count} | **Detection:** {'Yes' if detected else 'No'} | **Confidence:** {confidence:.2f}"
                status_text += f"\n\n**Prediction:** {'Abnormal' if pred_class == 1 else 'Normal'} | **Probability:** {prob:.2%}"
                
                status_placeholder.markdown(status_text)
                
                # Check for alert
                if monitor.check_alert(pred_class, prob):
                    alert_placeholder.empty()
                    with alert_placeholder.container():
                        render_alert_banner()
                    
                    # Play sound
                    if not settings['mute_audio']:
                        play_alert_sound(settings['sound_file'])
                    
                    # Add to log
                    add_alert_to_log(prob)
                    
                    # Update log display
                    with log_placeholder.container():
                        render_alert_log()
            
            # Control frame rate
            elapsed = time.time() - last_inference_time
            target_delay = 1.0 / settings['frame_rate']
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)
            last_inference_time = time.time()
            
            # Check if monitoring should stop
            if not st.session_state.monitoring_active:
                break
    
    finally:
        cap.release()
        extractor.reset()
        st.info("Monitoring stopped.")


def video_upload_mode(settings: dict):
    """Video upload and analysis mode."""
    st.header("Video Upload & Analysis")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first using the sidebar settings.")
        return
    
    # Initialize session state for video analysis
    if 'video_analyzed' not in st.session_state:
        st.session_state.video_analyzed = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # File uploader with key to maintain state
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for analysis",
        key="video_uploader"
    )
    
    if uploaded_file is None:
        st.info("Please upload a video file to analyze.")
        st.session_state.video_analyzed = False
        st.session_state.analysis_results = None
        return
    
    st.success(f"Video uploaded: {uploaded_file.name}")
    
    # Analyze button
    analyze_clicked = st.button("Analyze Video")
    
    if analyze_clicked or st.session_state.video_analyzed:
        if analyze_clicked and not st.session_state.video_analyzed:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            with st.spinner("Analyzing video... This may take a few minutes."):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(current, total):
                    progress = int((current / total) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {current}/{total} windows ({progress}%)")
                
                # Process video with faster detection
                processor = VideoProcessor(
                    model=st.session_state.model,
                    window_length=settings['window_length'],
                    threshold=settings['threshold'],
                    stride=3  # Even smaller stride for fastest detection
                )
                
                try:
                    events = processor.process_video(video_path, progress_callback)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store results in session state
                    st.session_state.analysis_results = events
                    st.session_state.video_analyzed = True
                    
                    # Play alert sound for 5 seconds if abnormal events detected
                    if len(events) > 0 and not settings['mute_audio']:
                        # Play sound multiple times for 5 seconds
                        import winsound
                        import threading
                        
                        def play_alert_loop():
                            for _ in range(10):  # Play 10 times (0.5s each = 5s total)
                                try:
                                    winsound.Beep(1000, 500)  # 1000Hz for 500ms
                                except:
                                    pass
                        
                        sound_thread = threading.Thread(target=play_alert_loop)
                        sound_thread.daemon = True
                        sound_thread.start()
                
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    return
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(video_path):
                        os.remove(video_path)
        
        # Display results from session state
        events = st.session_state.analysis_results
        
        if events is None:
            return
        
        # Display results with color coding
        if len(events) == 0:
            st.markdown("""
                <div style="background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745;">
                    <h3 style="color: #155724; margin: 0;">✓ No Abnormal Events Detected</h3>
                    <p style="color: #155724; margin: 5px 0 0 0;">Video analysis complete. All behavior appears normal.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 5px solid #dc3545;">
                    <h3 style="color: #721c24; margin: 0;">⚠ Abnormal Events Detected: {len(events)}</h3>
                    <p style="color: #721c24; margin: 5px 0 0 0;">Alert! Distress behavior detected in the video.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Detection Timeline")
            
            # Display events with red highlighting
            for i, event in enumerate(events):
                with st.expander(f"🔴 Event #{i+1} - {event['timestamp']} (Confidence: {event['confidence']:.2%})", expanded=False):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if 'frame' in event:
                            frame_rgb = cv2.cvtColor(event['frame'], cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption=f"Frame at {event['timestamp']}")
                    
                    with col2:
                        st.markdown(f"""
                            <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px;">
                                <p style="color: #000; margin: 5px 0;"><strong>Timestamp:</strong> {event['timestamp']}</p>
                                <p style="color: #000; margin: 5px 0;"><strong>Frame Index:</strong> {event['frame_idx']}</p>
                                <p style="color: #dc3545; margin: 5px 0; font-size: 18px;"><strong>Status: ABNORMAL</strong></p>
                                <p style="color: #000; margin: 5px 0;"><strong>Confidence:</strong> {event['confidence']:.2%}</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            # CSV export removed for faster workflow


def main():
    """Main application."""
    # Initialize session state
    init_session_state()
    
    # Custom CSS for light theme
    st.markdown("""
        <style>
        /* Force light theme everywhere */
        .stApp {
            background-color: #ffffff !important;
        }
        
        .main {
            background-color: #ffffff !important;
        }
        
        .block-container {
            background-color: #ffffff !important;
            padding-top: 2rem !important;
        }
        
        /* Remove black header */
        header {
            background-color: #ffffff !important;
        }
        
        [data-testid="stHeader"] {
            background-color: #ffffff !important;
        }
        
        /* Sidebar light theme */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background-color: #f8f9fa !important;
        }
        
        /* Simple card styling with visible text */
        .feature-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            text-align: center;
            color: #000000;
        }
        
        .feature-card h4 {
            color: #000000;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .feature-card p {
            color: #333333;
        }
        
        /* Metric styling */
        div[data-testid="metric-container"] {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #e0e0e0;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #ffffff !important;
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #e0e0e0;
            border-radius: 4px 4px 0 0;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #f8f9fa !important;
            border-bottom: 2px solid #ff4444 !important;
        }
        
        /* Ensure all text is visible */
        body, p, h1, h2, h3, h4, h5, h6, span, div, label, small {
            color: #000000 !important;
        }
        
        /* Fix text input visibility */
        input {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* Fix file uploader visibility */
        [data-testid="stFileUploader"] {
            background-color: #ffffff !important;
        }
        
        [data-testid="stFileUploader"] section {
            background-color: #f8f9fa !important;
            border: 2px dashed #cccccc !important;
        }
        
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] div {
            color: #000000 !important;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #e0e0e0;
        }
        
        .stButton > button:hover {
            background-color: #f8f9fa;
            border-color: #ff4444;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header at top with light pink box
    st.markdown("""
        <div style="background-color: #ffe6f0; padding: 30px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="font-family: 'Georgia', serif; font-size: 2.5rem; font-weight: 700; color: #000000; text-align: center; margin: 0;">
                Silent Screams – Distress Detection in Silent Surveillance Footage
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    settings = render_sidebar()
    
    # Main content - Remove the separator line
    
    # Custom styled tabs as menu bar
    st.markdown("""
        <style>
        /* Hide default tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 40px;
            background-color: #f5f5f5;
            padding: 15px 30px;
            border-radius: 50px;
            margin-bottom: 30px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: transparent;
            border: none;
            color: #000000;
            font-size: 16px;
            font-weight: 600;
            padding: 10px 25px;
            border-radius: 25px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #ffd699 !important;
            color: #000000;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #ffe6b3;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Mode selection with tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Live Monitoring", "Video Analysis"])
    
    with tab1:
        # Welcome section
        st.markdown("### Welcome to Silent Screams")
        st.markdown("AI-powered distress detection system optimized for hospital environments with instant alert capabilities.")
        
        st.markdown("")
        
        # Feature cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>Live Monitoring</h4>
                <p>Real-time patient monitoring with instant distress detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>Video Analysis</h4>
                <p>Review recorded footage for distress events with detailed timeline</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>High Sensitivity Mode</h4>
                <p>Optimized for instant detection with minimal delay</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("")
        
        # Quick Start Guide
        with st.expander("Quick Start Guide", expanded=False):
            st.markdown("""
            **Getting Started:**
            
            1. **Load Model** (Left Sidebar)
               - Load your trained distress detection model
               - System is pre-configured for high sensitivity
               
            2. **Select Mode** (Tabs Above)
               - Live Monitoring: Real-time patient monitoring
               - Video Analysis: Review recorded footage
               
            3. **Start Monitoring**
               - System will provide instant alerts for distress detection
               - Audio and visual alerts activated automatically
            """)
        
        # System Requirements
        with st.expander("System Requirements", expanded=False):
            st.markdown("""
            **Minimum Requirements:**
            - Python 3.8+
            - Webcam (for live monitoring)
            - 4GB RAM
            - CPU: Intel i5 or equivalent
            
            **Recommended:**
            - Python 3.10+
            - HD Webcam (720p or higher)
            - 8GB RAM
            - GPU: NVIDIA GTX 1060 or better
            """)
        
        # Privacy & Ethics
        with st.expander("Privacy & Ethics", expanded=False):
            st.markdown("""
            **Privacy Guarantees:**
            - All processing happens locally on your device
            - No data is stored or transmitted to external servers
            - No network calls - works completely offline
            - Video frames processed in memory only
            - HIPAA-compliant local processing
            
            **Hospital Use Guidelines:**
            - Designed for patient safety monitoring
            - Instant distress detection and alerting
            - Must comply with hospital privacy policies
            - Obtain proper consent as per medical regulations
            - Use only in designated monitoring areas
            - Staff must respond immediately to alerts
            
            **Always follow hospital protocols and obtain proper authorization.**
            """)
    
    with tab2:
        live_monitoring_mode(settings)
    
    with tab3:
        video_upload_mode(settings)


if __name__ == "__main__":
    main()
