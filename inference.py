"""
Inference Module
----------------
Real-time monitoring and video processing for abnormal behavior detection.
Includes rolling buffer management, cooldown logic, and event post-processing.

Key Features:
- Real-time monitoring with rolling buffer
- Configurable inference stride and cooldown
- Video file processing with event detection
- Event merging and timestamp extraction
"""

import time
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta

from keypoint_extractor import KeypointExtractor
from models import predict_sequence


# Configuration constants
COOLDOWN_SECONDS = 1.5  # Time between alerts to prevent spam
MERGE_THRESHOLD = 2.0   # Merge detections within N seconds
INFERENCE_STRIDE = 5    # Run inference every N frames


class RealtimeMonitor:
    """
    Real-time monitoring system for live camera feed.
    
    Maintains a rolling buffer of keypoint features and runs inference
    at regular intervals to detect abnormal behavior patterns.
    """
    
    def __init__(self,
                 model,
                 window_length: int = 30,
                 threshold: float = 0.8,
                 inference_stride: int = INFERENCE_STRIDE,
                 cooldown_seconds: float = COOLDOWN_SECONDS):
        """
        Initialize real-time monitor.
        
        Args:
            model: Trained Keras model for sequence classification
            window_length: Number of frames to analyze (e.g., 30 for 2s @ 15fps)
            threshold: Detection threshold (0.0-1.0)
            inference_stride: Run inference every N frames
            cooldown_seconds: Minimum time between alerts
        """
        self.model = model
        self.window_length = window_length
        self.threshold = threshold
        self.inference_stride = inference_stride
        self.cooldown_seconds = cooldown_seconds
        
        # Rolling buffer for keypoints
        self.buffer = deque(maxlen=window_length)
        
        # Frame counter for inference stride
        self.frame_count = 0
        
        # Cooldown tracking
        self.last_alert_time = None
        self.in_cooldown = False
        
        # Statistics
        self.total_frames_processed = 0
        self.total_alerts = 0
    
    def add_frame(self, keypoints: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Add a frame's keypoints to the rolling buffer and run inference if needed.
        
        Args:
            keypoints: Keypoint features for current frame (num_features,)
        
        Returns:
            (predicted_class, probability) if inference was run, else None
        """
        # Add to buffer
        self.buffer.append(keypoints)
        self.frame_count += 1
        self.total_frames_processed += 1
        
        # Check if we have enough frames and should run inference
        if len(self.buffer) < self.window_length:
            return None
        
        if self.frame_count % self.inference_stride != 0:
            return None
        
        # Create sequence from buffer
        sequence = np.array(list(self.buffer))  # (window_length, num_features)
        
        # Run inference
        predicted_class, probability = predict_sequence(self.model, sequence, self.threshold)
        
        return predicted_class, probability
    
    def check_alert(self, predicted_class: int, probability: float) -> bool:
        """
        Check if an alert should be triggered based on prediction and cooldown.
        
        Args:
            predicted_class: Predicted class (0=normal, 1=abnormal)
            probability: Prediction probability
        
        Returns:
            True if alert should be triggered, False otherwise
        """
        # Check if abnormal behavior detected
        if predicted_class != 1:
            return False
        
        # Check cooldown
        current_time = time.time()
        
        if self.last_alert_time is not None:
            time_since_last_alert = current_time - self.last_alert_time
            if time_since_last_alert < self.cooldown_seconds:
                return False
        
        # Trigger alert
        self.last_alert_time = current_time
        self.total_alerts += 1
        
        return True
    
    def reset(self):
        """Reset the monitor state."""
        self.buffer.clear()
        self.frame_count = 0
        self.last_alert_time = None
        self.in_cooldown = False
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics."""
        return {
            'total_frames': self.total_frames_processed,
            'total_alerts': self.total_alerts,
            'buffer_size': len(self.buffer),
            'alert_rate': self.total_alerts / max(1, self.total_frames_processed)
        }


class VideoProcessor:
    """
    Process video files for abnormal behavior detection.
    
    Analyzes entire video and returns a timeline of detected events.
    """
    
    def __init__(self,
                 model,
                 window_length: int = 30,
                 threshold: float = 0.8,
                 stride: int = 15):
        """
        Initialize video processor.
        
        Args:
            model: Trained Keras model
            window_length: Number of frames per window
            threshold: Detection threshold
            stride: Stride for sliding window
        """
        self.model = model
        self.window_length = window_length
        self.threshold = threshold
        self.stride = stride
    
    def process_video(self,
                     video_path: str,
                     progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Process entire video and detect abnormal events.
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback function(current_frame, total_frames)
        
        Returns:
            List of detected events with timestamps and confidence
            Each event: {'timestamp': str, 'frame_idx': int, 'confidence': float, 'frame': np.ndarray}
        """
        # Extract keypoints from video
        extractor = KeypointExtractor()
        
        # Open video to get metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Extract keypoints
        print(f"Extracting keypoints from video ({total_frames} frames)...")
        keypoints, confidences = extractor.process_video(video_path)
        
        # Create sliding windows
        windows = []
        window_frame_indices = []
        
        for start_idx in range(0, len(keypoints) - self.window_length + 1, self.stride):
            window = keypoints[start_idx:start_idx + self.window_length]
            windows.append(window)
            # Store the middle frame index for this window
            mid_frame_idx = start_idx + self.window_length // 2
            window_frame_indices.append(mid_frame_idx)
        
        if len(windows) == 0:
            print("Warning: Video too short for analysis")
            return []
        
        windows = np.array(windows)  # (num_windows, window_length, num_features)
        
        # Run inference on all windows
        print(f"Running inference on {len(windows)} windows...")
        probabilities = self.model.predict(windows, verbose=0)[:, 0]
        
        # Find detections above threshold
        detections = []
        for i, (prob, frame_idx) in enumerate(zip(probabilities, window_frame_indices)):
            if prob >= self.threshold:
                timestamp = self._frame_to_timestamp(frame_idx, fps)
                detections.append({
                    'window_idx': i,
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'confidence': float(prob)
                })
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(windows))
        
        # Merge close detections
        merged_events = self._merge_detections(detections, fps)
        
        # Extract frame thumbnails for events
        events_with_frames = self._extract_event_frames(video_path, merged_events)
        
        print(f"Detected {len(events_with_frames)} abnormal events")
        
        return events_with_frames
    
    def _frame_to_timestamp(self, frame_idx: int, fps: float) -> str:
        """Convert frame index to timestamp string (HH:MM:SS)."""
        seconds = frame_idx / fps
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _merge_detections(self, detections: List[Dict], fps: float) -> List[Dict]:
        """
        Merge detections that are close in time.
        
        Args:
            detections: List of detection dictionaries
            fps: Video frame rate
        
        Returns:
            Merged list of events
        """
        if len(detections) == 0:
            return []
        
        # Sort by frame index
        detections = sorted(detections, key=lambda x: x['frame_idx'])
        
        merged = []
        current_event = detections[0].copy()
        current_event['end_frame_idx'] = current_event['frame_idx']
        current_event['max_confidence'] = current_event['confidence']
        
        for detection in detections[1:]:
            frame_diff = detection['frame_idx'] - current_event['end_frame_idx']
            time_diff = frame_diff / fps
            
            if time_diff <= MERGE_THRESHOLD:
                # Merge into current event
                current_event['end_frame_idx'] = detection['frame_idx']
                current_event['max_confidence'] = max(
                    current_event['max_confidence'],
                    detection['confidence']
                )
            else:
                # Start new event
                merged.append(current_event)
                current_event = detection.copy()
                current_event['end_frame_idx'] = current_event['frame_idx']
                current_event['max_confidence'] = current_event['confidence']
        
        # Add last event
        merged.append(current_event)
        
        # Update confidence to max confidence
        for event in merged:
            event['confidence'] = event['max_confidence']
        
        return merged
    
    def _extract_event_frames(self, video_path: str, events: List[Dict]) -> List[Dict]:
        """
        Extract frame thumbnails for detected events.
        
        Args:
            video_path: Path to video file
            events: List of event dictionaries
        
        Returns:
            Events with added 'frame' key containing thumbnail image
        """
        if len(events) == 0:
            return []
        
        cap = cv2.VideoCapture(video_path)
        
        events_with_frames = []
        for event in events:
            frame_idx = event['frame_idx']
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize for thumbnail
                thumbnail = cv2.resize(frame, (320, 240))
                event['frame'] = thumbnail
                events_with_frames.append(event)
        
        cap.release()
        
        return events_with_frames


def process_video_file(video_path: str,
                       model,
                       settings: Dict) -> List[Dict]:
    """
    Convenience function to process a video file with given settings.
    
    Args:
        video_path: Path to video file
        model: Trained Keras model
        settings: Dictionary with 'window_length', 'threshold', 'stride'
    
    Returns:
        List of detected events
    
    Example:
        >>> settings = {'window_length': 30, 'threshold': 0.8, 'stride': 15}
        >>> events = process_video_file('video.mp4', model, settings)
        >>> for event in events:
        ...     print(f"Event at {event['timestamp']} with confidence {event['confidence']:.2f}")
    """
    processor = VideoProcessor(
        model=model,
        window_length=settings.get('window_length', 30),
        threshold=settings.get('threshold', 0.8),
        stride=settings.get('stride', 15)
    )
    
    events = processor.process_video(video_path)
    
    return events


if __name__ == "__main__":
    """
    Test inference module.
    """
    import os
    from models import create_mock_model
    
    print("Testing inference module...\n")
    
    # Create mock model
    print("1. Creating mock model:")
    model = create_mock_model(window_length=30, num_features=66)
    print()
    
    # Test RealtimeMonitor
    print("2. Testing RealtimeMonitor:")
    monitor = RealtimeMonitor(
        model=model,
        window_length=30,
        threshold=0.8,
        inference_stride=5
    )
    
    # Simulate adding frames
    for i in range(50):
        dummy_keypoints = np.random.rand(66)
        result = monitor.add_frame(dummy_keypoints)
        
        if result is not None:
            pred_class, prob = result
            alert = monitor.check_alert(pred_class, prob)
            print(f"   Frame {i}: Class={pred_class}, Prob={prob:.4f}, Alert={alert}")
    
    stats = monitor.get_stats()
    print(f"   Stats: {stats}")
    print()
    
    # Test VideoProcessor (with dummy video if available)
    print("3. Testing VideoProcessor:")
    print("   (Skipping - requires actual video file)")
    print()
    
    print("All tests passed!")
