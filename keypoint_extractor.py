"""
Keypoint Extractor Module
--------------------------
Uses MediaPipe Pose to extract body landmarks from video frames or webcam feed.
Provides normalized keypoint vectors with confidence scores and skeleton overlay visualization.

Key Features:
- 33 pose landmarks extraction (MediaPipe Pose)
- Normalization by torso length for scale invariance
- Robust handling of missed detections (fill-forward or zero-padding)
- Skeleton overlay drawing for visualization
- Support for both video files and live webcam streams
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

# Try to import MediaPipe, fall back to stub if unavailable
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    import mediapipe_stub as mp
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available - using stub (limited functionality)")


class KeypointExtractor:
    """
    Extracts pose keypoints from video frames using MediaPipe Pose.
    
    Attributes:
        mp_pose: MediaPipe Pose solution
        mp_drawing: MediaPipe drawing utilities
        pose: Pose detector instance
        last_valid_keypoints: Last successfully detected keypoints (for fill-forward)
    """
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Pose detector.
        
        Args:
            static_image_mode: If True, treats each frame independently
            model_complexity: 0, 1, or 2. Higher = more accurate but slower
            min_detection_confidence: Minimum confidence for person detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Store last valid keypoints for fill-forward on missed detections
        self.last_valid_keypoints = None
        self.last_valid_confidence = 0.0
    
    def extract_keypoints(self, frame: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Extract pose keypoints from a single frame.
        
        Args:
            frame: RGB or BGR image frame (H, W, 3)
        
        Returns:
            keypoints: Normalized keypoint array (66,) - 33 landmarks × 2 coords (x, y)
            confidence: Average visibility/confidence score (0.0-1.0)
            detected: Whether pose was successfully detected
        """
        # Convert BGR to RGB if needed (OpenCV uses BGR by default)
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Process frame with MediaPipe
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Convert to numpy array: (33 landmarks, 3 values: x, y, visibility)
            keypoints_raw = np.array([
                [lm.x, lm.y, lm.visibility] 
                for lm in landmarks
            ])  # Shape: (33, 3)
            
            # Normalize by torso length for scale invariance
            keypoints_normalized = self._normalize_keypoints(keypoints_raw)
            
            # Flatten to (66,) - only x, y coordinates
            keypoints_flat = keypoints_normalized[:, :2].flatten()
            
            # Calculate average confidence (visibility score)
            confidence = np.mean(keypoints_raw[:, 2])
            
            # Store for fill-forward
            self.last_valid_keypoints = keypoints_flat
            self.last_valid_confidence = confidence
            
            return keypoints_flat, confidence, True
        
        else:
            # No pose detected - use fill-forward or zeros
            if self.last_valid_keypoints is not None:
                # Fill-forward: use last valid keypoints
                return self.last_valid_keypoints, self.last_valid_confidence * 0.5, False
            else:
                # No previous keypoints - return zeros
                return np.zeros(66), 0.0, False
    
    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints by torso length and center.
        
        Args:
            keypoints: Raw keypoints (33, 3) with x, y, visibility
        
        Returns:
            Normalized keypoints (33, 3)
        """
        # MediaPipe landmark indices:
        # 11: Left shoulder, 12: Right shoulder
        # 23: Left hip, 24: Right hip
        
        left_shoulder = keypoints[11, :2]
        right_shoulder = keypoints[12, :2]
        left_hip = keypoints[23, :2]
        right_hip = keypoints[24, :2]
        
        # Calculate torso center and length
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        
        # Avoid division by zero
        if torso_length < 1e-6:
            torso_length = 1.0
        
        # Normalize: center at origin and scale by torso length
        normalized = keypoints.copy()
        normalized[:, 0] = (keypoints[:, 0] - shoulder_center[0]) / torso_length
        normalized[:, 1] = (keypoints[:, 1] - shoulder_center[1]) / torso_length
        
        return normalized
    
    def draw_skeleton(self, frame: np.ndarray, draw_landmarks: bool = True) -> np.ndarray:
        """
        Draw pose skeleton overlay on frame.
        
        Args:
            frame: Input frame (BGR format)
            draw_landmarks: Whether to draw landmarks and connections
        
        Returns:
            Frame with skeleton overlay
        """
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Process frame
        results = self.pose.process(frame_rgb)
        
        # Draw skeleton
        annotated_frame = frame.copy()
        if results.pose_landmarks and draw_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return annotated_frame
    
    def process_video(self, 
                     video_path: str, 
                     max_frames: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract keypoints from entire video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None = all)
        
        Returns:
            keypoints: Array of keypoints (num_frames, 66)
            confidences: Array of confidence scores (num_frames,)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        keypoints_list = []
        confidences_list = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keypoints
            keypoints, confidence, detected = self.extract_keypoints(frame)
            keypoints_list.append(keypoints)
            confidences_list.append(confidence)
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        
        # Convert to numpy arrays
        keypoints_array = np.array(keypoints_list)  # (num_frames, 66)
        confidences_array = np.array(confidences_list)  # (num_frames,)
        
        return keypoints_array, confidences_array
    
    def process_webcam_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Process a single webcam frame and return both keypoints and annotated frame.
        
        Args:
            frame: Input frame from webcam (BGR format)
        
        Returns:
            annotated_frame: Frame with skeleton overlay
            keypoints: Normalized keypoint array (66,)
            confidence: Detection confidence (0.0-1.0)
            detected: Whether pose was detected
        """
        # Extract keypoints
        keypoints, confidence, detected = self.extract_keypoints(frame)
        
        # Draw skeleton
        annotated_frame = self.draw_skeleton(frame, draw_landmarks=True)
        
        return annotated_frame, keypoints, confidence, detected
    
    def reset(self):
        """Reset the extractor state (clear fill-forward buffer)."""
        self.last_valid_keypoints = None
        self.last_valid_confidence = 0.0
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


def extract_keypoints_from_video(video_path: str, 
                                 max_frames: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to extract keypoints from a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process (None = all)
    
    Returns:
        keypoints: Array of keypoints (num_frames, 66)
        confidences: Array of confidence scores (num_frames,)
    
    Example:
        >>> keypoints, confidences = extract_keypoints_from_video("video.mp4")
        >>> print(f"Extracted {len(keypoints)} frames")
    """
    extractor = KeypointExtractor()
    keypoints, confidences = extractor.process_video(video_path, max_frames)
    return keypoints, confidences


def get_keypoint_names() -> List[str]:
    """
    Get list of MediaPipe Pose landmark names.
    
    Returns:
        List of 33 landmark names
    """
    return [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]


if __name__ == "__main__":
    """
    Test the keypoint extractor with webcam.
    Press 'q' to quit.
    """
    print("Testing KeypointExtractor with webcam...")
    print("Press 'q' to quit")
    
    extractor = KeypointExtractor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, keypoints, confidence, detected = extractor.process_webcam_frame(frame)
        
        # Display info
        status = "Detected" if detected else "Not Detected"
        color = (0, 255, 0) if detected else (0, 0, 255)
        
        cv2.putText(annotated_frame, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show frame
        cv2.imshow('Keypoint Extractor Test', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed!")
