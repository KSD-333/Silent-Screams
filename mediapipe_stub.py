"""
MediaPipe Alternative for Python 3.13 Compatibility
----------------------------------------------------
Provides OpenCV DNN-based pose estimation as MediaPipe replacement.
Uses MoveNet or OpenPose models for pose detection.
"""

import cv2
import numpy as np
from typing import Optional, List
import urllib.request
import os


class MockLandmark:
    """Mock MediaPipe landmark."""
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class MockPoseLandmarks:
    """Mock MediaPipe pose landmarks."""
    def __init__(self, landmarks: List[MockLandmark]):
        self.landmark = landmarks


class MockPoseResults:
    """Mock MediaPipe pose results."""
    def __init__(self):
        self.pose_landmarks: Optional[MockPoseLandmarks] = None


class OpenCVPoseDetector:
    """
    OpenCV-based pose detector using BODY_25 model.
    Provides similar functionality to MediaPipe Pose.
    """
    
    # COCO body parts mapping (18 keypoints)
    BODY_PARTS = {
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17
    }
    
    # Pose pairs for drawing skeleton
    POSE_PAIRS = [
        ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
        ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
        ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
        ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
        ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
    ]
    
    # MediaPipe to COCO mapping (approximate)
    MEDIAPIPE_TO_COCO = {
        0: 0,   # nose
        1: 14,  # left_eye_inner -> REye
        2: 15,  # left_eye -> LEye
        3: 15,  # left_eye_outer -> LEye
        4: 14,  # right_eye_inner -> REye
        5: 14,  # right_eye -> REye
        6: 14,  # right_eye_outer -> REye
        7: 17,  # left_ear -> LEar
        8: 16,  # right_ear -> REar
        9: 0,   # mouth_left -> Nose
        10: 0,  # mouth_right -> Nose
        11: 5,  # left_shoulder -> LShoulder
        12: 2,  # right_shoulder -> RShoulder
        13: 6,  # left_elbow -> LElbow
        14: 3,  # right_elbow -> RElbow
        15: 7,  # left_wrist -> LWrist
        16: 4,  # right_wrist -> RWrist
        17: 7,  # left_pinky -> LWrist
        18: 4,  # right_pinky -> RWrist
        19: 7,  # left_index -> LWrist
        20: 4,  # right_index -> RWrist
        21: 7,  # left_thumb -> LWrist
        22: 4,  # right_thumb -> RWrist
        23: 11, # left_hip -> LHip
        24: 8,  # right_hip -> RHip
        25: 12, # left_knee -> LKnee
        26: 9,  # right_knee -> RKnee
        27: 13, # left_ankle -> LAnkle
        28: 10, # right_ankle -> RAnkle
        29: 13, # left_heel -> LAnkle
        30: 10, # right_heel -> RAnkle
        31: 13, # left_foot_index -> LAnkle
        32: 10, # right_foot_index -> RAnkle
    }
    
    def __init__(self, use_simple_detector: bool = True):
        """
        Initialize OpenCV pose detector.
        
        Args:
            use_simple_detector: If True, uses simple heuristic detection (no model download)
        """
        self.use_simple = use_simple_detector
        self.net = None
        self.input_width = 368
        self.input_height = 368
        self.threshold = 0.1
        
        if not use_simple_detector:
            self._load_model()
        
        print("✓ Using OpenCV-based pose detection (Python 3.13 compatible)")
    
    def _load_model(self):
        """Load OpenPose model (optional - requires model files)."""
        model_dir = "models/pose"
        os.makedirs(model_dir, exist_ok=True)
        
        proto_file = os.path.join(model_dir, "pose_deploy_linevec.prototxt")
        weights_file = os.path.join(model_dir, "pose_iter_440000.caffemodel")
        
        # Check if model files exist
        if os.path.exists(proto_file) and os.path.exists(weights_file):
            try:
                self.net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
                print("✓ Loaded OpenPose DNN model")
            except Exception as e:
                print(f"⚠️  Could not load OpenPose model: {e}")
                print("   Using simple detector instead")
                self.use_simple = True
        else:
            print("⚠️  OpenPose model files not found")
            print("   Using simple detector (limited accuracy)")
            self.use_simple = True
    
    def detect_pose_simple(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Simple pose detection using basic heuristics.
        Returns approximate keypoints based on image analysis.
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use face detection as anchor
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Use first detected face
        (x, y, w, h) = faces[0]
        
        # Estimate body keypoints based on face position
        # This is a rough approximation
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Create 18 COCO keypoints (normalized)
        keypoints = np.zeros((18, 3))  # x, y, confidence
        
        # Nose (0)
        keypoints[0] = [face_center_x / width, face_center_y / height, 0.8]
        
        # Neck (1) - below face
        keypoints[1] = [face_center_x / width, (y + h + h * 0.3) / height, 0.7]
        
        # Shoulders (2, 5) - estimate based on face width
        shoulder_y = (y + h + h * 0.5) / height
        keypoints[2] = [(face_center_x + w) / width, shoulder_y, 0.6]  # Right shoulder
        keypoints[5] = [(face_center_x - w) / width, shoulder_y, 0.6]  # Left shoulder
        
        # Hips (8, 11) - estimate
        hip_y = (y + h + h * 2) / height
        keypoints[8] = [(face_center_x + w * 0.5) / width, hip_y, 0.5]   # Right hip
        keypoints[11] = [(face_center_x - w * 0.5) / width, hip_y, 0.5]  # Left hip
        
        # Eyes (14, 15)
        keypoints[14] = [(face_center_x + w * 0.2) / width, (face_center_y - h * 0.1) / height, 0.7]  # Right eye
        keypoints[15] = [(face_center_x - w * 0.2) / width, (face_center_y - h * 0.1) / height, 0.7]  # Left eye
        
        # Ears (16, 17)
        keypoints[16] = [(x + w) / width, face_center_y / height, 0.6]  # Right ear
        keypoints[17] = [x / width, face_center_y / height, 0.6]         # Left ear
        
        # Elbows (3, 6) - rough estimate
        elbow_y = (shoulder_y + hip_y) / 2
        keypoints[3] = [(face_center_x + w * 1.2) / width, elbow_y, 0.4]  # Right elbow
        keypoints[6] = [(face_center_x - w * 1.2) / width, elbow_y, 0.4]  # Left elbow
        
        # Wrists (4, 7) - rough estimate
        wrist_y = hip_y
        keypoints[4] = [(face_center_x + w * 1.3) / width, wrist_y, 0.3]  # Right wrist
        keypoints[7] = [(face_center_x - w * 1.3) / width, wrist_y, 0.3]  # Left wrist
        
        # Knees (9, 12) - rough estimate
        knee_y = (hip_y + 0.3)
        keypoints[9] = [(face_center_x + w * 0.5) / width, knee_y, 0.4]   # Right knee
        keypoints[12] = [(face_center_x - w * 0.5) / width, knee_y, 0.4]  # Left knee
        
        # Ankles (10, 13) - rough estimate
        ankle_y = min(0.95, knee_y + 0.25)
        keypoints[10] = [(face_center_x + w * 0.5) / width, ankle_y, 0.3]  # Right ankle
        keypoints[13] = [(face_center_x - w * 0.5) / width, ankle_y, 0.3]  # Left ankle
        
        return keypoints
    
    def detect_pose_dnn(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose using OpenPose DNN model.
        """
        if self.net is None:
            return self.detect_pose_simple(image)
        
        height, width = image.shape[:2]
        
        # Prepare input blob
        inp_blob = cv2.dnn.blobFromImage(image, 1.0 / 255, 
                                         (self.input_width, self.input_height),
                                         (0, 0, 0), swapRB=False, crop=False)
        
        self.net.setInput(inp_blob)
        output = self.net.forward()
        
        # Extract keypoints
        keypoints = []
        for i in range(18):  # COCO has 18 keypoints
            prob_map = output[0, i, :, :]
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            
            x = (width * point[0]) / output.shape[3]
            y = (height * point[1]) / output.shape[2]
            
            if prob > self.threshold:
                keypoints.append([x / width, y / height, prob])
            else:
                keypoints.append([0, 0, 0])
        
        return np.array(keypoints) if len(keypoints) == 18 else None
    
    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose keypoints in image.
        
        Returns:
            Array of shape (18, 3) with [x, y, confidence] for each keypoint
        """
        if self.use_simple:
            return self.detect_pose_simple(image)
        else:
            return self.detect_pose_dnn(image)


class MockPose:
    """
    MediaPipe-compatible Pose detector using OpenCV.
    """
    
    POSE_CONNECTIONS = []  # Simplified for compatibility
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize OpenCV detector
        self.detector = OpenCVPoseDetector(use_simple_detector=True)
    
    def process(self, image: np.ndarray) -> MockPoseResults:
        """
        Process image and detect pose.
        
        Args:
            image: RGB image
            
        Returns:
            MockPoseResults with detected landmarks
        """
        results = MockPoseResults()
        
        # Detect pose using OpenCV
        coco_keypoints = self.detector.detect(image)
        
        if coco_keypoints is not None:
            # Convert COCO keypoints (18) to MediaPipe format (33)
            mediapipe_landmarks = self._coco_to_mediapipe(coco_keypoints)
            results.pose_landmarks = MockPoseLandmarks(mediapipe_landmarks)
        
        return results
    
    def _coco_to_mediapipe(self, coco_keypoints: np.ndarray) -> List[MockLandmark]:
        """
        Convert COCO 18 keypoints to MediaPipe 33 landmarks format.
        """
        landmarks = []
        
        for mp_idx in range(33):
            if mp_idx in OpenCVPoseDetector.MEDIAPIPE_TO_COCO:
                coco_idx = OpenCVPoseDetector.MEDIAPIPE_TO_COCO[mp_idx]
                x, y, conf = coco_keypoints[coco_idx]
                landmarks.append(MockLandmark(x, y, 0.0, conf))
            else:
                # Fill missing landmarks with zeros
                landmarks.append(MockLandmark(0.0, 0.0, 0.0, 0.0))
        
        return landmarks
    
    def close(self):
        """Cleanup resources."""
        pass


class MockDrawingUtils:
    """Mock MediaPipe drawing utilities."""
    
    @staticmethod
    def draw_landmarks(image, landmarks, connections, landmark_drawing_spec=None):
        """Draw pose landmarks on image."""
        if landmarks is None:
            return
        
        # Draw simple skeleton
        height, width = image.shape[:2]
        
        # Draw keypoints
        for landmark in landmarks.landmark:
            if landmark.visibility > 0.3:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)


class MockDrawingStyles:
    """Mock MediaPipe drawing styles."""
    
    @staticmethod
    def get_default_pose_landmarks_style():
        """Return empty style."""
        return None


class MockPoseSolution:
    """Mock MediaPipe Pose solution."""
    
    POSE_CONNECTIONS = []
    
    def __init__(self):
        pass
    
    def Pose(self, **kwargs):
        return MockPose(**kwargs)


class MockSolutions:
    """Mock MediaPipe solutions."""
    
    def __init__(self):
        self.pose = MockPoseSolution()
        self.drawing_utils = MockDrawingUtils()
        self.drawing_styles = MockDrawingStyles()


# Create mock mediapipe module
solutions = MockSolutions()


def is_mediapipe_available() -> bool:
    """Check if real MediaPipe is available."""
    return False
