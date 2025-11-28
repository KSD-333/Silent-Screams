"""
Setup Verification Script
--------------------------
Verify that all dependencies are installed correctly and the system is ready to run.

Usage:
    python verify_setup.py
"""

import sys
import importlib


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {package_name} ({version})")
        return True
    except ImportError:
        print(f"  ✗ {package_name} (not installed)")
        return False


def check_camera_access():
    """Check if camera is accessible."""
    print("\nChecking camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("  ✓ Camera accessible")
                return True
            else:
                print("  ✗ Camera opened but cannot read frames")
                return False
        else:
            print("  ✗ Cannot open camera (check permissions)")
            return False
    
    except Exception as e:
        print(f"  ✗ Camera check failed: {e}")
        return False


def check_gpu_availability():
    """Check if GPU is available for TensorFlow."""
    print("\nChecking GPU availability...")
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 0:
            print(f"  ✓ {len(gpus)} GPU(s) detected:")
            for gpu in gpus:
                print(f"    - {gpu.name}")
            return True
        else:
            print("  ℹ No GPU detected (will use CPU)")
            return True
    
    except Exception as e:
        print(f"  ✗ GPU check failed: {e}")
        return False


def check_audio_backend():
    """Check if audio playback is available."""
    print("\nChecking audio backends...")
    
    backends = []
    
    # Check simpleaudio
    try:
        import simpleaudio
        backends.append("simpleaudio")
        print("  ✓ simpleaudio")
    except ImportError:
        print("  ✗ simpleaudio")
    
    # Check playsound
    try:
        import playsound
        backends.append("playsound")
        print("  ✓ playsound")
    except ImportError:
        print("  ✗ playsound")
    
    # Check platform-specific
    if sys.platform == 'win32':
        try:
            import winsound
            backends.append("winsound")
            print("  ✓ winsound (Windows)")
        except ImportError:
            print("  ✗ winsound")
    
    if len(backends) > 0:
        print(f"  ✓ Audio playback available ({', '.join(backends)})")
        return True
    else:
        print("  ⚠ No audio backends available (alerts will be silent)")
        return True  # Not critical


def check_project_files():
    """Check if all required project files exist."""
    print("\nChecking project files...")
    
    required_files = [
        'app.py',
        'keypoint_extractor.py',
        'models.py',
        'inference.py',
        'dataset_utils.py',
        'train_lstm.py',
        'sound_utils.py',
        'requirements.txt',
        'README.md'
    ]
    
    import os
    all_exist = True
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (missing)")
            all_exist = False
    
    return all_exist


def test_mediapipe():
    """Test MediaPipe pose detection."""
    print("\nTesting MediaPipe pose detection...")
    try:
        import mediapipe as mp
        import numpy as np
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        
        # Create dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pose.process(dummy_image)
        
        pose.close()
        
        print("  ✓ MediaPipe pose detection working")
        return True
    
    except Exception as e:
        print(f"  ✗ MediaPipe test failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    try:
        from models import build_lstm_model
        
        model = build_lstm_model(window_length=30, num_features=66)
        
        print(f"  ✓ Model created successfully")
        print(f"    Input shape: {model.input_shape}")
        print(f"    Output shape: {model.output_shape}")
        return True
    
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Silent Screams - Setup Verification")
    print("=" * 60)
    
    results = []
    
    # Check Python version
    print("\n[1/9] Python Version")
    results.append(check_python_version())
    
    # Check core packages
    print("\n[2/9] Core Dependencies")
    packages = [
        ('streamlit', 'streamlit'),
        ('opencv-python', 'cv2'),
        ('mediapipe', 'mediapipe'),
        ('tensorflow', 'tensorflow'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('Pillow', 'PIL'),
        ('tqdm', 'tqdm')
    ]
    
    for pkg_name, import_name in packages:
        results.append(check_package(pkg_name, import_name))
    
    # Check audio packages
    print("\n[3/9] Audio Dependencies")
    check_audio_backend()  # Not critical, so don't add to results
    
    # Check camera
    print("\n[4/9] Camera Access")
    results.append(check_camera_access())
    
    # Check GPU
    print("\n[5/9] GPU Availability")
    check_gpu_availability()  # Not critical
    
    # Check project files
    print("\n[6/9] Project Files")
    results.append(check_project_files())
    
    # Test MediaPipe
    print("\n[7/9] MediaPipe Test")
    results.append(test_mediapipe())
    
    # Test model creation
    print("\n[8/9] Model Creation Test")
    results.append(test_model_creation())
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print("\n✅ All checks passed! System is ready to run.")
        print("\nNext steps:")
        print("  1. Run the application: streamlit run app.py")
        print("  2. Load the model in the GUI sidebar")
        print("  3. Choose a mode and start monitoring!")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check camera permissions in system settings")
        print("  - Ensure all project files are present")
    
    print("\n" + "=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
