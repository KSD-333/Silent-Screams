# Quick Start Guide - Silent Screams

Get up and running in 5 minutes!

## 🚀 Installation

### 1. Install Python Dependencies

```bash
# Navigate to project directory
cd silent-screams

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** Installation may take 5-10 minutes due to TensorFlow and MediaPipe.

## 🎮 Running the Application

### Launch the GUI

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎯 First Time Setup

### Step 1: Load Model

1. The app will automatically create a **mock model** for demo purposes
2. In the left sidebar, click **"Load Model"**
3. The mock model will be created at `models/mock_model.h5`

### Step 2: Choose Mode

**Option A: Live Camera Monitoring**
1. Click **"📹 Live Camera Monitoring"**
2. Click **"▶️ Start Monitoring"**
3. Grant camera permissions when prompted
4. Watch the live feed with skeleton overlay
5. Alerts will trigger when abnormal behavior is detected (randomly with mock model)

**Option B: Video Upload**
1. Click **"📁 Analyze Uploaded Video"**
2. Upload a video file (MP4, AVI, MOV)
3. Click **"🔍 Analyze Video"**
4. View detected events with timestamps

## ⚙️ Adjusting Settings

In the left sidebar, you can configure:

- **Detection Threshold:** 0.8 (higher = fewer false positives)
- **Window Length:** 30 frames (2 seconds @ 15fps)
- **Processing Frame Rate:** 15 fps (lower = faster processing)
- **Mute Audio:** Toggle alert sounds on/off

## 🤖 Using a Real Model

The mock model produces random predictions. To use a real trained model:

### Option 1: Train Your Own Model

```bash
# Prepare your training data as .npz file with:
# - 'features': (num_sequences, window_length, num_features)
# - 'labels': (num_sequences,) with 0=normal, 1=abnormal

# Train model
python train_lstm.py --data_path ./training_data.npz --epochs 50

# The trained model will be saved to models/
```

### Option 2: Use Pre-trained Model

1. Place your `.h5` model file in the `models/` directory
2. In the GUI sidebar, update "Model Path" to your model filename
3. Click "Load Model"

## 🧪 Testing Individual Components

### Test Keypoint Extraction

```bash
python keypoint_extractor.py
```

This will open your webcam and display pose landmarks in real-time. Press 'q' to quit.

### Test Dataset Utilities

```bash
python dataset_utils.py
```

### Test Model Architectures

```bash
python models.py
```

### Test Inference

```bash
python inference.py
```

### Test Sound System

```bash
python sound_utils.py
```

## 📊 Collecting Training Data

To train a real model, you need labeled sequences:

### 1. Extract Keypoints from Videos

```python
from keypoint_extractor import extract_keypoints_from_video

# Extract from normal behavior videos
normal_keypoints, _ = extract_keypoints_from_video("normal_video.mp4")

# Extract from abnormal behavior videos
abnormal_keypoints, _ = extract_keypoints_from_video("abnormal_video.mp4")
```

### 2. Create Sliding Windows

```python
from dataset_utils import create_sliding_windows
import numpy as np

# Create windows
normal_windows = create_sliding_windows(normal_keypoints, window_length=30, stride=15)
abnormal_windows = create_sliding_windows(abnormal_keypoints, window_length=30, stride=15)

# Create labels
normal_labels = np.zeros(len(normal_windows))
abnormal_labels = np.ones(len(abnormal_windows))

# Combine
features = np.vstack([normal_windows, abnormal_windows])
labels = np.concatenate([normal_labels, abnormal_labels])

# Save
np.savez_compressed('training_data.npz', features=features, labels=labels)
```

### 3. Train Model

```bash
python train_lstm.py --data_path training_data.npz --epochs 50
```

## 🐛 Troubleshooting

### Camera Not Working

- **Windows:** Settings → Privacy → Camera → Allow apps to access camera
- **macOS:** System Preferences → Security & Privacy → Camera
- **Linux:** Check `/dev/video0` permissions

### Low FPS

- Reduce "Processing Frame Rate" to 10 or 5
- Reduce "Window Length" to 20
- Close other applications using the camera

### Audio Not Playing

- Check system volume
- Try toggling "Mute Audio" off
- The system will auto-generate a beep if `alarm.wav` is missing

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# If TensorFlow issues on Windows:
pip install tensorflow==2.15.0 --upgrade

# If MediaPipe issues:
pip install mediapipe==0.10.8 --upgrade
```

### Model Loading Errors

- Ensure model file exists at specified path
- Check TensorFlow version compatibility
- Try creating a fresh mock model

## 📚 Next Steps

1. **Collect Real Data:** Record videos of normal and abnormal behaviors
2. **Label Sequences:** Mark which sequences contain abnormal behavior
3. **Train Model:** Use `train_lstm.py` with your labeled data
4. **Evaluate:** Test on held-out videos
5. **Deploy:** Use your trained model in the GUI

## 🔒 Privacy Reminder

- All processing is 100% local
- No data is stored or transmitted
- Always obtain consent before monitoring
- Use responsibly and ethically

## 💡 Tips

- Start with a high threshold (0.8-0.9) to reduce false positives
- Collect diverse training data (different people, lighting, angles)
- Balance your dataset (equal normal and abnormal samples)
- Use data augmentation to improve generalization
- Monitor performance on validation set during training

## 📖 Full Documentation

See `README.md` for complete documentation, architecture details, and advanced usage.

---

**Need Help?** Check the troubleshooting section or review the code comments in each module.
