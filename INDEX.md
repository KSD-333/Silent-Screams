# Silent Screams - Project Index

## 📚 Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| **README.md** | Complete documentation with architecture, usage, and troubleshooting | All users |
| **QUICKSTART.md** | 5-minute setup and usage guide | New users |
| **PROJECT_SUMMARY.md** | High-level overview and technical summary | Developers |
| **INDEX.md** | This file - navigation guide | All users |
| **LICENSE** | MIT license with ethical use notice | All users |

## 🔧 Core Application Files

| File | Lines | Purpose |
|------|-------|---------|
| **app.py** | ~450 | Main Streamlit GUI application |
| **keypoint_extractor.py** | ~300 | MediaPipe pose estimation and normalization |
| **models.py** | ~450 | Neural network architectures (LSTM, Transformer) |
| **inference.py** | ~350 | Real-time monitoring and video processing |
| **dataset_utils.py** | ~400 | Data preprocessing and augmentation |
| **train_lstm.py** | ~300 | Model training pipeline |
| **sound_utils.py** | ~250 | Cross-platform audio alerts |

## 🛠️ Utility Scripts

| File | Purpose |
|------|---------|
| **collect_data.py** | Extract features from videos and create training datasets |
| **verify_setup.py** | Verify installation and system readiness |
| **setup.bat** | Automated Windows installation script |
| **setup.sh** | Automated Unix/Linux/macOS installation script |

## 📦 Configuration Files

| File | Purpose |
|------|---------|
| **requirements.txt** | Python package dependencies |
| **.gitignore** | Git ignore rules |
| **models/.gitkeep** | Placeholder for model directory |

## 🚀 Quick Navigation

### For First-Time Users
1. Read **QUICKSTART.md** (5 minutes)
2. Run `setup.bat` (Windows) or `setup.sh` (Unix/Linux/macOS)
3. Run `python verify_setup.py`
4. Run `streamlit run app.py`

### For Developers
1. Read **PROJECT_SUMMARY.md** for architecture overview
2. Review **README.md** for detailed documentation
3. Explore code files (all well-commented)
4. Check inline documentation in each module

### For Training Custom Models
1. Organize videos into `normal/` and `abnormal/` folders
2. Run `python collect_data.py --normal_videos ... --abnormal_videos ...`
3. Run `python train_lstm.py --data_path training_data.npz`
4. Use trained model in GUI

## 📊 File Dependencies

```
app.py
├── keypoint_extractor.py
│   └── mediapipe, opencv
├── models.py
│   └── tensorflow, keras
├── inference.py
│   ├── keypoint_extractor.py
│   └── models.py
└── sound_utils.py
    └── simpleaudio, playsound

train_lstm.py
├── models.py
└── dataset_utils.py

collect_data.py
├── keypoint_extractor.py
└── dataset_utils.py
```

## 🎯 Common Tasks

### Task: Run the application
```bash
streamlit run app.py
```

### Task: Train a model
```bash
python train_lstm.py --data_path training_data.npz --epochs 50
```

### Task: Collect training data
```bash
python collect_data.py \
    --normal_videos "./videos/normal/*.mp4" \
    --abnormal_videos "./videos/abnormal/*.mp4" \
    --output training_data.npz
```

### Task: Test camera and pose detection
```bash
python keypoint_extractor.py
```

### Task: Verify installation
```bash
python verify_setup.py
```

### Task: Test audio system
```bash
python sound_utils.py
```

## 🔍 Code Organization

### Module Structure
Each Python module follows this structure:
1. **Docstring** - Module purpose and features
2. **Imports** - Standard library, third-party, local
3. **Constants** - Configuration constants
4. **Classes** - Main functionality
5. **Functions** - Helper functions
6. **Main block** - Testing/demo code

### Naming Conventions
- **Classes:** PascalCase (e.g., `KeypointExtractor`)
- **Functions:** snake_case (e.g., `extract_keypoints`)
- **Constants:** UPPER_SNAKE_CASE (e.g., `COOLDOWN_SECONDS`)
- **Private:** Leading underscore (e.g., `_normalize_keypoints`)

## 📈 Project Statistics

- **Total Python Files:** 8
- **Total Lines of Code:** ~2,500
- **Documentation Files:** 5
- **Setup Scripts:** 2
- **Dependencies:** 12 core packages

## 🔗 Key Concepts

### Pose Estimation
- **MediaPipe Pose:** 33 body landmarks
- **Normalization:** Torso-relative coordinates
- **Features:** 66 dimensions (x, y for each landmark)

### Sequence Classification
- **Window Length:** 30 frames (2 seconds @ 15fps)
- **Stride:** 15 frames (50% overlap)
- **Models:** Bi-LSTM or Transformer
- **Output:** Binary (normal/abnormal)

### Real-time Monitoring
- **Rolling Buffer:** Last 30 frames
- **Inference Stride:** Every 5 frames
- **Cooldown:** 1.5 seconds between alerts
- **Alert:** Visual banner + audio sound

## 🎓 Learning Path

### Beginner
1. Run the application with mock model
2. Try live monitoring mode
3. Upload and analyze a video
4. Adjust settings and observe behavior

### Intermediate
1. Collect sample training data
2. Train a simple model
3. Evaluate model performance
4. Deploy custom model in GUI

### Advanced
1. Implement custom model architecture
2. Add new feature extraction methods
3. Optimize performance for your use case
4. Extend GUI with new functionality

## 🔒 Privacy & Ethics

All files in this project follow these principles:
- ✅ Local-only processing
- ✅ No data storage
- ✅ No network calls
- ✅ User consent required
- ✅ Ethical use guidelines

## 📞 Support

- **Documentation:** See README.md
- **Quick Help:** See QUICKSTART.md
- **Code Issues:** Check inline comments
- **Setup Issues:** Run verify_setup.py

---

**Last Updated:** 2025-09-30  
**Version:** 1.0  
**License:** MIT with Ethical Use Notice
