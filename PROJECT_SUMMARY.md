# Silent Screams - Project Summary

## 📋 Overview

**Silent Screams** is a local, privacy-first ML-based behavior monitoring system that detects abnormal patterns through pose estimation and sequence classification. The system supports both live camera monitoring and video file analysis with a simple Streamlit GUI.

## 🎯 Key Features

✅ **Live Camera Monitoring** - Real-time pose tracking with instant alerts  
✅ **Video Upload Analysis** - Batch processing with timeline view  
✅ **Privacy-First** - 100% local processing, no cloud connectivity  
✅ **Customizable** - Configurable thresholds, sounds, and parameters  
✅ **Cross-Platform** - Works on Windows, macOS, and Linux  
✅ **Modular Design** - Well-commented, extensible codebase  
✅ **Mock Model** - Demo mode without requiring training data  

## 📁 Project Structure

```
silent-screams/
├── app.py                    # Main Streamlit GUI application
├── keypoint_extractor.py     # MediaPipe pose extraction
├── models.py                 # ML model architectures (LSTM, Transformer)
├── inference.py              # Real-time monitoring and video processing
├── dataset_utils.py          # Data preprocessing utilities
├── train_lstm.py             # Training script
├── collect_data.py           # Data collection helper
├── sound_utils.py            # Cross-platform audio alerts
├── requirements.txt          # Python dependencies
├── README.md                 # Full documentation
├── QUICKSTART.md             # Quick start guide
└── models/                   # Trained model files
    └── .gitkeep
```

## 🔧 Technical Stack

### Core Technologies
- **Python 3.8+** - Programming language
- **TensorFlow/Keras 2.15** - Deep learning framework
- **MediaPipe 0.10.8** - Pose estimation
- **OpenCV 4.8** - Computer vision
- **Streamlit 1.28** - GUI framework

### ML Architecture
- **Input:** 33 pose landmarks (66 features: x, y coordinates)
- **Window:** 30 frames (2 seconds @ 15fps)
- **Models:** Bi-LSTM or Transformer encoder
- **Output:** Binary classification (normal/abnormal)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py

# Load model (auto-creates mock model for demo)
# Click "Load Model" in sidebar
```

## 📊 Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `app.py` | Streamlit GUI | Live monitoring, video upload, settings |
| `keypoint_extractor.py` | Pose estimation | MediaPipe integration, normalization |
| `models.py` | Neural networks | LSTM/Transformer architectures |
| `inference.py` | Real-time processing | Rolling buffer, event detection |
| `dataset_utils.py` | Data prep | Sliding windows, augmentation |
| `train_lstm.py` | Model training | Training pipeline with validation |
| `collect_data.py` | Data collection | Video processing, labeling |
| `sound_utils.py` | Audio alerts | Cross-platform sound playback |

## 🎓 Training Your Own Model

### Step 1: Organize Videos
```
videos/
├── normal/      # Normal behavior videos
└── abnormal/    # Abnormal behavior videos
```

### Step 2: Collect Data
```bash
python collect_data.py \
    --normal_videos "./videos/normal/*.mp4" \
    --abnormal_videos "./videos/abnormal/*.mp4" \
    --output training_data.npz
```

### Step 3: Train Model
```bash
python train_lstm.py \
    --data_path training_data.npz \
    --epochs 50 \
    --batch_size 32
```

### Step 4: Deploy
- Place trained model in `models/` directory
- Update model path in GUI settings
- Start monitoring!

## 🔒 Privacy & Ethics

### Privacy Guarantees
- ✅ 100% local processing
- ✅ No data storage
- ✅ No network calls
- ✅ User-controlled permissions

### Ethical Use
- ✅ Obtain informed consent
- ✅ Comply with privacy laws
- ✅ Use for legitimate purposes
- ❌ No unauthorized surveillance

## 📈 Performance

| Hardware | Live FPS | Video Processing |
|----------|----------|------------------|
| CPU (i5) | 10-15 fps | 0.5x realtime |
| CPU (i7) | 15-20 fps | 1.0x realtime |
| GPU (GTX 1060) | 25-30 fps | 3.0x realtime |

## 🐛 Troubleshooting

**Camera not working?** Check system permissions  
**Low FPS?** Reduce frame rate in settings  
**Model errors?** Try mock model first  
**No audio?** Check system volume and audio backend  

## 📚 Documentation

- **README.md** - Complete documentation
- **QUICKSTART.md** - 5-minute setup guide
- **Code comments** - Inline documentation in all modules

## 🔬 Advanced Features

### Custom Model Architecture
Edit `models.py` to add new architectures

### Custom Features
Extend `keypoint_extractor.py` for face/hand landmarks

### Custom Alert Logic
Modify `inference.py` for different detection strategies

## 🤝 Contributing

This is a modular, well-documented codebase designed for extension:
- Add new model architectures
- Improve feature extraction
- Enhance GUI/UX
- Optimize performance
- Add unit tests

## 📄 License

Educational and research purposes. Use responsibly and ethically.

---

**Built with privacy and ethics in mind. Use responsibly.**
