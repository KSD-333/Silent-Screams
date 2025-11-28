# Testing Checklist - Silent Screams

Use this checklist to verify all functionality works correctly.

## ✅ Installation Testing

### Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] No import errors when running `python verify_setup.py`

### Verification
- [ ] `verify_setup.py` passes all checks
- [ ] Camera accessible
- [ ] GPU detected (if available)
- [ ] Audio backend available

## ✅ Module Testing

### keypoint_extractor.py
```bash
python keypoint_extractor.py
```
- [ ] Webcam opens successfully
- [ ] Pose landmarks detected and drawn
- [ ] Skeleton overlay visible
- [ ] FPS reasonable (>10 fps)
- [ ] Press 'q' to quit works

### dataset_utils.py
```bash
python dataset_utils.py
```
- [ ] All 6 tests pass
- [ ] Sliding windows created correctly
- [ ] Velocity features computed
- [ ] Normalization works
- [ ] No errors or warnings

### models.py
```bash
python models.py
```
- [ ] LSTM model builds successfully
- [ ] Transformer model builds successfully
- [ ] Mock model created and saved
- [ ] Model loading works
- [ ] Predictions run without errors

### inference.py
```bash
python inference.py
```
- [ ] Mock model created
- [ ] RealtimeMonitor processes frames
- [ ] Statistics tracked correctly
- [ ] No errors in output

### sound_utils.py
```bash
python sound_utils.py
```
- [ ] Beep sound generated
- [ ] Audio plays (you should hear a beep)
- [ ] No errors
- [ ] Cleanup successful

## ✅ GUI Testing (app.py)

### Initial Launch
```bash
streamlit run app.py
```
- [ ] Application opens in browser
- [ ] No errors in terminal
- [ ] Welcome screen displays
- [ ] Sidebar visible with settings

### Model Loading
- [ ] Click "Load Model" button
- [ ] Mock model created automatically (if not exists)
- [ ] Success message appears
- [ ] Model status shows "Model Loaded ✓"

### Settings Configuration
- [ ] Detection threshold slider works (0.0-1.0)
- [ ] Window length input accepts values
- [ ] Frame rate input accepts values
- [ ] Sound file path can be edited
- [ ] Mute audio checkbox toggles

### Live Monitoring Mode
- [ ] Click "📹 Live Camera Monitoring" button
- [ ] Mode switches to live monitoring
- [ ] Click "▶️ Start Monitoring"
- [ ] Camera permission granted
- [ ] Video feed displays
- [ ] Skeleton overlay visible on person
- [ ] Frame counter increments
- [ ] Detection status shows (✓ or ✗)
- [ ] Prediction probability displays
- [ ] Alert banner appears (randomly with mock model)
- [ ] Alert sound plays (if not muted)
- [ ] Alert log updates with timestamps
- [ ] Click "⏹️ Stop Monitoring" stops feed
- [ ] Camera released properly

### Video Upload Mode
- [ ] Click "📁 Analyze Uploaded Video" button
- [ ] Mode switches to upload
- [ ] File uploader appears
- [ ] Upload a test video (MP4/AVI/MOV)
- [ ] Video file name displays
- [ ] Click "🔍 Analyze Video"
- [ ] Progress bar shows processing
- [ ] Analysis completes without errors
- [ ] Results display (events or "no abnormal events")
- [ ] Event expandable sections work
- [ ] Frame thumbnails display
- [ ] Timestamps formatted correctly (HH:MM:SS)
- [ ] Confidence scores shown
- [ ] CSV download button appears
- [ ] Downloaded CSV contains correct data

## ✅ Training Pipeline Testing

### Data Collection
```bash
# Create test video folders first
mkdir -p videos/normal videos/abnormal
# Place test videos in folders

python collect_data.py \
    --normal_videos "./videos/normal/*.mp4" \
    --abnormal_videos "./videos/abnormal/*.mp4" \
    --output test_data.npz
```
- [ ] Videos found and processed
- [ ] Keypoints extracted
- [ ] Windows created
- [ ] Labels assigned correctly
- [ ] Dataset saved as .npz
- [ ] Statistics printed
- [ ] No errors

### Model Training
```bash
python train_lstm.py \
    --data_path test_data.npz \
    --epochs 5 \
    --batch_size 8
```
- [ ] Data loaded successfully
- [ ] Model built
- [ ] Training starts
- [ ] Progress shows for each epoch
- [ ] Validation metrics displayed
- [ ] Model saved to models/
- [ ] Metadata saved
- [ ] Test evaluation runs
- [ ] No errors

## ✅ Edge Cases & Error Handling

### No Camera
- [ ] Graceful error message when camera unavailable
- [ ] Application doesn't crash
- [ ] User can still use video upload mode

### Invalid Video File
- [ ] Error message for corrupted video
- [ ] Application doesn't crash
- [ ] User can upload another file

### Model File Missing
- [ ] Mock model auto-generated
- [ ] Warning message displayed
- [ ] Application continues to work

### No Pose Detected
- [ ] Fill-forward mechanism works
- [ ] No crashes on missed detections
- [ ] Status shows "Not Detected"

### Audio Backend Missing
- [ ] Fallback to system beep
- [ ] Or silent operation
- [ ] No crashes

## ✅ Performance Testing

### Live Monitoring
- [ ] FPS ≥ 10 (acceptable)
- [ ] FPS ≥ 15 (good)
- [ ] FPS ≥ 20 (excellent)
- [ ] No significant lag
- [ ] Memory usage stable (<2GB)
- [ ] CPU usage reasonable (<50% per core)

### Video Processing
- [ ] Progress updates smoothly
- [ ] Processing speed acceptable
- [ ] Memory doesn't grow unbounded
- [ ] Large videos (>5 min) process successfully

## ✅ Cross-Platform Testing

### Windows
- [ ] `setup.bat` runs successfully
- [ ] All modules work
- [ ] Camera accessible
- [ ] Audio plays (winsound fallback)
- [ ] GUI renders correctly

### macOS
- [ ] `setup.sh` runs successfully
- [ ] All modules work
- [ ] Camera accessible
- [ ] Audio plays (afplay fallback)
- [ ] GUI renders correctly

### Linux
- [ ] `setup.sh` runs successfully
- [ ] All modules work
- [ ] Camera accessible
- [ ] Audio plays (aplay/paplay fallback)
- [ ] GUI renders correctly

## ✅ Integration Testing

### End-to-End: Live Monitoring
1. [ ] Launch application
2. [ ] Load model
3. [ ] Configure settings
4. [ ] Start live monitoring
5. [ ] Perform normal movements
6. [ ] Perform abnormal movements (if using trained model)
7. [ ] Verify alerts trigger appropriately
8. [ ] Check alert log
9. [ ] Stop monitoring
10. [ ] Verify camera released

### End-to-End: Video Analysis
1. [ ] Launch application
2. [ ] Load model
3. [ ] Upload video
4. [ ] Analyze video
5. [ ] Review detected events
6. [ ] Download results CSV
7. [ ] Verify CSV contents

### End-to-End: Training
1. [ ] Collect videos
2. [ ] Run `collect_data.py`
3. [ ] Verify dataset created
4. [ ] Run `train_lstm.py`
5. [ ] Verify model trained
6. [ ] Load model in GUI
7. [ ] Test with live monitoring
8. [ ] Verify predictions reasonable

## ✅ Documentation Testing

- [ ] README.md renders correctly
- [ ] QUICKSTART.md steps work
- [ ] PROJECT_SUMMARY.md accurate
- [ ] INDEX.md links valid
- [ ] Code comments helpful
- [ ] Docstrings complete

## 🐛 Known Issues

Document any issues found during testing:

| Issue | Severity | Workaround | Status |
|-------|----------|------------|--------|
| Example: Low FPS on old CPU | Low | Reduce frame rate | Known limitation |
| | | | |
| | | | |

## 📝 Testing Notes

**Date:** ___________  
**Tester:** ___________  
**Platform:** ___________  
**Python Version:** ___________  

**Overall Result:** ⬜ Pass ⬜ Fail ⬜ Partial

**Comments:**
_____________________________________________
_____________________________________________
_____________________________________________

---

**Testing Complete!** 🎉

If all checks pass, the system is ready for deployment.
