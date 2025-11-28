"""
Data Collection Script
----------------------
Helper script to collect and prepare training data from videos.

This script helps you:
1. Extract keypoints from video files
2. Create sliding windows
3. Label sequences (normal/abnormal)
4. Save as training-ready .npz file

Usage:
    python collect_data.py --normal_videos ./normal/*.mp4 --abnormal_videos ./abnormal/*.mp4 --output training_data.npz
"""

import os
import argparse
import glob
import numpy as np
from tqdm import tqdm

from keypoint_extractor import extract_keypoints_from_video
from dataset_utils import create_sliding_windows, save_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Collect training data from videos')
    
    parser.add_argument('--normal_videos', type=str, required=True,
                       help='Path pattern for normal behavior videos (e.g., "./normal/*.mp4")')
    parser.add_argument('--abnormal_videos', type=str, required=True,
                       help='Path pattern for abnormal behavior videos (e.g., "./abnormal/*.mp4")')
    parser.add_argument('--output', type=str, default='training_data.npz',
                       help='Output file path for training data')
    parser.add_argument('--window_length', type=int, default=30,
                       help='Number of frames per window')
    parser.add_argument('--stride', type=int, default=15,
                       help='Stride for sliding window')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to process per video (None = all)')
    
    return parser.parse_args()


def process_video_files(video_pattern: str, 
                       label: int,
                       window_length: int,
                       stride: int,
                       max_frames: int = None):
    """
    Process multiple video files and extract windowed features.
    
    Args:
        video_pattern: Glob pattern for video files
        label: Label for these videos (0=normal, 1=abnormal)
        window_length: Frames per window
        stride: Stride for sliding window
        max_frames: Max frames per video
    
    Returns:
        windows: Array of feature windows
        labels: Array of labels
    """
    # Find video files
    video_files = glob.glob(video_pattern)
    
    if len(video_files) == 0:
        print(f"Warning: No videos found matching pattern: {video_pattern}")
        return np.array([]), np.array([])
    
    print(f"\nProcessing {len(video_files)} videos with label={label}...")
    
    all_windows = []
    
    for video_path in tqdm(video_files, desc=f"Label {label}"):
        try:
            # Extract keypoints
            keypoints, confidences = extract_keypoints_from_video(
                video_path, 
                max_frames=max_frames
            )
            
            # Check if we got enough frames
            if len(keypoints) < window_length:
                print(f"  Skipping {os.path.basename(video_path)}: too short ({len(keypoints)} frames)")
                continue
            
            # Create sliding windows
            windows = create_sliding_windows(
                keypoints,
                window_length=window_length,
                stride=stride,
                pad_last=False
            )
            
            all_windows.append(windows)
            
            print(f"  ✓ {os.path.basename(video_path)}: {len(windows)} windows")
        
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(video_path)}: {e}")
            continue
    
    if len(all_windows) == 0:
        return np.array([]), np.array([])
    
    # Combine all windows
    all_windows = np.vstack(all_windows)
    labels = np.full(len(all_windows), label)
    
    return all_windows, labels


def main():
    """Main data collection pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("Silent Screams - Data Collection")
    print("=" * 60)
    
    # Process normal videos
    print("\n[1/2] Processing NORMAL behavior videos...")
    normal_windows, normal_labels = process_video_files(
        args.normal_videos,
        label=0,
        window_length=args.window_length,
        stride=args.stride,
        max_frames=args.max_frames
    )
    
    # Process abnormal videos
    print("\n[2/2] Processing ABNORMAL behavior videos...")
    abnormal_windows, abnormal_labels = process_video_files(
        args.abnormal_videos,
        label=1,
        window_length=args.window_length,
        stride=args.stride,
        max_frames=args.max_frames
    )
    
    # Check if we got any data
    if len(normal_windows) == 0 and len(abnormal_windows) == 0:
        print("\n❌ No data collected! Please check your video paths.")
        return
    
    # Combine datasets
    print("\n" + "=" * 60)
    print("Combining datasets...")
    
    if len(normal_windows) > 0 and len(abnormal_windows) > 0:
        features = np.vstack([normal_windows, abnormal_windows])
        labels = np.concatenate([normal_labels, abnormal_labels])
    elif len(normal_windows) > 0:
        features = normal_windows
        labels = normal_labels
        print("⚠️  Warning: Only normal videos found!")
    else:
        features = abnormal_windows
        labels = abnormal_labels
        print("⚠️  Warning: Only abnormal videos found!")
    
    # Shuffle
    print("Shuffling data...")
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    print(f"Total sequences: {len(features)}")
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"\nClass distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "Normal" if label == 0 else "Abnormal"
        percentage = count / len(labels) * 100
        print(f"  {label_name} (label={label}): {count} sequences ({percentage:.1f}%)")
    
    # Save dataset
    print("\n" + "=" * 60)
    print("Saving dataset...")
    
    metadata = {
        'window_length': args.window_length,
        'stride': args.stride,
        'num_features': features.shape[2],
        'normal_videos': args.normal_videos,
        'abnormal_videos': args.abnormal_videos
    }
    
    save_dataset(args.output, features, labels, metadata)
    
    print("\n" + "=" * 60)
    print("✅ Data collection completed successfully!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Train model: python train_lstm.py --data_path {args.output} --epochs 50")
    print(f"2. Use trained model in GUI: streamlit run app.py")


if __name__ == "__main__":
    main()
