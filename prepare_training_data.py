"""
Prepare Training Data for Distress Detection Model
---------------------------------------------------
This script extracts keypoints from videos and prepares them for training.

Usage:
    python prepare_training_data.py --normal_videos_dir ./videos/normal --distress_videos_dir ./videos/distress --output training_data.npz
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from keypoint_extractor import KeypointExtractor
from dataset_utils import create_sliding_windows

def extract_features_from_videos(video_dir, label, window_length=30, stride=15):
    """
    Extract keypoint features from all videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        label: 0 for normal, 1 for distress
        window_length: Number of frames per window
        stride: Stride for sliding window
    
    Returns:
        X: Feature windows (N, window_length, num_features)
        y: Labels (N,)
    """
    extractor = KeypointExtractor()
    all_windows = []
    all_labels = []
    
    video_files = list(Path(video_dir).glob('*.mp4')) + \
                  list(Path(video_dir).glob('*.avi')) + \
                  list(Path(video_dir).glob('*.mov'))
    
    print(f"\nProcessing {len(video_files)} videos from {video_dir}...")
    
    for video_path in tqdm(video_files):
        try:
            # Extract keypoints from video
            features, confidences = extractor.process_video(str(video_path))
            
            if len(features) < window_length:
                print(f"Skipping {video_path.name} - too short ({len(features)} frames)")
                continue
            
            # Create sliding windows
            windows = create_sliding_windows(
                features, 
                window_length=window_length, 
                stride=stride
            )
            
            # Add to dataset
            all_windows.extend(windows)
            all_labels.extend([label] * len(windows))
            
            print(f"  ✓ {video_path.name}: {len(windows)} windows extracted")
            
        except Exception as e:
            print(f"  ✗ Error processing {video_path.name}: {e}")
            continue
    
    extractor.reset()
    
    return np.array(all_windows), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description='Prepare training data from videos')
    parser.add_argument('--normal_videos_dir', type=str, required=True,
                       help='Directory containing normal behavior videos')
    parser.add_argument('--distress_videos_dir', type=str, required=True,
                       help='Directory containing distress behavior videos')
    parser.add_argument('--output', type=str, default='training_data.npz',
                       help='Output file path (.npz)')
    parser.add_argument('--window_length', type=int, default=30,
                       help='Number of frames per window')
    parser.add_argument('--stride', type=int, default=15,
                       help='Stride for sliding window')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.normal_videos_dir):
        print(f"Error: Normal videos directory not found: {args.normal_videos_dir}")
        return
    
    if not os.path.exists(args.distress_videos_dir):
        print(f"Error: Distress videos directory not found: {args.distress_videos_dir}")
        return
    
    print("=" * 60)
    print("TRAINING DATA PREPARATION")
    print("=" * 60)
    
    # Extract features from normal videos (label=0)
    print("\n[1/2] Processing NORMAL behavior videos...")
    X_normal, y_normal = extract_features_from_videos(
        args.normal_videos_dir,
        label=0,
        window_length=args.window_length,
        stride=args.stride
    )
    
    # Extract features from distress videos (label=1)
    print("\n[2/2] Processing DISTRESS behavior videos...")
    X_distress, y_distress = extract_features_from_videos(
        args.distress_videos_dir,
        label=1,
        window_length=args.window_length,
        stride=args.stride
    )
    
    # Combine datasets
    X = np.concatenate([X_normal, X_distress], axis=0)
    y = np.concatenate([y_normal, y_distress], axis=0)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Save to file
    np.savez(args.output, X=X, y=y)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal samples: {len(X)}")
    print(f"  - Normal: {np.sum(y == 0)} samples")
    print(f"  - Distress: {np.sum(y == 1)} samples")
    print(f"\nData shape: {X.shape}")
    print(f"  - Window length: {X.shape[1]} frames")
    print(f"  - Features per frame: {X.shape[2]}")
    print(f"\nSaved to: {args.output}")
    print("\nNext step: Train the model using:")
    print(f"  python train_lstm.py --data_path {args.output} --epochs 50")


if __name__ == "__main__":
    main()
