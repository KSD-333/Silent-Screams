"""
Dataset Utilities Module
-------------------------
Utilities for preparing pose keypoint sequences for training and inference.
Includes sliding window creation, normalization, and batch preparation.

Key Features:
- Sliding window generation with configurable stride
- Velocity feature computation (delta x, y between frames)
- Data augmentation utilities
- Batch preparation for training/inference
"""

import numpy as np
from typing import Tuple, Optional, List


def create_sliding_windows(features: np.ndarray,
                           window_length: int = 30,
                           stride: int = 15,
                           pad_last: bool = True) -> np.ndarray:
    """
    Create sliding windows from sequential features.
    
    Args:
        features: Sequential features (num_frames, num_features)
        window_length: Number of frames per window
        stride: Step size between windows
        pad_last: If True, pad the last window if it's shorter than window_length
    
    Returns:
        windows: Array of windows (num_windows, window_length, num_features)
    
    Example:
        >>> features = np.random.rand(100, 66)  # 100 frames, 66 features
        >>> windows = create_sliding_windows(features, window_length=30, stride=15)
        >>> print(windows.shape)  # (5, 30, 66)
    """
    num_frames, num_features = features.shape
    
    if num_frames < window_length:
        if pad_last:
            # Pad with zeros if sequence is too short
            padding = np.zeros((window_length - num_frames, num_features))
            features = np.vstack([features, padding])
            return features[np.newaxis, :, :]  # (1, window_length, num_features)
        else:
            raise ValueError(f"Not enough frames ({num_frames}) for window length {window_length}")
    
    windows = []
    start_idx = 0
    
    while start_idx + window_length <= num_frames:
        window = features[start_idx:start_idx + window_length]
        windows.append(window)
        start_idx += stride
    
    # Handle last window if it doesn't fit perfectly
    if pad_last and start_idx < num_frames:
        last_window = features[start_idx:]
        padding_needed = window_length - len(last_window)
        padding = np.zeros((padding_needed, num_features))
        last_window = np.vstack([last_window, padding])
        windows.append(last_window)
    
    return np.array(windows)  # (num_windows, window_length, num_features)


def add_velocity_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Add velocity features (delta x, y) to keypoint sequences.
    
    Args:
        keypoints: Keypoint sequences (num_frames, num_features)
                  where num_features = 66 (33 landmarks × 2 coords)
    
    Returns:
        enhanced_features: Keypoints with velocity (num_frames, num_features * 2)
                          First half: original keypoints, second half: velocities
    
    Example:
        >>> keypoints = np.random.rand(100, 66)
        >>> enhanced = add_velocity_features(keypoints)
        >>> print(enhanced.shape)  # (100, 132) - doubled features
    """
    num_frames, num_features = keypoints.shape
    
    # Calculate velocities (difference between consecutive frames)
    velocities = np.zeros_like(keypoints)
    velocities[1:] = keypoints[1:] - keypoints[:-1]
    velocities[0] = 0  # First frame has zero velocity
    
    # Concatenate keypoints and velocities
    enhanced_features = np.hstack([keypoints, velocities])
    
    return enhanced_features


def normalize_sequence(sequence: np.ndarray,
                       mean: Optional[np.ndarray] = None,
                       std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize a sequence using z-score normalization.
    
    Args:
        sequence: Input sequence (num_frames, num_features) or (batch, num_frames, num_features)
        mean: Pre-computed mean (if None, compute from sequence)
        std: Pre-computed std (if None, compute from sequence)
    
    Returns:
        normalized_sequence: Normalized sequence
        mean: Mean used for normalization
        std: Std used for normalization
    """
    if mean is None:
        mean = np.mean(sequence, axis=tuple(range(len(sequence.shape) - 1)), keepdims=True)
    
    if std is None:
        std = np.std(sequence, axis=tuple(range(len(sequence.shape) - 1)), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)  # Avoid division by zero
    
    normalized_sequence = (sequence - mean) / std
    
    return normalized_sequence, mean, std


def augment_sequence(sequence: np.ndarray,
                    noise_level: float = 0.01,
                    time_stretch_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """
    Apply data augmentation to a keypoint sequence.
    
    Args:
        sequence: Input sequence (window_length, num_features)
        noise_level: Standard deviation of Gaussian noise to add
        time_stretch_range: Range for temporal stretching (min, max)
    
    Returns:
        augmented_sequence: Augmented sequence (window_length, num_features)
    """
    window_length, num_features = sequence.shape
    augmented = sequence.copy()
    
    # Add Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, sequence.shape)
        augmented += noise
    
    # Time stretching (temporal augmentation)
    stretch_factor = np.random.uniform(*time_stretch_range)
    if stretch_factor != 1.0:
        stretched_length = int(window_length * stretch_factor)
        indices = np.linspace(0, window_length - 1, stretched_length)
        
        # Interpolate to new length
        stretched = np.zeros((stretched_length, num_features))
        for i in range(num_features):
            stretched[:, i] = np.interp(indices, np.arange(window_length), augmented[:, i])
        
        # Resample back to original length
        indices_back = np.linspace(0, stretched_length - 1, window_length)
        for i in range(num_features):
            augmented[:, i] = np.interp(indices_back, np.arange(stretched_length), stretched[:, i])
    
    return augmented


def prepare_batch(windows: np.ndarray,
                 labels: Optional[np.ndarray] = None,
                 batch_size: int = 32,
                 shuffle: bool = True) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Prepare batches for training or inference.
    
    Args:
        windows: Array of windows (num_windows, window_length, num_features)
        labels: Optional labels (num_windows,) for training
        batch_size: Batch size
        shuffle: Whether to shuffle data
    
    Returns:
        batches: List of (batch_windows, batch_labels) tuples
    
    Example:
        >>> windows = np.random.rand(100, 30, 66)
        >>> labels = np.random.randint(0, 2, 100)
        >>> batches = prepare_batch(windows, labels, batch_size=32)
        >>> print(len(batches))  # 4 batches (100 / 32 = 3.125 -> 4)
    """
    num_windows = len(windows)
    indices = np.arange(num_windows)
    
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    for start_idx in range(0, num_windows, batch_size):
        end_idx = min(start_idx + batch_size, num_windows)
        batch_indices = indices[start_idx:end_idx]
        
        batch_windows = windows[batch_indices]
        batch_labels = labels[batch_indices] if labels is not None else None
        
        batches.append((batch_windows, batch_labels))
    
    return batches


def split_dataset(features: np.ndarray,
                 labels: np.ndarray,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 shuffle: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        features: Feature array (num_samples, ...)
        labels: Label array (num_samples,)
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        shuffle: Whether to shuffle before splitting
    
    Returns:
        (train_features, train_labels), (val_features, val_labels), (test_features, test_labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    num_samples = len(features)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_data = (features[train_indices], labels[train_indices])
    val_data = (features[val_indices], labels[val_indices])
    test_data = (features[test_indices], labels[test_indices])
    
    return train_data, val_data, test_data


def balance_classes(features: np.ndarray,
                   labels: np.ndarray,
                   method: str = 'oversample') -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance class distribution in dataset.
    
    Args:
        features: Feature array (num_samples, ...)
        labels: Label array (num_samples,)
        method: 'oversample' or 'undersample'
    
    Returns:
        balanced_features: Balanced feature array
        balanced_labels: Balanced label array
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if method == 'oversample':
        # Oversample minority classes to match majority
        max_count = np.max(counts)
        
        balanced_features = []
        balanced_labels = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            label_features = features[label_indices]
            
            # Oversample to max_count
            oversample_indices = np.random.choice(
                len(label_features),
                size=max_count,
                replace=True
            )
            
            balanced_features.append(label_features[oversample_indices])
            balanced_labels.append(np.full(max_count, label))
        
        balanced_features = np.vstack(balanced_features)
        balanced_labels = np.concatenate(balanced_labels)
    
    elif method == 'undersample':
        # Undersample majority classes to match minority
        min_count = np.min(counts)
        
        balanced_features = []
        balanced_labels = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            label_features = features[label_indices]
            
            # Undersample to min_count
            undersample_indices = np.random.choice(
                len(label_features),
                size=min_count,
                replace=False
            )
            
            balanced_features.append(label_features[undersample_indices])
            balanced_labels.append(np.full(min_count, label))
        
        balanced_features = np.vstack(balanced_features)
        balanced_labels = np.concatenate(balanced_labels)
    
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    # Shuffle the balanced dataset
    indices = np.arange(len(balanced_labels))
    np.random.shuffle(indices)
    
    return balanced_features[indices], balanced_labels[indices]


def save_dataset(filepath: str,
                features: np.ndarray,
                labels: np.ndarray,
                metadata: Optional[dict] = None):
    """
    Save dataset to .npz file.
    
    Args:
        filepath: Output file path (e.g., 'dataset.npz')
        features: Feature array
        labels: Label array
        metadata: Optional metadata dictionary
    """
    save_dict = {
        'features': features,
        'labels': labels
    }
    
    if metadata:
        for key, value in metadata.items():
            save_dict[f'meta_{key}'] = value
    
    np.savez_compressed(filepath, **save_dict)
    print(f"Dataset saved to {filepath}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load dataset from .npz file.
    
    Args:
        filepath: Input file path
    
    Returns:
        features: Feature array
        labels: Label array
        metadata: Metadata dictionary
    """
    data = np.load(filepath)
    
    features = data['features']
    labels = data['labels']
    
    # Extract metadata
    metadata = {}
    for key in data.keys():
        if key.startswith('meta_'):
            metadata[key[5:]] = data[key]
    
    print(f"Dataset loaded from {filepath}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return features, labels, metadata


if __name__ == "__main__":
    """
    Test dataset utilities.
    """
    print("Testing dataset utilities...")
    
    # Generate dummy data
    num_frames = 100
    num_features = 66
    features = np.random.rand(num_frames, num_features)
    
    # Test sliding windows
    print("\n1. Testing sliding windows:")
    windows = create_sliding_windows(features, window_length=30, stride=15)
    print(f"   Input: {features.shape}")
    print(f"   Output: {windows.shape}")
    
    # Test velocity features
    print("\n2. Testing velocity features:")
    enhanced = add_velocity_features(features)
    print(f"   Input: {features.shape}")
    print(f"   Output: {enhanced.shape}")
    
    # Test normalization
    print("\n3. Testing normalization:")
    normalized, mean, std = normalize_sequence(windows)
    print(f"   Input: {windows.shape}")
    print(f"   Output: {normalized.shape}")
    print(f"   Mean shape: {mean.shape}, Std shape: {std.shape}")
    
    # Test augmentation
    print("\n4. Testing augmentation:")
    augmented = augment_sequence(windows[0])
    print(f"   Input: {windows[0].shape}")
    print(f"   Output: {augmented.shape}")
    
    # Test batch preparation
    print("\n5. Testing batch preparation:")
    labels = np.random.randint(0, 2, len(windows))
    batches = prepare_batch(windows, labels, batch_size=8)
    print(f"   Input: {windows.shape}, {labels.shape}")
    print(f"   Output: {len(batches)} batches")
    print(f"   First batch: {batches[0][0].shape}, {batches[0][1].shape}")
    
    # Test dataset split
    print("\n6. Testing dataset split:")
    train, val, test = split_dataset(windows, labels)
    print(f"   Train: {train[0].shape}, {train[1].shape}")
    print(f"   Val: {val[0].shape}, {val[1].shape}")
    print(f"   Test: {test[0].shape}, {test[1].shape}")
    
    print("\nAll tests passed!")
