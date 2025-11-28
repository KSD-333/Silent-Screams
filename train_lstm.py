"""
Training Script for LSTM Model
-------------------------------
Train a Bi-LSTM model on labeled pose keypoint sequences.

This script provides a complete training pipeline:
- Load and preprocess training data
- Build and configure model
- Train with validation and checkpointing
- Save best model

Usage:
    python train_lstm.py --data_path ./training_data --epochs 50

Data Format:
    Training data should be in .npz format with:
    - 'features': (num_sequences, window_length, num_features)
    - 'labels': (num_sequences,) with 0=normal, 1=abnormal
"""

import os
import argparse
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

from models import build_lstm_model, build_transformer_model, save_model_with_metadata
from dataset_utils import split_dataset, balance_classes, prepare_batch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LSTM model for behavior detection')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data (.npz file or directory)')
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'transformer'],
                       help='Model architecture to use')
    parser.add_argument('--window_length', type=int, default=30,
                       help='Number of frames per sequence')
    parser.add_argument('--num_features', type=int, default=66,
                       help='Number of features per frame')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--lstm_units', type=int, default=256,
                       help='Number of LSTM units')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of LSTM/Transformer layers')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                       help='Dropout rate')
    parser.add_argument('--balance_classes', action='store_true', default=True,
                       help='Balance class distribution')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Apply data augmentation')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save trained model')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name for saved model (default: auto-generated)')
    
    return parser.parse_args()


def load_training_data(data_path: str):
    """
    Load training data from .npz file or directory.
    
    Args:
        data_path: Path to .npz file or directory containing data files
    
    Returns:
        features: (num_sequences, window_length, num_features)
        labels: (num_sequences,)
    """
    if os.path.isfile(data_path) and data_path.endswith('.npz'):
        # Load single .npz file
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)
        # Support both 'features'/'labels' and 'X'/'y' naming conventions
        features = data['X'] if 'X' in data else data['features']
        labels = data['y'] if 'y' in data else data['labels']
    
    elif os.path.isdir(data_path):
        # Load all .npz files from directory
        print(f"Loading data from directory {data_path}...")
        features_list = []
        labels_list = []
        
        for filename in os.listdir(data_path):
            if filename.endswith('.npz'):
                filepath = os.path.join(data_path, filename)
                data = np.load(filepath)
                features_list.append(data['features'])
                labels_list.append(data['labels'])
        
        if len(features_list) == 0:
            raise ValueError(f"No .npz files found in {data_path}")
        
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
    
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    
    print(f"Loaded {len(features)} sequences")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    return features, labels


def build_model(args):
    """
    Build model based on arguments.
    
    Args:
        args: Command line arguments
    
    Returns:
        Compiled Keras model
    """
    print(f"\nBuilding {args.model_type.upper()} model...")
    
    if args.model_type == 'lstm':
        model = build_lstm_model(
            window_length=args.window_length,
            num_features=args.num_features,
            lstm_units=args.lstm_units,
            num_lstm_layers=args.num_layers,
            dropout_rate=args.dropout_rate
        )
    
    elif args.model_type == 'transformer':
        model = build_transformer_model(
            window_length=args.window_length,
            num_features=args.num_features,
            num_heads=4,
            num_transformer_blocks=args.num_layers,
            dropout_rate=args.dropout_rate
        )
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Recompile with custom learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(),
                keras.metrics.AUC(name='auc')]
    )
    
    print(model.summary())
    
    return model


def train_model(model, train_data, val_data, args):
    """
    Train the model with validation and checkpointing.
    
    Args:
        model: Keras model
        train_data: (train_features, train_labels)
        val_data: (val_features, val_labels)
        args: Command line arguments
    
    Returns:
        Training history
    """
    train_features, train_labels = train_data
    val_features, val_labels = val_data
    
    print(f"\nTraining model...")
    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate model name if not provided
    if args.model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_name = f"{args.model_type}_model_{timestamp}"
    
    model_path = os.path.join(args.output_dir, f"{args.model_name}.h5")
    
    # Callbacks
    callbacks = [
        # Model checkpoint - save best model
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping with more patience
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.output_dir, 'logs', args.model_name),
            histogram_freq=1
        )
    ]
    
    # Train model with class weights for imbalanced data
    class_weights = None
    if not args.balance_classes:
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = dict(enumerate(class_weights))
        print(f"Using class weights: {class_weights}")
    
    # Train model
    history = model.fit(
        train_features, train_labels,
        validation_data=(val_features, val_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {model_path}")
    
    return history


def evaluate_model(model, test_data):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained Keras model
        test_data: (test_features, test_labels)
    """
    test_features, test_labels = test_data
    
    print(f"\nEvaluating model on test set ({len(test_features)} samples)...")
    
    results = model.evaluate(test_features, test_labels, verbose=0)
    
    print(f"Test Results:")
    for metric_name, value in zip(model.metrics_names, results):
        print(f"  {metric_name}: {value:.4f}")
    
    # Confusion matrix
    predictions = model.predict(test_features, verbose=0)
    pred_classes = (predictions[:, 0] >= 0.5).astype(int)
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(test_labels, pred_classes)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(test_labels, pred_classes, 
                                target_names=['Normal', 'Abnormal']))


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("Silent Screams - Model Training")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("\nNo GPU detected, using CPU")
    
    # Load data
    features, labels = load_training_data(args.data_path)
    
    # Balance classes if requested
    if args.balance_classes:
        print("\nBalancing classes...")
        features, labels = balance_classes(features, labels, method='oversample')
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Balanced class distribution:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} samples")
    
    # Split dataset
    print("\nSplitting dataset...")
    train_data, val_data, test_data = split_dataset(
        features, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True
    )
    
    # Build model
    model = build_model(args)
    
    # Train model
    history = train_model(model, train_data, val_data, args)
    
    # Evaluate on test set
    evaluate_model(model, test_data)
    
    # Save model with metadata
    model_path = os.path.join(args.output_dir, f"{args.model_name}.h5")
    metadata = {
        'model_type': args.model_type,
        'window_length': args.window_length,
        'num_features': args.num_features,
        'training_samples': len(train_data[0]),
        'validation_samples': len(val_data[0]),
        'test_samples': len(test_data[0]),
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_auc': float(history.history['auc'][-1]),
        'final_val_auc': float(history.history['val_auc'][-1]),
        'best_val_auc': float(max(history.history['val_auc'])),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'lstm_units': args.lstm_units,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate,
        'balanced_classes': args.balance_classes,
        'augmentation': args.augment,
        'timestamp': datetime.now().isoformat()
    }
    
    save_model_with_metadata(model, model_path, metadata)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Model saved: {model_path}")
    print(f"Best validation AUC: {metadata['best_val_auc']:.4f}")
    print(f"Training samples: {metadata['training_samples']}")
    print(f"Validation samples: {metadata['validation_samples']}")
    print(f"Test samples: {metadata['test_samples']}")
    print(f"Model architecture: {args.model_type.upper()}")
    print(f"LSTM units: {args.lstm_units}, Layers: {args.num_layers}")
    print(f"Class balancing: {args.balance_classes}")
    print(f"Data augmentation: {args.augment}")
    print("\nNext steps:")
    print("1. Test the model: python -c \"from models import load_model; model = load_model('models/lstm_model_*.h5')\"")
    print("2. Run live monitoring: streamlit run app.py")


if __name__ == "__main__":
    main()
