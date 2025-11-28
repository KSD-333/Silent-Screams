"""
Model Architectures Module
---------------------------
Provides neural network architectures for sequence classification of pose keypoints.
Includes LSTM and Transformer-based models for temporal pattern recognition.

Expected Input Shape: (batch_size, window_length, num_features)
- window_length: Number of frames in sequence (e.g., 30 for 2 seconds @ 15fps)
- num_features: Flattened keypoints (e.g., 66 for 33 landmarks × 2 coords)

Output: Binary classification (0=normal, 1=abnormal) with probability
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Optional, Tuple


def build_lstm_model(window_length: int = 30,
                     num_features: int = 66,
                     lstm_units: int = 128,
                     num_lstm_layers: int = 2,
                     dropout_rate: float = 0.3,
                     dense_units: int = 64) -> keras.Model:
    """
    Build a Bidirectional LSTM model for sequence classification.
    
    Architecture:
    - Input: (window_length, num_features)
    - Bi-LSTM layers with dropout
    - Dense layers with dropout
    - Output: Binary classification (sigmoid)
    
    Args:
        window_length: Number of frames per sequence
        num_features: Number of features per frame
        lstm_units: Number of units in each LSTM layer
        num_lstm_layers: Number of stacked LSTM layers
        dropout_rate: Dropout rate for regularization
        dense_units: Number of units in dense layer
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(window_length, num_features), name='input_sequence')
    
    x = inputs
    
    # Stacked Bi-LSTM layers
    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1)  # Last layer doesn't return sequences
        
        x = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate * 0.5,
                name=f'bilstm_{i+1}'
            )
        )(x)
        
        if return_sequences:
            x = layers.Dropout(dropout_rate, name=f'dropout_lstm_{i+1}')(x)
    
    # Dense layers
    x = layers.Dense(dense_units, activation='relu', name='dense_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_dense')(x)
    
    # Output layer (binary classification)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='BiLSTM_Classifier')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def build_transformer_model(window_length: int = 30,
                           num_features: int = 66,
                           num_heads: int = 4,
                           num_transformer_blocks: int = 2,
                           ff_dim: int = 128,
                           dropout_rate: float = 0.3) -> keras.Model:
    """
    Build a Transformer encoder model for sequence classification.
    
    Architecture:
    - Input: (window_length, num_features)
    - Positional encoding
    - Multi-head self-attention blocks
    - Feed-forward network
    - Global average pooling
    - Output: Binary classification (sigmoid)
    
    Args:
        window_length: Number of frames per sequence
        num_features: Number of features per frame
        num_heads: Number of attention heads
        num_transformer_blocks: Number of transformer encoder blocks
        ff_dim: Dimension of feed-forward network
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(window_length, num_features), name='input_sequence')
    
    # Positional encoding
    positions = tf.range(start=0, limit=window_length, delta=1)
    position_embedding = layers.Embedding(
        input_dim=window_length,
        output_dim=num_features,
        name='position_embedding'
    )(positions)
    
    x = inputs + position_embedding
    
    # Transformer encoder blocks
    for i in range(num_transformer_blocks):
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=num_features // num_heads,
            dropout=dropout_rate,
            name=f'attention_{i+1}'
        )(x, x)
        
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f'norm_1_{i+1}')(x + attention_output)
        
        # Feed-forward network
        ff_output = layers.Dense(ff_dim, activation='relu', name=f'ff_1_{i+1}')(x1)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        ff_output = layers.Dense(num_features, name=f'ff_2_{i+1}')(ff_output)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        
        x = layers.LayerNormalization(epsilon=1e-6, name=f'norm_2_{i+1}')(x1 + ff_output)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu', name='dense_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_dense')(x)
    
    # Output layer (binary classification)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Transformer_Classifier')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def load_model(model_path: str) -> keras.Model:
    """
    Load a trained Keras model from file.
    
    Supports both .h5 and SavedModel formats.
    
    Args:
        model_path: Path to model file (.h5) or directory (SavedModel)
    
    Returns:
        Loaded Keras model
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model format is not supported
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    try:
        # Try loading as Keras model
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
    
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {str(e)}")


def create_mock_model(window_length: int = 30,
                     num_features: int = 66,
                     save_path: Optional[str] = None) -> keras.Model:
    """
    Create a mock/demo model for testing without training data.
    
    This generates a simple LSTM model with random weights for demonstration purposes.
    The model will produce random (but deterministic) predictions.
    
    Args:
        window_length: Number of frames per sequence
        num_features: Number of features per frame
        save_path: Optional path to save the mock model
    
    Returns:
        Mock Keras model
    """
    print("Creating mock model for demo/testing...")
    
    # Build a simple LSTM model
    model = build_lstm_model(
        window_length=window_length,
        num_features=num_features,
        lstm_units=64,
        num_lstm_layers=1,
        dropout_rate=0.2,
        dense_units=32
    )
    
    # Initialize with random weights (already done by default)
    # Set a seed for reproducibility
    tf.random.set_seed(42)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Mock model created with input shape: {model.input_shape}")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Mock model saved to {save_path}")
    
    return model


def get_model_summary(model: keras.Model) -> str:
    """
    Get a string summary of the model architecture.
    
    Args:
        model: Keras model
    
    Returns:
        String summary
    """
    from io import StringIO
    
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary = stream.getvalue()
    stream.close()
    
    return summary


def predict_sequence(model: keras.Model,
                    sequence: np.ndarray,
                    threshold: float = 0.5) -> Tuple[int, float]:
    """
    Predict class for a single sequence.
    
    Args:
        model: Trained Keras model
        sequence: Input sequence (window_length, num_features)
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        predicted_class: 0 (normal) or 1 (abnormal)
        probability: Probability of abnormal class
    """
    # Add batch dimension if needed
    if len(sequence.shape) == 2:
        sequence = np.expand_dims(sequence, axis=0)
    
    # Predict
    probability = model.predict(sequence, verbose=0)[0, 0]
    predicted_class = 1 if probability >= threshold else 0
    
    return predicted_class, float(probability)


def predict_batch(model: keras.Model,
                 sequences: np.ndarray,
                 threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict classes for a batch of sequences.
    
    Args:
        model: Trained Keras model
        sequences: Input sequences (batch_size, window_length, num_features)
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        predicted_classes: Array of predictions (batch_size,)
        probabilities: Array of probabilities (batch_size,)
    """
    # Predict
    probabilities = model.predict(sequences, verbose=0)[:, 0]
    predicted_classes = (probabilities >= threshold).astype(int)
    
    return predicted_classes, probabilities


def save_model_with_metadata(model: keras.Model,
                            save_path: str,
                            metadata: dict):
    """
    Save model with metadata (training info, hyperparameters, etc.).
    
    Args:
        model: Keras model to save
        save_path: Path to save model
        metadata: Dictionary of metadata to save
    """
    import json
    
    # Save model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Save metadata
    metadata_path = save_path.replace('.h5', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")


def load_model_with_metadata(model_path: str) -> Tuple[keras.Model, dict]:
    """
    Load model with metadata.
    
    Args:
        model_path: Path to model file
    
    Returns:
        model: Loaded Keras model
        metadata: Metadata dictionary
    """
    import json
    
    # Load model
    model = load_model(model_path)
    
    # Load metadata
    metadata_path = model_path.replace('.h5', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata loaded from {metadata_path}")
    else:
        metadata = {}
        print("No metadata file found")
    
    return model, metadata


if __name__ == "__main__":
    """
    Test model architectures.
    """
    print("Testing model architectures...\n")
    
    # Test LSTM model
    print("1. Building LSTM model:")
    lstm_model = build_lstm_model(window_length=30, num_features=66)
    print(lstm_model.summary())
    print()
    
    # Test Transformer model
    print("2. Building Transformer model:")
    transformer_model = build_transformer_model(window_length=30, num_features=66)
    print(transformer_model.summary())
    print()
    
    # Test prediction
    print("3. Testing prediction:")
    dummy_sequence = np.random.rand(30, 66)
    pred_class, pred_prob = predict_sequence(lstm_model, dummy_sequence)
    print(f"   Predicted class: {pred_class}")
    print(f"   Probability: {pred_prob:.4f}")
    print()
    
    # Test batch prediction
    print("4. Testing batch prediction:")
    dummy_batch = np.random.rand(5, 30, 66)
    pred_classes, pred_probs = predict_batch(lstm_model, dummy_batch)
    print(f"   Predicted classes: {pred_classes}")
    print(f"   Probabilities: {pred_probs}")
    print()
    
    # Test mock model creation
    print("5. Creating mock model:")
    mock_model = create_mock_model(save_path='models/mock_model.h5')
    print()
    
    # Test model loading
    print("6. Loading mock model:")
    loaded_model = load_model('models/mock_model.h5')
    print()
    
    print("All tests passed!")
