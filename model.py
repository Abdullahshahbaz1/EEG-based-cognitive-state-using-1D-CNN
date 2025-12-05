"""
FILE 3: model.py
This file defines the CNN model for classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_cnn_model():
    """
    Create 1D-CNN model for binary classification
    
    Input: [500 samples, 2 channels] (2 seconds of data)
    Output: 0 (LOW) or 1 (HIGH)
    """
    print("\nBuilding CNN model...")
    
    model = keras.Sequential([
        # Input layer: [500, 2]
        layers.Input(shape=(500, 2)),
        
        # Conv Block 1
        layers.Conv1D(32, kernel_size=7, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Conv Block 2
        layers.Conv1D(64, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Conv Block 3
        layers.Conv1D(64, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        
        # Output: binary classification
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model created!")
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the model
    
    Returns:
    - trained model
    - training history
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


if __name__ == "__main__":
    print("This file contains the CNN model")
    print("It will be called by main.py")
