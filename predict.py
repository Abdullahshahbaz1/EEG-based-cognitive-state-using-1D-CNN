"""
FILE 5: predict.py
This file makes predictions on new .txt files

Usage:
    python predict.py data/your_file.txt
"""

import sys
import os
import numpy as np
import tensorflow as tf

# Import our functions
from load_data import load_txt_file
from filter_eeg import bandpass_filter, create_windows


def predict_file(filepath, model):
    """
    Predict cognitive state for one file
    """
    print(f"\nAnalyzing: {os.path.basename(filepath)}")
    
    # Load data
    data = load_txt_file(filepath)
    
    # Select channels 0 and 1
    data_selected = data[:, [0, 1]]
    print(f"  Using channels: 0, 1")
    
    # Filter
    print("  Filtering signal...")
    filtered = bandpass_filter(data_selected, fs=250)
    
    # Create windows
    windows = create_windows(filtered)
    print(f"  Created {len(windows)} windows")
    
    # Predict
    print("  Making predictions...")
    predictions = model.predict(windows, verbose=0)
    pred_classes = (predictions > 0.5).astype(int).flatten()
    
    # Calculate statistics
    high_count = np.sum(pred_classes == 1)
    low_count = np.sum(pred_classes == 0)
    high_ratio = high_count / len(pred_classes)
    
    # Overall prediction (majority vote)
    if high_ratio > 0.5:
        overall = "HIGH"
        confidence = high_ratio * 100
    else:
        overall = "LOW"
        confidence = (1 - high_ratio) * 100
    
    return {
        'overall': overall,
        'confidence': confidence,
        'high_count': high_count,
        'low_count': low_count,
        'total_windows': len(pred_classes)
    }


def main():
    print("="*60)
    print("EEG BINARY CLASSIFIER - PREDICTION")
    print("="*60)
    
    # Check if file path provided
    if len(sys.argv) < 2:
        print("\nUsage: python predict.py <path_to_txt_file>")
        print("\nExample:")
        print("  python predict.py data/test_file.txt")
        return
    
    filepath = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"\nERROR: File not found: {filepath}")
        return
    
    # Check if model exists
    if not os.path.exists('trained_model.h5'):
        print("\nERROR: trained_model.h5 not found!")
        print("Please run 'python main.py' first to train the model")
        return
    
    # Load model
    print("\nLoading trained model...")
    model = tf.keras.models.load_model('trained_model.h5')
    print("✓ Model loaded")
    
    # Make prediction
    result = predict_file(filepath, model)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nOverall State: {result['overall']}")
    print(f"Confidence:    {result['confidence']:.1f}%")
    print()
    print("Breakdown:")
    print(f"  HIGH: {result['high_count']} windows ({result['high_count']/result['total_windows']*100:.1f}%)")
    print(f"  LOW:  {result['low_count']} windows ({result['low_count']/result['total_windows']*100:.1f}%)")
    print(f"  Total: {result['total_windows']} windows")
    print()
    
    # Interpretation
    if result['confidence'] > 80:
        print("→ Very confident prediction")
    elif result['confidence'] > 60:
        print("→ Confident prediction")
    else:
        print("→ Uncertain prediction (borderline case)")


if __name__ == "__main__":
    main()
