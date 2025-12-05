"""
MAIN FILE: main.py
This file runs the entire pipeline:
1. Load data from data/ folder
2. Filter signals
3. Create windows
4. Train model
5. Save model
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Import our custom files
from load_data import load_all_files, get_label_from_filename
from filter_eeg import bandpass_filter, create_windows
from model import create_cnn_model, train_model


def main():
    """
    Main function that runs everything
    """
    print("="*60)
    print("EEG BINARY CLASSIFIER - TRAINING PIPELINE")
    print("="*60)
    print("\nThis script will:")
    print("1. Load .txt files from data/ folder")
    print("2. Filter signals (remove noise)")
    print("3. Create 2-second windows")
    print("4. Train CNN model")
    print("5. Save trained model")
    print()
    
    # ==========================================
    # STEP 1: LOAD DATA
    # ==========================================
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    try:
        all_data, all_labels, filenames = load_all_files('data')
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you have:")
        print("  1. Created 'data/' folder")
        print("  2. Put your .txt files in 'data/' folder")
        print("  3. Named files like: high_cognitive_left.txt, low_cognitive_left.txt")
        return
    
    print(f"\nLoaded {len(all_data)} files successfully!")
    
    
    # ==========================================
    # STEP 2: FILTER & WINDOW DATA
    # ==========================================
    print("\n" + "="*60)
    print("STEP 2: FILTERING & WINDOWING")
    print("="*60)
    
    all_windows = []
    window_labels = []
    
    for i, (data, label) in enumerate(zip(all_data, all_labels)):
        print(f"\nProcessing file {i+1}/{len(all_data)}: {filenames[i]}")
        
        # Select only Channel 0 and 1 (first 2 channels)
        data_selected = data[:, [0, 1]]
        print(f"  Using channels: 0, 1")
        print(f"  Selected shape: {data_selected.shape}")
        
        # Filter the signal (remove noise)
        filtered = bandpass_filter(data_selected, fs=250, low_freq=1.0, high_freq=40.0)
        
        # Create 2-second windows
        windows = create_windows(filtered, window_size=500, stride=250)
        print(f"  Created {len(windows)} windows")
        
        # Store windows and their labels
        all_windows.append(windows)
        window_labels.extend([label] * len(windows))
    
    # Combine all windows
    X = np.vstack(all_windows)  # Shape: [total_windows, 500, 2]
    y = np.array(window_labels)  # Shape: [total_windows]
    
    print(f"\nTotal dataset:")
    print(f"  X shape: {X.shape} (windows, samples, channels)")
    print(f"  y shape: {y.shape} (labels)")
    print(f"  HIGH samples: {np.sum(y == 1)}")
    print(f"  LOW samples:  {np.sum(y == 0)}")
    
    
    # ==========================================
    # STEP 3: SPLIT DATA
    # ==========================================
    print("\n" + "="*60)
    print("STEP 3: SPLITTING DATA")
    print("="*60)
    
    # Split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train)} windows")
    print(f"Val set:   {len(X_val)} windows")
    print(f"Test set:  {len(X_test)} windows")
    
    
    # ==========================================
    # STEP 4: CREATE & TRAIN MODEL
    # ==========================================
    print("\n" + "="*60)
    print("STEP 4: CREATING MODEL")
    print("="*60)
    
    model = create_cnn_model()
    
    # Train model
    model, history = train_model(model, X_train, y_train, X_val, y_val)
    
    
    # ==========================================
    # STEP 5: EVALUATE MODEL
    # ==========================================
    print("\n" + "="*60)
    print("STEP 5: EVALUATION")
    print("="*60)
    
    # Test the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              LOW    HIGH")
    print(f"Actual LOW    {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       HIGH   {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Detailed report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['LOW', 'HIGH'],
                                digits=3))
    
    
    # ==========================================
    # STEP 6: SAVE MODEL
    # ==========================================
    print("\n" + "="*60)
    print("STEP 6: SAVING MODEL")
    print("="*60)
    
    model.save('trained_model.h5')
    print("\n✓ Model saved as: trained_model.h5")
    
    
    # ==========================================
    # DONE!
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nYou can now use predict.py to make predictions on new files")
    print("Example: python predict.py data/new_file.txt")


if __name__ == "__main__":
    main()
