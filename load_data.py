"""
FILE 2: load_data.py
This file loads .txt files from data/ folder
"""

import os
import pandas as pd
import numpy as np


def load_txt_file(filepath):
    """
    Load one .txt file
    
    Returns:
    - data: numpy array [samples, 8 channels]
    """
    print(f"\nLoading: {os.path.basename(filepath)}")
    
    # Read file, skip lines starting with %
    df = pd.read_csv(filepath, comment='%', skipinitialspace=True)
    
    # Get the 8 EXG channels
    channels = []
    for i in range(8):
        col_name = f'EXG Channel {i}'
        channels.append(df[col_name].values)
    
    # Convert to numpy array [samples, 8]
    data = np.array(channels).T.astype(np.float64)
    
    print(f"  Shape: {data.shape} (samples, channels)")
    
    return data


def get_label_from_filename(filename):
    """
    Get label from filename
    high_cognitive_left.txt -> 1 (HIGH)
    low_cognitive_left.txt -> 0 (LOW)
    """
    if 'high' in filename.lower():
        return 1  # HIGH cognitive
    elif 'low' in filename.lower():
        return 0  # LOW cognitive
    else:
        raise ValueError(f"Cannot find 'high' or 'low' in filename: {filename}")


def load_all_files(data_folder='data'):
    """
    Load all .txt files from data/ folder
    
    Returns:
    - all_data: list of numpy arrays
    - all_labels: list of labels (0 or 1)
    - filenames: list of filenames
    """
    if not os.path.exists(data_folder):
        raise ValueError(f"Folder '{data_folder}' does not exist! Please create it and add your .txt files")
    
    # Find all .txt files
    txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
    
    if len(txt_files) == 0:
        raise ValueError(f"No .txt files found in '{data_folder}' folder!")
    
    print(f"\nFound {len(txt_files)} files in '{data_folder}/':")
    for f in txt_files:
        print(f"  - {f}")
    
    # Load each file
    all_data = []
    all_labels = []
    filenames = []
    
    for txt_file in txt_files:
        filepath = os.path.join(data_folder, txt_file)
        
        # Load data
        data = load_txt_file(filepath)
        
        # Get label
        label = get_label_from_filename(txt_file)
        label_name = "HIGH" if label == 1 else "LOW"
        print(f"  Label: {label_name}")
        
        all_data.append(data)
        all_labels.append(label)
        filenames.append(txt_file)
    
    return all_data, all_labels, filenames


if __name__ == "__main__":
    print("This file contains data loading functions")
    print("It will be called by main.py")
