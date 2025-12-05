"""
FILE 1: filter_eeg.py
This file filters EEG signals to remove noise
"""

import numpy as np
from scipy import signal


def bandpass_filter(data, fs=250, low_freq=1.0, high_freq=40.0):
    """
    Remove noise from EEG signal
    
    Parameters:
    - data: numpy array [samples, channels]
    - fs: sampling rate (250 Hz)
    - low_freq: remove below 1 Hz (slow drift)
    - high_freq: remove above 40 Hz (noise)
    
    Returns:
    - filtered data
    """
    print(f"  Filtering signal: {low_freq}-{high_freq} Hz...")
    
    # Calculate normalized frequencies
    nyquist = fs / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Design Butterworth bandpass filter
    sos = signal.butter(5, [low, high], btype='band', output='sos')
    
    # Apply filter to each channel
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = signal.sosfiltfilt(sos, data[:, ch])
    
    return filtered


def create_windows(data, window_size=500, stride=250):
    """
    Split long signal into 2-second windows
    
    Parameters:
    - data: filtered signal [samples, channels]
    - window_size: 500 samples = 2 seconds at 250 Hz
    - stride: 250 samples = 50% overlap
    
    Returns:
    - windows: array [num_windows, 500, 2]
    """
    n_samples = data.shape[0]
    n_windows = (n_samples - window_size) // stride + 1
    
    windows = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = data[start:end, :]
        windows.append(window)
    
    return np.array(windows)


if __name__ == "__main__":
    print("This file contains filtering functions")
    print("It will be called by main.py")
