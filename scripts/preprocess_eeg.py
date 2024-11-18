import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter

def bandpass_filter(data, lowcut, highcut, fs):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(5, [low, high], btype='band')
    return lfilter(b, a, data)

def standardize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def preprocess_eeg(input_path, output_path):
    eeg_data = pd.read_csv(input_path)
    filtered = eeg_data.apply(lambda x: bandpass_filter(x, 0.5, 30, 256), axis=0)
    standardized = standardize(filtered.values)
    np.save(output_path, standardized)
