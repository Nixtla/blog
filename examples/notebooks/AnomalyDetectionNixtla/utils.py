import pandas as pd
import numpy as np 
from constants import * 

def pandas_loader(data_path, datetime_column = None):
    if datetime_column is None:
        return pd.read_csv(data_path)
    else:
        return pd.read_csv(data_path, parse_dates= [datetime_column])
    

def add_anomaly(signal, location = None, threshold = None, window = None):
    signal = np.array(signal).copy()
    if window is None:
        window = DEFAULT_WINDOW
    if location is None:
        location = np.random.choice(len(signal))
    max_value = identify_mean(signal = signal, location = location, window = window)
    if threshold is None:
        threshold_list = np.arange(-0.1,0.1,0.001)
        threshold_list = threshold_list[threshold_list != 0]
        threshold = np.random.choice(threshold_list)
    signal[location] += threshold * max_value
    return signal



def identify_mean(signal, location, window):
    left = max(0, location - window)
    right = min(len(signal), location + window)
    return np.mean(signal[left:right])


