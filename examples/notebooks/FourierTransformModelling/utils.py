import numpy as np 
def filter_freq(y_fft, start, end):
    y_fft_filtered = np.zeros_like(y_fft)
    y_fft_filtered[start*100:end*100] = y_fft[start*100:end*100]
    return y_fft_filtered