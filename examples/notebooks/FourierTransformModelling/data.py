import numpy as np 
import matplotlib.pyplot 
from constants import * 
from plotter import * 

class Data():
    def __init__(self, x_min = X_MIN, x_max = X_MAX, ts_length = TIME_SERIES_LENGTH):
        self.x = np.linspace(x_min, x_max, ts_length)
        self.y = np.zeros(ts_length)

    
    def plot_time_series(self):
        timeseries_plotter(self.x,self.y)
