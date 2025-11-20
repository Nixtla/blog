import matplotlib.pyplot as plt 
from constants import * 
import numpy as np
import pandas as pd

def set_dark_mode():
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#121212'
    plt.rcParams['figure.facecolor'] = '#121212'
    plt.rcParams['savefig.facecolor'] = '#121212'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['legend.edgecolor'] = 'white'
    plt.rcParams['axes.grid'] = True  
    plt.rcParams['grid.alpha'] = 0.4


def plot_timeseries(df, date_column = DATE_COLUMN, y_column = Y_COLUMN, title = None):
    set_dark_mode()
    x = np.array(pd.to_datetime(df[date_column]))
    y = np.array(df[y_column]).astype('float')
    plt.figure(figsize = IMAGE_FIGSIZE)
    if title:
        plt.title(title, fontsize = 12)
    plt.plot(x, y, color = 'cyan')
    plt.xlabel('Date (t)', fontsize = 12)
    plt.ylabel(f'{y_column}-Euro Conversion', fontsize = 12)
    plt.savefig(IMAGE_PATH + 'raw_timeseries_plot.svg')