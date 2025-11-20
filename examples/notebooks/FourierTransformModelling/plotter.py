import numpy as np 
import matplotlib.pyplot as plt 
from constants import * 


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


def timeseries_plotter(x,y):
    set_dark_mode()
    plt.figure(figsize=(10,5))
    plt.plot(x,y, color ='cyan', lw = 3)
    plt.xlabel('Time (t)', fontsize = 20)
    plt.ylabel('Amplitude (y)', fontsize = 20)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.grid(True, alpha = 0.4)
    plt.savefig(IMAGE_PATH + 'timeseries_plot.svg')