import numpy as np 
import pandas as pd 
from constants import *

def filter_index(data, column = Y_COLUMN):
    y = np.array(data[column])
    index = []
    count = 0
    for i in y:
        try:
            float(i)
            index.append(count)
            count += 1
        except:
            count += 1
    return index


def aggregate_data(data, key='year', column = Y_COLUMN):
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'])  # ensure proper dtype
    if key == 'year':
        data['time_group'] = data['Date'].dt.to_period('Y').astype(str)
    elif key == 'month':
        data['time_group'] = data['Date'].dt.to_period('M').astype(str)
    elif key == 'week':
        data['time_group'] = data['Date'].dt.to_period('W').astype(str)
    elif key == 'day':
        data['time_group'] = data['Date'].dt.to_period('D').astype(str)
    else:
        raise ValueError("Unsupported key. Choose from: 'year', 'month', 'day', 'hour', 'week'.")

    return data.groupby('time_group').mean(numeric_only=True).reset_index()[['time_group',column]]