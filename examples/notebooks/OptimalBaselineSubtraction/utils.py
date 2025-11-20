import numpy as np 
import pandas as pd 
from constants import * 

def standardize_data(df,city):
    df = df.copy()
    df.set_index('datetime', inplace=True)
    start_date = df.index.min().normalize()
    end_date = df.index.max().normalize() + pd.Timedelta(days=1)
    full_range = pd.date_range(start=start_date, end=end_date - pd.Timedelta(hours=1), freq='H')
    df_standardized = df.reindex(full_range)
    df_standardized.index.name = 'datetime'
    df_standardized[city] = df_standardized[city].interpolate()  # or use fillna()
    df_standardized = df_standardized.fillna(method = 'bfill').reset_index()
    return df_standardized


def enhance_data(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['day'] = (df['datetime'].dt.floor('D') - df['datetime'].dt.floor('D').min()).dt.days + 1
    df['week'] = ((df['day'] - 1) // 7) + 1
    df['month'] = ((df['datetime'].dt.year - df['datetime'].dt.year.min()) * 12 + df['datetime'].dt.month -
                df['datetime'].dt.month.min()) + 1
    df['year'] = df['datetime'].dt.year - df['datetime'].dt.year.min() + 1
    return df


def split_segments(bank_of_data, city, segment_class = 'day'):
    list_of_segments = np.array([np.array(group[city]).reshape(-1,1)[:,0] for _, group in bank_of_data.groupby(segment_class)])
    return list_of_segments
