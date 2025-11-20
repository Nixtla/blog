import numpy as np 
import pandas as pd 
from constants import * 
from utils import * 

class Data:
    def __init__(self, data_path=DATA_PATH, datetime_column = None):
        self.data_path = data_path
        self.datetime_column = datetime_column
        self.raw_data = pandas_loader(self.data_path, datetime_column = self.datetime_column)
        self.processed_data = None
        self.processed_data_info = {'city': None, 'start': None, 'end':None}


    def isolate_city(self, city = DEFAULT_CITY):
        if self.processed_data is None:
            self.processed_data = self.raw_data
        self.processed_data = self.processed_data[[city, self.datetime_column]]
        self.processed_data_city = city
        self.processed_data_info['city'] = city
        return self.processed_data


    def isolate_portion(self, start_portion = START_PORTION, end_portion = END_PORTION):
        if self.processed_data is None:
            self.processed_data = self.raw_data
        self.processed_data = self.processed_data.loc[start_portion: end_portion].reset_index(drop=True)
        self.processed_data_info['start'] = start_portion
        self.processed_data_info['end'] = end_portion
        return self.processed_data

    def prepare_for_nixtla(self):
        self.processed_data['unique_id'] = 0
        self.processed_data = self.processed_data.rename(columns={self.processed_data_city : 'y', self.datetime_column: 'ds'})
        return self.processed_data

