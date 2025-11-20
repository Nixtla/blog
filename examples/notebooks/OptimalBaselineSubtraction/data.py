import numpy as np 
import pandas as pd 
from constants import * 
from utils import * 
from plotter import * 
import warnings

class TimeSeriesData():
    def __init__(self, data_folder = DEFAULT_DATA_PATH):
        warnings.simplefilter("ignore")  # Ignore all warnings
        self.city_attribute_data = pd.read_csv(data_folder+'/city_attributes.csv')
        self.temperature_data = pd.read_csv(data_folder +'/temperature.csv')
        self.preprocess_temperature_data()
        self.build_temperature_dict_data()


    def select_city(self, city):
        self.selected_city_data = self.temperature_data[['datetime',city]].copy()
        self.selected_city_data = self.selected_city_data.reset_index().drop('index',axis=1)
        self.selected_city_data  = standardize_data(self.selected_city_data,city)
        self.selected_city_data = enhance_data(self.selected_city_data)
        return self.selected_city_data

    
    def preprocess_temperature_data(self):
        self.temperature_data = self.temperature_data.dropna().reset_index().drop('index',axis=1)
        self.temperature_data['datetime'] = pd.to_datetime(self.temperature_data['datetime'])

    
    def build_temperature_dict_data(self):
        city_list = set(self.temperature_data.columns.tolist())
        city_list.remove('datetime')
        self.temperature_dict_data = {}
        for city in city_list:
            self.select_city(city)
            self.temperature_dict_data[city] = self.selected_city_data

    
    def isolate_city_and_time(self, city, segment_class = 'day', segment_idx = 1):
        city_data = self.temperature_dict_data[city]
        self.target_data = city_data[city_data[segment_class] == segment_idx]
        self.bank_of_data = city_data[city_data[segment_class] != segment_idx]
        self.list_of_candidates = split_segments(self.bank_of_data, city = city, segment_class = segment_class)
        self.city = city
        self.segment_class = segment_class
        self.segment_idx = segment_idx


    def find_optimal_baseline(self):
        self.optimal_baseline = None
        self.min_mae_error = float('inf')
        target_curve = np.array(self.target_data[self.city]).reshape(-1,1)[:,0]
        self.optimal_difference = None
        for candidate in self.list_of_candidates:
            diff = np.abs(candidate - target_curve)
            error = np.mean(diff,axis=0)
            if error < self.min_mae_error:
                self.min_mae_error = error
                self.optimal_baseline = candidate
                self.optimal_difference = diff
        self.optimal_baseline_data = {'optimal_baseline_curve' : self.optimal_baseline, 'optimal_baseline_diff': self.optimal_difference,
                                       'optimal_baseline_error': self.min_mae_error, 'target_curve': target_curve}
        return self.optimal_baseline_data
    

    def run_anomaly_detection(self, threshold, plot = True):
        self.threshold = threshold
        scaled_signal = self.optimal_baseline_data['optimal_baseline_diff']/self.optimal_baseline_data['target_curve'].max()
        fixed_target = np.where(scaled_signal > threshold, self.optimal_baseline_data['optimal_baseline_curve'], self.optimal_baseline_data['target_curve'])
        self.anomaly_data = {'time':np.arange(0,len(scaled_signal)), 'residual': scaled_signal, 'mask': scaled_signal > threshold, 'target_replaced_anomaly': fixed_target}
        if plot:
            self.plot_anomaly_detection_result()
        return self.anomaly_data


    def plot_target_and_baseline(self):
        target_vs_bank_of_candidates_plotter(target_data = self.target_data, bank_of_data = self.bank_of_data, city = self.city,
                                    segment_class = self.segment_class, segment_idx = self.segment_idx)
        

    def plot_target_and_candidates(self):
        target_vs_single_candidate_plotter(list_of_candidates = self.list_of_candidates, target_data = self.target_data, 
                                       city = self.city, segment_class = self.segment_class, segment_idx = self.segment_idx)
        
    
    def plot_target_and_optimal_baseline(self):
        target_vs_optimal_baseline_plotter(optimal_baseline_data = self.optimal_baseline_data, city = self.city,
                                           segment_class = self.segment_class, segment_idx = self.segment_idx)


    def plot_target_and_optimal_baseline(self):
        target_vs_optimal_baseline_plotter(optimal_baseline_data = self.optimal_baseline_data, city = self.city,
                                           segment_class = self.segment_class, segment_idx = self.segment_idx)


    def plot_anomaly_detection_result(self):
        anomaly_detection_plotter(anomaly_data = self.anomaly_data, threshold = self.threshold, 
                                  city = self.city, segment_class = self.segment_class, 
                                  segment_idx = self.segment_idx)



