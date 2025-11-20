import numpy as np 
from utils import * 
from plotter import * 
from constants import * 
import os
import pandas as pd
from dotenv import load_dotenv
from nixtla import NixtlaClient
import os
import logging


class AnomalyCalibrator:
    def __init__(self, processed_data, largest_threshold_size = 0.1, step_threshold = 0.01):
        self.input_data = processed_data
        self.input_signal = np.array(processed_data['y']).copy()
        self.n = len(self.input_signal)
        self.curr_threshold = largest_threshold_size
        self.step_threshold = step_threshold
        self.all_possible_locations = None

    def inject_anomaly(self, location=None, threshold=None, window=50):
        self.anomalous_signal = add_anomaly(
            signal=self.input_signal,
            location=location,
            threshold=threshold,
            window=window
        )
        self.location_anomaly = location
        return {'anomalous_signal': self.anomalous_signal, 'normal_signal': self.input_signal}


    def load_nixtla_client(self, nixtla_client = None):
        if nixtla_client is None:
            load_dotenv()  # looks for .env in current directory
            api_key = os.environ["NIXTLA_API_KEY"]
            nixtla_client = NixtlaClient(api_key=api_key)
        self.nixtla_client = nixtla_client


    def build_possible_locations(self,
                                 min_location = DEFAULT_MIN_LOCATION, max_location = DEFAULT_MAX_LOCATION, 
                                 num_signals = DEFAULT_NUM_LOCATION):
        if self.all_possible_locations is None:
            self.all_possible_locations = np.linspace(int(min_location*self.n), int(max_location*self.n), DEFAULT_DENSE_LOCATION)
        self.possible_locations = np.random.choice(self.all_possible_locations, size = num_signals).astype(int)
        

    def build_anomalous_dataset(self, min_location = DEFAULT_MIN_LOCATION, max_location = DEFAULT_MAX_LOCATION,
                                num_location = DEFAULT_NUM_LOCATION):
        self.build_possible_locations(min_location, max_location, num_location)
        self.anomaly_dataset = np.zeros((len(self.possible_locations),self.n))
        i = 0
        self.anomaly_dict = {}
        for location in self.possible_locations:
            anomalous_signal = self.inject_anomaly(location = location, threshold = self.curr_threshold)['anomalous_signal']
            self.anomaly_dict[i] = {}
            self.anomaly_dict[i]['signal'] = anomalous_signal
            self.anomaly_dict[i]['location'] = location
            self.anomaly_dataset[i] = anomalous_signal
            i += 1 


    def run_anomaly_detection(self,anomalous_signal, freq = 'H'):
        self.anomaly_data = self.input_data.copy()
        self.anomaly_data['y'] = anomalous_signal
        self.anomaly_result = self.nixtla_client.detect_anomalies(self.anomaly_data,
            freq=freq,
            model="timegpt-1",
        )
        return self.anomaly_result
    

    def calibration_run(self, number_of_runs = 5, print_statement = True):
        logger = logging.getLogger("nixtla.nixtla_client")
        previous_level = logger.level
        logger.setLevel(logging.WARNING)  # Suppress INFO logs temporarily
        self.build_anomalous_dataset(num_location = number_of_runs)
        count_anomalies = 0
        for key in self.anomaly_dict:
            anomalous_signal = self.anomaly_dict[key]['signal']
            location = self.anomaly_dict[key]['location']
            dt_location = self.input_data.loc[location]['ds']
            res = self.run_anomaly_detection(anomalous_signal)
            anomaly_bool = np.array(res[res['ds'] == dt_location]['anomaly'])[0].copy()
            if anomaly_bool == True:
                count_anomalies += 1 
                if print_statement  == True:
                    str_output = f'Running TimeGPT for signal number {key+1}, with location {location}, and size {self.curr_threshold}'
                    print(str_output)
                    print('The anomaly has been detected')
        logger.setLevel(previous_level)  # Restore original level
        accuracy = count_anomalies/len(self.anomaly_dict)
        print(f'The accuracy for size {self.curr_threshold} is {accuracy}')
        return accuracy
    

    def calibration_loop(self, number_of_runs = 10, desired_accuracy = 0.5):
        logger = logging.getLogger("nixtla.nixtla_client")
        previous_level = logger.level
        logger.setLevel(logging.WARNING)  # Suppress INFO logs temporarily
        accuracy = None
        self.accuracy_dict = {}
        while accuracy is None or accuracy > desired_accuracy:
            accuracy = self.calibration_run(number_of_runs = number_of_runs, print_statement = False)
            self.accuracy_dict[self.curr_threshold] = accuracy
            if accuracy < desired_accuracy:
                print(f'The accuracy for size {self.curr_threshold} is {accuracy}, which is smaller than the desired accuracy {desired_accuracy}')
                print(f'The minimum detectable anomaly is {self.curr_threshold+self.step_threshold}')
            self.curr_threshold -= self.step_threshold
            self.curr_threshold = np.round(self.curr_threshold,3)
            if self.curr_threshold <= 0:
                break
        return self.accuracy_dict
            

    def plot_anomalous_dataset(self):
        plt.figure(figsize = (10,5))
        for i in range(len(self.anomaly_dataset)):
            plt.plot(self.anomaly_dataset[i])
        plt.xlabel('Time (h)')
        plt.ylabel('Temperature (K)')
        plt.show()


    def plot_anomaly_detection(self,location = None):
        if location is None:
            location = self.location_anomaly
        ds_anomaly = self.input_data['ds'].loc[location]
        plot_timegpt_anomalies(self.anomaly_result, ds_anomaly)
    


