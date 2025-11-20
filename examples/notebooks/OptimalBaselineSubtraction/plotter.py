import matplotlib.pyplot as plt
import numpy as np


def target_vs_bank_of_candidates_plotter(target_data, bank_of_data, city, segment_class, segment_idx):
    x_target = np.array(target_data['datetime'])  # or .values
    y_target = np.array(target_data[city]).reshape(-1,1)[:,0]
    plt.subplot(2,1,1)
    plt.title(f'Target Segment, City = {city}, Segment Class = {segment_class}, Segment Index = {segment_idx}', fontsize = 12)
    plt.plot(x_target,y_target, color='darkorange')
    plt.ylabel('Temperature (K)')
    plt.xticks(rotation=45)
    x_candidates = np.array(bank_of_data['datetime'])  # or .values
    y_candidates = np.array(bank_of_data[city]).reshape(-1,1)[:,0]  # or .values
    plt.subplot(2,1,2)
    plt.title('Remaining part of the time series', fontsize = 20)
    plt.plot(x_candidates,y_candidates, color='navy')
    plt.xticks(rotation=45)
    plt.ylabel('Temperature (K)')
    plt.tight_layout()
    plt.savefig(f'images/target_and_bank_city_{city}_segment_class_{segment_class}_idx_{segment_idx}.png')


def target_vs_single_candidate_plotter(list_of_candidates, target_data, city, segment_class, segment_idx, display_row = 2, display_column = 5):
    idx_random = np.arange(1,len(list_of_candidates))
    np.random.shuffle(idx_random)
    shuffled_candidates = list_of_candidates[idx_random]
    count_plot = 1 
    max_count_plot = display_row*display_column
    y_target = np.array(target_data[city]).reshape(-1,1)[:,0]
    x_target = np.arange(0,len(y_target),1)
    plt.figure(figsize = (16,8))
    while count_plot <= max_count_plot:
        plt.subplot(display_row,display_column,count_plot)
        plt.plot(x_target, y_target, color = 'darkorange', label = 'Target Time Series')
        plt.plot(x_target, shuffled_candidates[count_plot-1], color = 'navy', label = 'Possible Candidate')
        plt.xlabel('Time (hour)')
        plt.ylabel('Temperature (K)')
        plt.legend()
        count_plot += 1
    plt.tight_layout()
    plt.savefig(f'images/target_and_candidates_{city}_segment_class_{segment_class}_idx_{segment_idx}.png')


def target_vs_optimal_baseline_plotter(optimal_baseline_data, city, segment_class, segment_idx):
    target_curve = optimal_baseline_data['target_curve']
    optimal_baseline = optimal_baseline_data['optimal_baseline_curve']
    optimal_baseline_diff = optimal_baseline_data['optimal_baseline_diff']/target_curve.max()
    x_target = np.arange(0,len(target_curve),1)
    plt.figure(figsize = (10,6))
    plt.subplot(2,1,1)
    plt.title('Optimal baseline and target')
    plt.plot(x_target, target_curve, color = 'darkorange', label = 'Target Time Series')
    plt.plot(x_target, optimal_baseline, color = 'navy', label = 'Optimal Baseline')
    plt.xlabel('Time (hour)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.subplot(2,1,2)
    plt.title('Scaled optimal baseline difference')
    plt.plot(x_target, optimal_baseline_diff, color = 'k', label = 'Optimal Baseline')
    plt.xlabel('Time (hour)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'images/target_vs_optimal_baseline_{city}_segment_class_{segment_class}_idx_{segment_idx}.png')


def anomaly_detection_plotter(anomaly_data, threshold, city, segment_class, segment_idx):
    plt.figure(figsize = (8,4))
    plt.plot(anomaly_data['time'], anomaly_data['residual'], color='k', linewidth=2, label='Residual')
    plt.plot(anomaly_data['time'][anomaly_data['mask']], anomaly_data['residual'][anomaly_data['mask']], 'o', color='red', label='Anomaly')
    plt.axhline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold = {threshold:.3f}')
    plt.xlabel('Time (hour)')
    plt.ylabel('Scaled Residual (K)')
    plt.legend()
    plt.savefig(f'images/anomaly_detection_{city}_segment_class_{segment_class}_idx_{segment_idx}.png')
    plt.tight_layout()





