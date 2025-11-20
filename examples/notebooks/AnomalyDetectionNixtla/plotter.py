import matplotlib.pyplot as plt
import numpy as np
from constants import *
import pandas as pd


def plot_normal_and_anomalous_signal(signal, anomaly_signal, image_path=IMAGE_FOLDER):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(anomaly_signal, color="darkorange", label="Anomalous Signal")
    plt.plot(signal, color="navy", label="Non anomalous Signal")
    plt.xlabel("Time (t)")
    plt.ylabel("Temperature (y)")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(np.abs(anomaly_signal - signal), color="k")
    plt.xlabel("Time (t)")
    plt.ylabel("|Signal - Anomalous Signal|")
    plt.tight_layout()
    # plt.savefig(image_path + 'normal_vs_anomalous_signal.png')


def plot_timegpt_anomalies(df, ds_anomaly):
    # Ensure datetime is parsed
    df["ds"] = pd.to_datetime(df["ds"])
    x, y = np.array(df["ds"]), np.array(df["y"])
    timegpt_y = np.array(df["TimeGPT"])
    timegpt_y_low = np.array(df["TimeGPT-lo-99"])
    timegpt_y_high = np.array(df["TimeGPT-hi-99"])
    plt.figure(figsize=(10, 5))

    # Plot actual values
    plt.plot(x, y, label="Actual", color="black", linewidth=1.5)

    # Plot TimeGPT predictions
    plt.plot(x, timegpt_y, label="TimeGPT Prediction", color="blue", linestyle="--")

    # Confidence interval using fill_between
    plt.fill_between(
        x,
        timegpt_y_low,
        timegpt_y_high,
        color="blue",
        alpha=0.2,
        label="99% Confidence Interval",
    )

    # Highlight anomalies
    match = df[df["ds"] == pd.to_datetime(ds_anomaly)]
    if not match.empty and match.iloc[0]["anomaly"]:
        plt.scatter(
            match["ds"],
            match["y"],
            color="red",
            label="Anomaly Detected",
            zorder=5,
            s=30,
        )

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Time Series with TimeGPT Predictions and Anomalies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
