---
title: Savitzky Golay Filtering for Time Series Denoising
description: Denoise your time series with Polynomial Smoothing using Saviztky-Goaly filter
image: /images/Polynomial_Filtering/Noise_Definition.svg
author_name: Piero Paialunga
author_image: "/images/authors/piero.jpg"
author_position: Data Scientist
publication_date: 2025-08-26
---

Imagine listening to a beautiful melody of a violin in the comfort of your home. You are there, relaxed, with your hot tea and you are listening to every frequency, every detail and every harmonization of that beautiful sound.

Now, imagine trying to hear the same exact violin player playing the same exact melody, but in the middle of New York City. The sound is buried under car horns, chatter, and the hum of engines. The melody is the same, but the output is completely different.

Real-world time series usually reflect the second scenario: the signal we care about is almost always tangled with noise, measurement errors, and unexpected fluctuations.

More rigorously, data scientists view time series as the combination of two components:

1. **The underlying signal**: This represents the true behavior and essence of the process we aim to study. Examples include temperature trends, stock movements, or system responses (or, in the example above, the violin).
2. **The noise**: This is the unwanted variation introduced by external disturbances, sensor inaccuracies, or environmental factors (or, in the example above, the city and its sounds).

![png](/images/Polynomial_Filtering/Noise_Definition.svg)

> Note: In this article, the term _"time series"_ and _"signal"_ will be used interchangeably.

It is important to notice that, in real world application, we don't have the **underlying signal**, but we are always dealing with the **measured signal**, where **noise is always present**. For example, in the example above, we cannot perfectly extract the violin from our microphone in the city, because the violin and the city are both recorded by our microphone.

In general, we have three different scenarios:

1. **The noise is almost negligible with respect to the underlying signal.** In this case, the observed time series closely follows the true process, and minimal preprocessing (or no preprocessing at all) is required.
2. **The noise is smaller than the signal.** In this case, the noise interferes with the signal's structure, and some preprocessing is needed.
3. **The noise dominates the signal.** In this case, there is nothing we can do: the data are unusable.

In the figure below, the lime line represents low noise, cyan represents moderate noise, and blue represents high noise

![png](/images/Polynomial_Filtering/Noise_Scenarios.svg)

Even when the noise is smaller than the source signal (cyan scenario), the noise can still disturb further analysis. For example:

1. **Noise might influence the quality in forecasting tasks**. Our goal in forecasting is to predict the underlying signal: we are not trying to predict the noise!

2. **Noise can trigger false positives in anomaly detection**. If your signal is very noisy, how can you tell the difference between a noise fluctuation and an anomaly?

3. **Noise can degrade classification or segmentation accuracy**. In tasks such as fault classification or activity recognition, noisy inputs can blur the boundaries between categories, reducing model reliability.

For these reasons, it is important to run **denoising** (also known as **smoothing** or **filtering**) algorithms that aim to reduce the amount of noise present in our time series.

While it is impossible to retrieve the perfect underlying signal, these algorithms reduce the gap between the measured signal and the underlying one, outputting a more clean and less noisy time series.

There are multiple ways to denoise our time series. The approach that will be explored in this blog post is the Polynomial Filtering, also known as **Savitzky-Golay Filtering**.

The steps of the Savitzky-Golay filter are the following:

1. **Define a window of fixed length:**
   Choose an odd number of data points (e.g., 5, 7, 9...) to form the smoothing window that slides over the signal. This determines the local neighborhood around each point.

2. **Choose the polynomial degree:**
   Decide the degree of the polynomial (e.g., 2 or 3) that will be fit to the data within each window. A higher-degree polynomial captures more curvature but risks overfitting.

3. **Fit the polynomial locally:**
   For each position of the window, fit the chosen polynomial to the data points in the window using least squares regression.

4. **Evaluate the polynomial at the center of the window:**
   Use the fitted polynomial to estimate the smoothed value at the central point of the window.

5. **Slide the window across the signal:**
   Repeat the process for each point in the signal by moving the window,generating a smoothed signal.

Here is a visual summary of the workflow:

![png](/images/Polynomial_Filtering/Saviztky_Golay_Workflow.svg)

The implementation and results of this method will be shown in the next chapter, while in the last chapter, we will apply a forecasting task on the Savitzky-Golay cleaned time series, displaying the power of the denoising algorithm in a real world problem scenario.

## Savitzky-Golay Approach

### Libraries and Setup

To have a controlled experiment, we will build our own time series. This will allow us to inject the wanted level of noise and measure the quality of our denoising algorithm. For this reason, the time series will be synthetically generated with our code.

For this blog post, we will need the following libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
```

### Measured Signal and Noise Injection

Let's create our "underlying signal". This signal represents the true behavior and essence of the process we aim to study.

We are assuming that the underlying signal has a given analytical definition. This is not far from reality, where the essence of a process is described by an analytical function. Examples include:

- **Daily temperature variations** often follow a predictable sinusoidal trend, especially in stable climates.
- **Long-term GDP growth** or inflation trends are often modeled with smooth polynomial or exponential functions
- **In engineering**, the response of a system to a periodic force is analytically modeled as a damped sinusoidal function.

For this blogpost, we will use the following analytical expression:

![png](/images/Polynomial_Filtering/sine_wave.png)

With x (time) defined in the 0-10 range, and y defined as the amplitude of our time series. We can quickly plot this function using the following block of code.

```python
x = np.linspace(0,10,1000)
y = 0.2* np.sin(x) + x + 0.8*np.cos(5*x)
plt.plot(x,y, color = 'white')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Underlying Signal')
```

The plot displays, in white, the underlying signal amplitude (on the y axis) with respect to time (on the x axis).

![png](/images/Polynomial_Filtering/underlying_signal.svg)

```chart
{
  "id": "chart-1",
  "title": "Underlying Signal",
  "dataSource": "chart-1.csv",
  "xAxis": {
    "key": "x"
  },
  "yAxis": {
    "label": "Amplitude"
  },
  "series": [
    {
      "column": "y",
      "name": "Data",
      "type": "line"
    }
  ]
}
```

Now, let's consider the noise to be a random gaussian noise (a noise sampled from a gaussian distribution), with mean = 0 and standard deviation = 0.3.

> More information about the Gaussian Noise definition can be found [here](https://www.geeksforgeeks.org/electronics-engineering/gaussian-noise/)

```python
mean, std = 0, 0.3
noise_custom = mean + std * np.random.randn(1000)
plt.figure(figsize = (15,10))
plt.plot(x, noise_custom, color = 'lime')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Noise')
```

![png](/images/Polynomial_Filtering/noise.svg)

```chart
{
  "id": "chart-2",
  "title": "Noise",
  "dataSource": "chart-2.csv",
  "xAxis": {
    "key": "x"
  },
  "yAxis": {
    "label": "Amplitude"
  },
  "series": [
    {
      "column": "noise",
      "name": "Data",
      "type": "line"
    }
  ]
}
```

A common assumption is that the measured signal is defined as the sum of the underlying signal and an unwanted (and unknown) level of noise. This kind of noise is known as **additive noise**.

In the following graph, we can see the measured signal amplitude on the y axis with respect to time, on the x axis.

```python
noisy_signal = y + noise_custom
plt.plot(x,noisy_signal, color = 'cyan')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Measured signal')
```

![png](/images/Polynomial_Filtering/normal_noise.svg)

```chart
{
  "id": "chart-3",
  "title": "Measured Signal",
  "dataSource": "chart-3.csv",
  "xAxis": {
    "key": "x"
  },
  "yAxis": {
    "label": "Amplitude"
  },
  "series": [
    {
      "column": "measured_signal",
      "name": "Data",
      "type": "line"
    }
  ]
}
```

### Savitzky-Golay Filter Implementation

The Savitzky-Golay implementation can be run in a single line of code thanks to the [scipy.signal.savgol_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) function.

The function has three main arguments:

1. **The signal we are aiming to denoise/clean**, as an array.
2. **The window size**, as an integer number.
3. **The order of our polynomial**, also as an integer number.

For example, cleaning the `noisy_signal`signal, applying a 3rd order polynomial on a 5 sized window can be done using the following line of code:

`filtered_signal = savgol_filter(noisy_signal, 5, 3)`

> Note: Some tuning can be done to find the optimal window length and polynomial order. In this article, a simple manual visual evaluation of multiple window lengths and polynomial sizes have been conducted, and the optimal one has been tested. In general, it is recommended not to exceed the 3-5 polynomial order and testing multiple window lengths to find the best one. A good read about a rigorous approach to find the optimal window length is [the following work](https://arxiv.org/pdf/1808.10489)

In order to visually evaluate how our filter performs, a plot has been generated using the following code.

```python
filtered_signal = savgol_filter(noisy_signal, 5, 3)
plt.figure(figsize = (15,15))
plt.subplot(4,1,1)
plt.plot(x, noisy_signal , lw = 3, color = 'cyan', label = 'Measured signal')
plt.plot(x,savgol_filter(noisy_signal, 5, 2), color = 'white', label = 'Savitzky-Golay Filtered Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(fontsize = 16)
plt.subplot(4,1,2)
plt.plot(x, noisy_signal , lw = 3, color = 'cyan', label = 'Measured signal')
plt.plot(x,savgol_filter(noisy_signal, 5, 2), color = 'white', label = 'Savitzky-Golay Filtered Signal')
plt.xlim(2,2.5)
plt.ylim(1,4)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(fontsize = 16)

plt.subplot(4,1,3)
plt.plot(x,filtered_signal - noisy_signal, label = 'Measured signal - Filtered Signal', color = 'white')
plt.xlabel('Time')
plt.ylabel('Filtered - Noisy Signal')
plt.subplot(4,1,4)
plt.plot(x,noise_custom, label = 'Manually Injected Noise')
plt.ylabel('Injected Noise')
```

Here is a brief description of this plot:

- **Top Plot**:

  - Cyan Line: Represents the measured signal, which is the noisy observation of the underlying process.

  - White Line: The Savitzky-Golay filtered signal, i.e., the measured signal after the denoising.

- **Second Plot**: A zoomed in representation of the first plot

- **Third Plot**: The difference between the filtered signal and the noisy signal is shown in white.

- **Bottom Plot**: The actual noise that was artificially injected into the true signal to simulate a noisy real-world observation is shown in lime.

  ![png](/images/Polynomial_Filtering/Filtering_result.svg)

```chart
{
  "id": "chart-4",
  "title": "Savitzky-Golay Filtering",
  "dataSource": "chart-4.csv",
  "xAxis": {
    "key": "x"
  },
  "yAxis": {
    "label": "Amplitude"
  },
  "series": [
    {
      "column": "noisy_signal",
      "name": "Measured Signal",
      "type": "line"
    },
    {
      "column": "savgol_filtered",
      "name": "Savitzky-Golay Filtered Signal",
      "type": "line"
    }
  ]
}
```

```chart
{
  "id": "chart-5",
  "title": "Savitzky-Golay Filtering",
  "dataSource": "chart-4.csv",
  "xAxis": {
    "key": "x"
  },
  "yAxis": {
    "label": "Amplitude"
  },
  "series": [
    {
      "column": "difference",
      "name": "Difference",
      "type": "line"
    }
  ]
}
```

```chart
{
  "id": "chart-6",
  "title": "Savitzky-Golay Filtering",
  "dataSource": "chart-4.csv",
  "xAxis": {
    "key": "x"
  },
  "yAxis": {
    "label": "Amplitude"
  },
  "series": [
    {
      "column": "injected_noise",
      "name": "Injected Noise",
      "type": "line"
    }
  ]
}
```

What we can notice is:

1. The filter crucially denoises the noisy input, as the filtered signal is much more smooth than the real one. (Top and Second Plot)
2. The difference between the filtered signal and the measured signal follows the same kind of distribution of the injected noise, confirming the quality of the denoising (Third and Bottom Plot).

## Forecasting on Cleaned Signal

In time series, cleaning the signal should really be in the preprocessing steps for any Machine Learning task. This is because, most of the times, we are interested in forecasting the underlying signal, and not the random variation that we see in the measured one.

In order to show this we will use the Nixtla Forecasting algorithm [TimeGPT-1](https://www.nixtla.io/docs/introduction/introduction) and we will test it on the measured signal and on the measured signal after the application of the Savitzky-Golay filter.

### Define your NixtlaClient

The first thing we will need is to define the API Key. You can follow the instructions [here](https://www.nixtla.io/docs/introduction/introduction)

```python
from nixtla import NixtlaClient
api_key = "api_key"
model = NixtlaClient(api_key=api_key)
```

### TimeGPT Predictions

Now we can run the TimeGPT predictions on the noisy and the Savitzky-Golay filtered signal:

```python
#Pick one of the two
input_signal = noisy_signal
input_signal = filtered_signal

def run_timegpt(input_signal = input_signal)
    window = 15
    ds = pd.date_range(start='2023-01-01', periods=len(x)-window, freq='D')
    df = pd.DataFrame({'ds': ds[900:], 'y': input_signal[900:-window]})

    # Run forecast with TimeGPT-1
    forecast = model.forecast(df, h=window)  # h = forecast horizon

    ds = pd.date_range(start='2023-01-01', periods=len(x), freq='D')
    df = pd.DataFrame({'ds': ds, 'y': y})
    return df, forecast
```

This is how we can run TimeGPT-1 on both the real and the cleaned signal.

```python
df, noisy_forecast = run_timegpt(input_signal = noisy_signal)
_, cleaned_forecast = run_timegpt(input_signal = cleaned_forecast)
```

We can now visually explore the difference between forecasting on a cleaned and a noisy/measured signal using the code below:

```python
x_df = np.array(df['ds'])
y_df = np.array(df['y'])
x_forecast = np.array(forecast['ds'])
y_forecast = np.array(cleaned_forecast['TimeGPT'])
y_messy_forecast = np.array(noisy_forecast['TimeGPT'])
plt.figure(figsize = (15,10))
plt.subplot(2,1,1)
plt.plot(x_df, y_df, label = 'Measured signal')
plt.plot(x_forecast, y_forecast, color = 'cyan', label = 'TimeGPT Prediction (on cleaned input)')
plt.plot(x_forecast, y_messy_forecast, color = 'white', label = 'TimeGPT Prediction (on noisy input)')

plt.ylabel('Amplitude', fontsize = 20)
plt.xlabel('Time', fontsize = 20)
plt.legend(fontsize = 15, loc = 'upper left')
plt.xticks(fontsize = 10, rotation = 45)
plt.yticks(fontsize = 10, rotation = 0)
plt.subplot(2,1,2)
plt.plot(x_df, y_df, label = 'Measured signal')
plt.plot(x_forecast, y_forecast, color = 'cyan', label = 'TimeGPT Prediction (on cleaned input)')
plt.plot(x_forecast, y_messy_forecast, color = 'white', label = 'TimeGPT Prediction (on noisy input)')

plt.legend(fontsize = 15, loc = 'upper left')
plt.xticks(fontsize = 10, rotation = 45)
plt.yticks(fontsize = 10, rotation = 0)
plt.xlim(pd.Timestamp("2025-09-01"),pd.Timestamp("2025-09-26"))
plt.ylabel('Amplitude', fontsize = 20)
plt.xlabel('Time', fontsize = 20)
plt.ylim(9.5,)
plt.savefig('images/TimeGPT_Prediction_vs_noisy.svg')
plt.tight_layout()
```

A brief description of the plot is shown below:

- The top subplot shows the full time series of the underlying signal (in green), the TimeGPT forecast (in cyan) applied on the **Savitzky-Golay filtered signal**, and the TimeGPT forecast applied on the noisy input (real) signal.
- The bottom subplot shows a zoomed in version of the top subplot.

![png](/images/Polynomial_Filtering/TimeGPT_Prediction_vs_noisy.svg)

```chart
{
  "id": "chart-7",
  "title": "TimeGPT Predictions vs Noisy Input",
  "dataSource": "chart-5.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "Amplitude"
  },
  "series": [
    {
      "column": "measured_signal",
      "name": "Measured Signal",
      "type": "line"
    },
    {
      "column": "timegpt_cleaned",
      "name": "TimeGPT Prediction Cleaned",
      "type": "line"
    },
    {
      "column": "timegpt_noisy",
      "name": "TimeGPT Prediction Noisy",
      "type": "line"
    }
  ]
}
```

```chart
{
  "id": "chart-8",
  "title": "TimeGPT Predictions vs Noisy Input (zoom)",
  "dataSource": "chart-6.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "Amplitude"
  },
  "series": [
    {
      "column": "measured_signal",
      "name": "Measured Signal",
      "type": "line"
    },
    {
      "column": "timegpt_cleaned",
      "name": "TimeGPT Prediction Cleaned",
      "type": "line"
    },
    {
      "column": "timegpt_noisy",
      "name": "TimeGPT Prediction Noisy",
      "type": "line"
    }
  ]
}
```

As we can see, the cyan predictions, on the cleaned input, are much more in line with the green signal. This is because the cyan predictions are generated on a preprocessed cleaned signal, while the white predictions are generated on a noisy non cleaned input one.

## Conclusions
Let's summarize the topics treated in the blogposts:

- Measured time series are usually noisy and noise can impact your Machine Learning downstream tasks (e.g. in forecasting, anomaly detection, or classification)

- Savitzky–Golay filtering has been applied as a smoothing technique to smooth the measured signal and remove its noise. The idea of the filter is to fit a local polynomial over a sliding window.

- Implementation is straightforward. With scipy.signal.savgol_filter, only the window length and polynomial degree need to be tuned.

- Filtering improves model performance. In the example, TimeGPT forecasts on the denoised signal were closer to the true underlying signal than forecasts on noisy data.
