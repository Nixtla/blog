---
title: Time Series Frequency Modelling with Fourier Transform and TimeGPT-1
description: Discover how to decompose your time series in multiple components with Fourier Transform and model each component with TimeGPT-1.
image: /images/fourier_modelling/main_picture.svg
categories: ["Time Series Forecasting"]
tags:
  - TimeGPT
  - Fourier Transform
  - frequency analysis
  - signal decomposition
author_name: Piero Paialunga
author_image: "/images/authors/piero.jpg"
author_position: Data Scientist
publication_date: 2025-08-26
---

Real-world time series often consist of patterns that occur **periodically** and at **multiple scales**. This is because Nature is inherently rhythmic, and so are human behaviors.

For example:

- We wake up **every day** at a 7.00 AM.
- We celebrate **our birthday** after 365 days.
- We relax **every Sunday**.

Nonetheless, in practice, signals are composed of multiple overlapping components acting at different scales, which are all bundled together. These **time scales** refer to the characteristic durations over which patterns repeat or evolve. In the examples above:

- The time scale of our birthday is 365 days
- The time scale of our day is 24 hours
- The time scale of the weekend is 7 days.

Each of these scales may reflect a different underlying cause or behavior in the data.
By isolating and analyzing each time scale separately, we can uncover a deeper understanding of the problem by revealing its multiple sources, which operate at different rhythms and periodicities.

A very helpful and widely used mathematical transformation in this context is the **Fourier Transform (FT)**. The FT transforms a time-domain signal into its periodic components, revealing the underlying sources that drive the observed patterns. This method is very well known and frequently used in signal processing engineering.

In this blog post, we will combine the intuition of the Fourier Transform with the power of [TimeGPT-1](https://www.nixtla.io/docs), a state-of-the-art foundation model for time series forecasting developed by Nixtla, capable of modelling time series with a very high degree of accuracy. TimeGPT-1 will be thus adopted at different scales and will be used to model all the sub-patterns of our time series.

## Fourier Transform

You might be familiar with the equation of a **sine wave**:
![Alt text](/images/fourier_modelling/sinewave.svg)
In this equation:

- y(t) is the value of the signal at time t
- A is the amplitude (how tall the wave is)
- f is the frequency (how often it repeats)
- ϕ is the phase (how shifted the wave is left or right)

For example, let's consider the time series defined by the equation `y(t) = 1 * sin(1t) + 2 * sin(5t)`. The time series is the sum of:

- A sine wave with frequency f = 1 and amplitude A = 1
- A sine wave with frequency f = 5 and amplitude A = 2

Let's represent that. The picture below displays the sine wave y as a function of time t.

```python
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0,8*np.pi,2000)
y = np.sin(t) + 2*np.sin(5*t)
plt.plot(t,y, color ='cyan')
plt.xlabel('Time (t)', fontsize = 20)
plt.ylabel('Amplitude (y)', fontsize = 20)
```

    Text(0, 0.5, 'Amplitude (y)')

```chart
{
  "id": "chart-1",
  "title": "Simple Sine Wave",
  "dataSource": "chart-1.csv",
  "xAxis": {
    "key": "t"
  },
  "yAxis": {
    "label": "Amplitude (y)"
  },
  "series": [
    {
      "column": "y",
      "name": "Actual Data",
      "type": "line"
    }
  ]
}
```

This represents an often encountered situation where our time series is the **sum of multiple sine waves operating at different frequencies, amplitudes, and phases**. In these scenarios, the Fourier Transform shines.

The [Fourier Transform](https://www.sciencedirect.com/topics/engineering/fourier-transform) allows you to explore the time series in the **frequency domain**. More specifically, FT takes this complex time series, made up of many overlapping waves, and breaks it down into a set of sine and cosine components, each with its own frequency and amplitude. This allows you to find all the implicit components and their characteristics in your messy, raw, time series input.

Running the Fourier Transform in Python is extremely easy within the `numpy` module. We'll run it and take a look at the output:

```python
import numpy as np
t = np.linspace(0,8*np.pi,2000)
y = np.sin(t) + 2*np.sin(5*t)
y_fft = np.fft.fft(y)
print(y_fft[0:5])
```

    [2.15105711e-14+0.00000000e+00j 5.75978513e-04-3.66679008e-01j
     2.72675550e-03-8.67950376e-01j 9.50889536e-03-2.01783533e+00j
     6.28414709e+00-1.00013991e+03j]

As we can see, the output of the Fourier Transform is an array of complex numbers. Each value in this array corresponds to a specific frequency component of the original signal.

More precisely:

- The index of the array (or the corresponding frequency from `np.fft.fftfreq`) tells you which frequency this component represents.

- The magnitude (i.e., `np.abs(y_fft[k])`) represents the amplitude of that frequency in the original signal, which represents how strong that frequency is.

- The angle (i.e., `np.angle(y_fft[k])`) represents the phase shift of that frequency.

Now the big question:

**_"Using the Fourier Transform, can we retrieve the two sine waves, with amplitudes 1 and 2 and frequency 1 and 5?"_**

Let's take a look at the Fourier Transform signal. The picture below shows the **amplitude** of the Fourier Transform (y) across multiple frequencies (f).

```python
import numpy as np
from plotter import set_dark_mode
t = np.linspace(0,8*np.pi,2000)
y = np.sin(t) + 2*np.sin(5*t)
y_fft = np.fft.fft(y)
f = t*20 #Numpy implementation constant
y_amp = 2*np.abs(y_fft)/2000 #Amplitude definition
plot_lim = 100
plt.plot(f[:plot_lim],y_amp[:plot_lim], color ='cyan')
plt.xlabel('Frequency (f)',fontsize = 20)
plt.ylabel('FT Amplitude (y)',fontsize = 20)
```

```chart
{
  "id": "chart-2",
  "title": "FFT Amplitude Spectrum",
  "dataSource": "chart-2.csv",
  "xAxis": {
    "key": "frequency"
  },
  "yAxis": {
    "label": "FT Amplitude (y)"
  },
  "series": [
    {
      "column": "amplitude",
      "name": "Actual Data",
      "type": "line"
    }
  ]
}
```

This plot shows that the Fourier Transform successfully reveals the two components that make up the original time series:

- A sine wave with amplitude = 1 and frequency = 1 (first peak)
- A sine wave with amplitude = 2 and frequency = 5 (second peak)

### Inverse Fourier Transform

Now the beauty of the Fourier Transform is that it is **reversible**: if we run the **inverse Fourier Transform** function on the **Fourier Transform time series** we can reconstruct the input time series with a high degree of accuracy.

![png](/images/fourier_modelling/FT_vs_IFFT.svg)

The picture below displays the **inverse fourier transform applied on the Fourier Transform time series** (above, lime color) and the **input time series** (below, cyan color). As we can see, there is a very good match between the two.

```python
import numpy as np
t = np.linspace(0,8*np.pi,2000)
y = np.sin(t) + 2*np.sin(5*t)
y_fft = np.fft.fft(y)
y_iff_fft = np.fft.ifft(y_fft).real #Inverse Fourier Transform!
plt.subplot(2,1,1)
plt.plot(t, y_iff_fft, color ='lime', label = 'Inverse FT')
plt.legend()
plt.subplot(2,1,2)
plt.plot(t, y, color ='cyan', label = 'Original Time Series')
plt.legend()
```

```chart
{
  "id": "chart-3",
  "title": "Inverse FFT Comparison",
  "dataSource": "chart-3.csv",
  "xAxis": {
    "key": "t"
  },
  "yAxis": {
    "label": "Amplitude (y)"
  },
  "series": [
    {
      "column": "inverse_fft",
      "name": "Inverse FT",
      "type": "line",
      "color": "chart-3"
    }
  ]
}
```

```chart
{
  "id": "chart-4",
  "title": "Inverse FFT Comparison",
  "dataSource": "chart-3.csv",
  "xAxis": {
    "key": "t"
  },
  "yAxis": {
    "label": "Amplitude (y)"
  },
  "series": [
    {
      "column": "original",
      "name": "Original Time Series",
      "type": "line"
    }
  ]
}
```

Pretty amazing right? Now let's see how we can use this tool.

### Fourier Transform Filtering

Now, let's make things really complicated. Let's build a time series with:

- 3 low-frequency sine wave components `f = [0.1,0.5,1]`
- 3 high-frequency sine wave components: `f = [7,8,15]`

The time series will be composed of the sum of these 6 sine waves: three low-frequency and three high-frequency components. The resulting time series can be seen below:

```python
from plotter import set_dark_mode
low_frequencies = [0.1,0.5,1]
high_frequencies = [7,8,15]
frequencies = low_frequencies + high_frequencies
t = np.linspace(0,8*np.pi,2000)
y = np.zeros_like(t)
for f in frequencies:
    rand_amp = np.random.choice(np.linspace(0,5,2000))
    y += rand_amp*np.sin(t*f)
plt.plot(t,y, color ='cyan')
plt.xlabel('Time (t)',fontsize = 20)
plt.ylabel('Amplitude (y)',fontsize = 20)
```

```chart
{
  "id": "chart-5",
  "title": "Fourier Filtering - Frequency Bands",
  "dataSource": "chart-4.csv",
  "xAxis": {
    "key": "t"
  },
  "yAxis": {
    "label": "Amplitude (y)"
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

For such a complex time series, retrospectively identifying the **specific frequency** component is not trivial nor necessary. Nonetheless, we can run the Fourier Transform and split it into two non overlapping parts:

- A low-frequency band, where we zero out all components above a certain threshold.

- A high-frequency band, where we zero out all components below that same threshold.

This produces two separate Fourier Transforms:

- One that retains only the lower-frequency content

- One that retains only the higher-frequency content

The graph below shows the result of this filtering: the low-frequency Fourier Transform is shown in cyan while the high-frequency Fourier Transform is shown in lime

```python
from numpy.fft import fft, ifft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

# Compute FFT
y_fft = fft(y)
freqs = fftfreq(len(t), d=(t[1] - t[0]))

# Helper function to filter by frequency range
def filter_freq_component(y_fft, freqs, low, high):
    mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
    filtered_fft = np.zeros_like(y_fft, dtype=complex)
    filtered_fft[mask] = y_fft[mask]
    return filtered_fft

# Define frequency bands
low_band = (0.0, 1.0)
high_band = (1.0, 20)

# Filter each band
low_fft = filter_freq_component(y_fft, freqs, *low_band)
high_fft = filter_freq_component(y_fft, freqs, *high_band)

# Plot FFT magnitudes
plt.figure(figsize=(10, 5))
plt.plot(np.abs(low_fft)[0:100], label="Low Frequencies", linewidth=4,color='cyan')
plt.plot(np.abs(high_fft)[0:100]*1, label="High Frequencies", linewidth=4,color='lime')
plt.xlabel("Frequency Bin", fontsize=14)
plt.ylabel("Magnitude", fontsize=14)
plt.title("Low vs High Frequency Bands in FFT", fontsize=16)
plt.legend()
plt.grid(alpha=0.3)
plt.show()



```

```chart
{
  "id": "chart-6",
  "title": "Low vs High Frequency Bands in FFT",
  "dataSource": "chart-5.csv",
  "xAxis": {
    "key": "frequency_bin"
  },
  "yAxis": {
    "label": "Magnitude (y)"
  },
  "series": [
    {
      "column": "low_freq_magnitude",
      "name": "Low Frequencies",
      "type": "line"
    },
    {
      "column": "high_freq_magnitude",
      "name": "High Frequencies",
      "type": "line"
    }
  ]
}
```

As we can see, we retrieve three peaks for the low frequencies and three peaks for the high frequencies. Now we can do even more: for each one of these parts we can run the **inverse** Fourier Transform.

In the following plot:

- The cyan time series is the Inverse Fourier Transform of the Low Frequency Fourier Transform signal

- The lime time series is the Inverse Fourier Transform of the High Frequency Fourier Transform signal

```python
low_ifft = np.fft.ifft(low_fft).real
high_ifft = np.fft.ifft(high_fft).real
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t, low_ifft, lw = 5, color = 'cyan', label = 'Low IFT')
plt.xlabel('Time (t)',fontsize= 18)
plt.ylabel('Amplitude (y)',fontsize= 18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize = 12)
plt.subplot(2,1,2)
plt.plot(t, high_ifft, lw = 5, color = 'lime', label = 'High IFT ')
plt.xlabel('Time (t)',fontsize= 18)
plt.ylabel('Amplitude (y)',fontsize= 18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.tight_layout()
plt.legend(fontsize = 12)
```

```chart
{
  "id": "chart-7",
  "title": "Filtered Inverse FFT Components",
  "dataSource": "chart-6.csv",
  "xAxis": {
    "key": "t"
  },
  "yAxis": {
    "label": "Amplitude (y)"
  },
  "series": [
    {
      "column": "low_ifft",
      "name": "Low IFT",
      "type": "line"
    }
  ]
}
```

```chart
{
  "id": "chart-8",
  "title": "Filtered Inverse FFT Components",
  "dataSource": "chart-6.csv",
  "xAxis": {
    "key": "t"
  },
  "yAxis": {
    "label": "Amplitude (y)"
  },
  "series": [
    {
      "column": "high_ifft",
      "name": "High IFT",
      "type": "line",
      "color": "lime-500"
    }
  ]
}
```

As we can imagine, the **high frequency** signal is very crazy-up and downy, as you would expect from a high frequency sine wave, and the **low frequency** signal has a more relaxed shape, as you would expect from a low frequency sine wave.

This is very important as these two sub-time series might indicate two different phenomena in our system. For example, in a financial time series, one part might represent the long-scale variation of a quantity, and another might capture short-scale behavior (e.g., impulse decisions).

To summarize: we have a method that can identify multiple sources of the same time series based on their differences in **frequency**. In other words, given a single time series, our method is able to split it in multiple time series that, summed together, will return the original one. This is helpful because it can precisely identify the different phenomena of our system, so that we are able to study them individually.

## Modelling Stage

Now that we have the different source time series, we can worry about **modelling** them. By "modelling" I mean building mathematical or machine learning models that can learn the behavior of each individual component over time, **forecast** its future evolution, and help us understand its contribution to the overall dynamics of the original time series.

The model we are going to use in this blog post is the [Time GPT-1](https://www.nixtla.io/docs) developed by the [Nixtla](https://www.nixtla.io/) team. This model is a large, pre-trained foundation model specifically designed for time series forecasting.

### Nixtla API and Implementation

The first thing we will need is to define your API key. You can get one from [Nixtla's platform](https://dashboard.nixtla.io/). Once you have your API key, store it in your system and replace the `api_key` variable below:

```python
from nixtla import NixtlaClient
api_key = "your_api_key"
nixtla_client = NixtlaClient(api_key=api_key)
```

With the API ready, we can now move on to the actual modeling stage: using the `.forecast` method to predict the next steps for each of the time series components (the low-frequency and high-frequency signals).

```python
import pandas as pd
timeseries_collection = [low_ifft, high_ifft]
df_s = []
forecast_dfs = []
pred_len = 5
quartile = 90
for timeseries in timeseries_collection:
    timestamps = pd.date_range(start='2020-01-01', periods=len(timeseries), freq='H')
    df = pd.DataFrame({
        'ds': timestamps,
        'y': timeseries,
        'unique_id': 0
    })

    test = df[-pred_len:]
    input_seq = df[-pred_len-2000:-pred_len]
    fcst_df = nixtla_client.forecast(
        df=input_seq,
        h=pred_len,
        level=[90],
        finetune_steps=5,
        finetune_loss='mae',
        model='timegpt-1',
        time_col='ds',
        target_col='y'
    )
    forecast_dfs.append(fcst_df)
    df_s.append(df)
```

    INFO:nixtla.nixtla_client:Validating inputs...
    INFO:nixtla.nixtla_client:Inferred freq: H
    INFO:nixtla.nixtla_client:Preprocessing dataframes...
    INFO:nixtla.nixtla_client:Calling Forecast Endpoint...
    INFO:nixtla.nixtla_client:Validating inputs...
    INFO:nixtla.nixtla_client:Inferred freq: H
    INFO:nixtla.nixtla_client:Preprocessing dataframes...
    INFO:nixtla.nixtla_client:Calling Forecast Endpoint...

The code above uses `nixtla_client.forecast()` on each one of the two time series. This means that we are not just forecasting the full time series, but we are individually modelling the low frequency component and the high frequency one. The very promising results are shown for the picture below for the high frequency time series. In the following plot:

- The high frequency time series is shown in lime.
- The forecasting part of the high frequency time series is shown in blue
- `TimeGPT-1` predictions and its confidence is shown in cyan.

```python
test = df_s[1][-pred_len:]
input_seq = df_s[1][-pred_len-2000:-pred_len]
fcst_df = forecast_dfs[1]
plt.suptitle('High Frequency Time Series Forecasting', fontsize = 20)
plt.subplot(2,1,1)
plt.plot(np.array(test['ds']),np.array(test['y']), color = 'blue')
plt.plot(np.array(input_seq['ds'][100:]),np.array(input_seq['y'][100:]), color = 'lime')
plt.plot(np.array(fcst_df['ds']),np.array(fcst_df['TimeGPT']), color = 'cyan')
plt.fill_between(np.array(fcst_df['ds']),np.array(fcst_df['TimeGPT-lo-90']), np.array(fcst_df['TimeGPT-hi-90']), color ='cyan', alpha = 0.2)
plt.subplot(2,1,2)
plt.plot(np.array(test['ds']),np.array(test['y']), color = 'blue')
plt.plot(np.array(input_seq['ds'][1800:]),np.array(input_seq['y'][1800:]), color = 'lime')
plt.plot(np.array(fcst_df['ds']),np.array(fcst_df['TimeGPT']), color = 'cyan')
plt.fill_between(np.array(fcst_df['ds']),np.array(fcst_df['TimeGPT-lo-90']), np.array(fcst_df['TimeGPT-hi-90']), color ='cyan', alpha = 0.2)
```

```chart
{
  "id": "chart-9",
  "title": "High Frequency Time Series Forecasting",
  "dataSource": "chart-7.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "Target [y]"
  },
  "series": [
    {
      "column": "actual",
      "name": "Actual Data",
      "type": "line",
      "color": "blue-700",
      "zIndex": 4
    },
    {
      "column": "forecast",
      "name": "Forecast",
      "type": "line",
      "color": "cyan-400",
      "zIndex": 5
    },
    {
      "type": "area",
      "columns": {
        "high": "hi_90",
        "low": "lo_90"
      },
      "name": "90% Interval"
    }
  ]
}
```

```chart
{
  "id": "chart-10",
  "title": "High Frequency Time Series Forecasting (zoom)",
  "dataSource": "chart-9.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "Target [y]"
  },
  "series": [
    {
      "column": "actual",
      "name": "Actual Data",
      "type": "line",
      "color": "blue-700",
      "zIndex": 4
    },
    {
      "column": "forecast",
      "name": "Forecast",
      "type": "line",
      "color": "cyan-400",
      "zIndex": 5
    },
    {
      "type": "area",
      "columns": {
        "high": "hi_90",
        "low": "lo_90"
      },
      "name": "90% Interval"
    }
  ]
}
```

Similarly strong performance can also be observed for the low-frequency time series:

```python
test = df_s[0][-pred_len:]
input_seq = df_s[0][-pred_len-2000:-pred_len]
fcst_df = forecast_dfs[0]
plt.figure(figsize = (10,8))
plt.suptitle('Low Frequency Time Series Forecasting', fontsize = 20)
plt.subplot(2,1,1)
plt.plot(np.array(test['ds']),np.array(test['y']), color = 'blue')
plt.plot(np.array(input_seq['ds'][100:]),np.array(input_seq['y'][100:]), color = 'lime')
plt.plot(np.array(fcst_df['ds']),np.array(fcst_df['TimeGPT']), color = 'cyan')
plt.fill_between(np.array(fcst_df['ds']),np.array(fcst_df['TimeGPT-lo-90']), np.array(fcst_df['TimeGPT-hi-90']), color ='cyan', alpha = 0.2)
plt.subplot(2,1,2)
plt.plot(np.array(test['ds']),np.array(test['y']), color = 'blue')
plt.plot(np.array(input_seq['ds'][1800:]),np.array(input_seq['y'][1800:]), color = 'lime')
plt.plot(np.array(fcst_df['ds']),np.array(fcst_df['TimeGPT']), color = 'cyan')
plt.fill_between(np.array(fcst_df['ds']),np.array(fcst_df['TimeGPT-lo-90']), np.array(fcst_df['TimeGPT-hi-90']), color ='cyan', alpha = 0.2)
```

```chart
{
  "id": "chart-11",
  "title": "Low Frequency Time Series Forecasting",
  "dataSource": "chart-8.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "Target [y]"
  },
  "series": [
    {
      "column": "actual",
      "name": "Actual Data",
      "type": "line",
      "color": "blue-700"
    },
    {
      "column": "forecast",
      "name": "Forecast",
      "type": "line",
      "color": "cyan-400"
    },
    {
      "type": "area",
      "columns": {
        "high": "hi_90",
        "low": "lo_90"
      },
      "name": "90% Interval"
    }
  ]
}
```

```chart
{
  "id": "chart-12",
  "title": "Low Frequency Time Series Forecasting (zoom)",
  "dataSource": "chart-10.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "Target [y]"
  },
  "series": [
    {
      "column": "actual",
      "name": "Actual Data",
      "type": "line",
      "color": "blue-700"
    },
    {
      "column": "forecast",
      "name": "Forecast",
      "type": "line",
      "color": "cyan-400"
    },
    {
      "type": "area",
      "columns": {
        "high": "hi_90",
        "low": "lo_90"
      },
      "name": "90% Interval"
    }
  ]
}
```

## Putting Everything Together

What we did in this blog post can be summarized in these steps:

1. We applied the Fourier Transform to the input time series to reveal its frequency components.

2. We split those components into low- and high-frequency bands using a threshold filter.

3. We reconstructed each band separately using the Inverse Fourier Transform.

4. We used `TimeGPT-1` to model and forecast each reconstructed component individually.

5. Optionally, we can sum the predictions to recover the full signal, or use the individual predictions if we are interested in the specific source.

This workflow is summarized in the following image

![image](/images/fourier_modelling/mermaid_chart.svg)

## Conclusions

In this article, we introduced a method to decompose and model multi-scale time series using the Fourier Transform and Nixtla’s TimeGPT-1. Specifically, we did the following:

- We decomposed a time series into low- and high-frequency components using the Fourier Transform.
- We reconstructed each component using the Inverse Fourier Transform to isolate different temporal behaviors.
- We modeled each reconstructed signal separately using `TimeGPT-1` to forecast short- and long-term patterns.
- We demonstrated that this approach improves interpretability by analyzing distinct sources within the original time series.
