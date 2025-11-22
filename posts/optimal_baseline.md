---
title: Simple Anomaly Detection in Time Series via Optimal Baseline Subtraction (OBS)
description: Discover how to detect anomalies using Optimal Baseline Subtraction and enhance your forecasts with Nixtla’s TimeGPT on real-world weather data.
image: /images/optimal_baseline/main_image.svg
categories: ["Anomaly Detection"]
tags:
  ["anomaly detection", "optimal baseline subtraction", "OBS", "statsforecast"]
author_name: Piero Paialunga
author_image: "/images/authors/piero.jpg"
author_position: Data Scientist
publication_date: 2025-08-26
---

**Anomaly detection** in time series is used to identify unexpected patterns in your time series, and it is widely applied in different fields:

- In energy engineering, a spike in power usage might signal a fault.
- In finance, sudden drops or peaks can indicate major market events.
- In mechanical systems, unusual vibrations may reveal early signs of failure.

In this blog post, we will use weather data as an example use case, and we will find the anomalies in temperature time series for different cities all over the world.

## How Optimal Baseline Subtraction (OBS) Works

If you have a bank of time series and you want to understand if and in what portion of the time series you have an anomaly, a simple but very efficient method is called **optimal baseline subtraction (OBS)**. OBS is based on comparing each time series segment to the most similar historical pattern and analyzing the difference to detect unexpected deviations.

The OBS algorithm is the following:

- **Split the time series into individual segments**, where each segment represents a unit of repeated behavior (e.g., a day, a cycle, or a process run).
- **Build a library of historical segments** by collecting all previous segments in the time series bank.
- **Compare your target segment** with all other segments in the library using a similarity metric, such as Mean Absolute Error (MAE).
- **Select the most similar segment** from the library. We define the most similar segment as the **optimal baseline**.
- **Subtract the optimal baseline from the target segment** to isolate the residual (i.e., the absolute difference).
- **Analyze the residual** to identify large deviations, which are flagged as potential anomalies.

![png](/images/optimal_baseline/mermaid_workflow.svg)

## Optimal Baseline Subtraction Application

### Data and Setup

To download the code and data you need for this tutorial, clone the [PieroPaialungaAI/OptimalBaselineSubtraction](https://github.com/PieroPaialungaAI/OptimalBaselineSubtraction.git) repo:

```bash
git clone https://github.com/PieroPaialungaAI/OptimalBaselineSubtraction.git
```

The data used for this article originally come from an Open Database on Kaggle. You can find the original source of the dataset [here](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data). Nonetheless, note that **you don't need to download it again, as everything you need is in the [OBS_Data folder](https://github.com/PieroPaialungaAI/OptimalBaselineSubtraction/tree/main/OBS_data).**

The "preprocessing" part of the data is handled by the [data.py](https://github.com/PieroPaialungaAI/OptimalBaselineSubtraction/blob/main/data.py) code, so we can just deal with the fun stuff here. If you want to see the specific code for every block, feel free to visit the source code. A table with the attributes for the cities can be found in the `.city_attribute_data`:

```python
from data import *
data = TimeSeriesData()
data.city_attribute_data.head()
```

|     | City          | Country       | Latitude  | Longitude   |
| --- | ------------- | ------------- | --------- | ----------- |
| 0   | Vancouver     | Canada        | 49.249660 | -123.119339 |
| 1   | Portland      | United States | 45.523449 | -122.676208 |
| 2   | San Francisco | United States | 37.774929 | -122.419418 |
| 3   | Seattle       | United States | 47.606209 | -122.332069 |
| 4   | Los Angeles   | United States | 34.052231 | -118.243683 |

While the corresponding time series can be found in the `.temperature_data`, where each column represents a city, and the `.datetime` is the time step column. For example, the data for the city of **Vancouver** are the following:

```python
data.temperature_data[['Vancouver','datetime']].head()
```

|     | Vancouver  | datetime            |
| --- | ---------- | ------------------- |
| 0   | 284.630000 | 2012-10-01 13:00:00 |
| 1   | 284.629041 | 2012-10-01 14:00:00 |
| 2   | 284.626998 | 2012-10-01 15:00:00 |
| 3   | 284.624955 | 2012-10-01 16:00:00 |
| 4   | 284.622911 | 2012-10-01 17:00:00 |

### Selecting a Target Segment

So let's say we have our dataset and we want to see if there is an anomaly in a specific section (target curve). This part of the blog post allows you to select the city of interest and to pick a specific window (e.g. a day, a week or a month) for that city. The chosen window and city represents a segment of the time series that we want to analyze with our anomaly detection method.

For example, let's pick `day` number `377` for `city = Los Angeles`. This will be our **target curve**: we will perform anomaly detection on this chosen section of the time series.

```python
city = 'Los Angeles'
segment_class = 'day'
segment_idx = 377
data.isolate_city_and_time(city = city, segment_idx = segment_idx, segment_class = segment_class)
```

### Visualizing the Optimal Baseline

We will look for the optimal baseline in the remaining part of the dataset (everything but the target curve) which can be seen using the following plot function:

```python
data.plot_target_and_baseline()
```

![png](/images/optimal_baseline/OBS_Notebook_5_0.svg)

```chart
{
  "id": "chart-1",
  "title": "Target Segment City = Los Angeles, Segment Class = day, Segment Index = 377",
  "dataSource": "chart-1.csv",
  "xAxis": {
    "key": "hour"
  },
  "yAxis": {
    "label": "Temperature (K)"
  },
  "series": [
    {
      "column": "target_curve",
      "name": "Target Time Series",
      "type": "line"
    }
  ]
}
```

```chart
{
  "id": "chart-2",
  "title": "Remaining Part of the Time Series",
  "dataSource": "chart-1-2.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "Temperature (K)"
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

This plot shows how OBS identifies anomalies by comparing a target segment to historical data:

- The top subplot shows the target day for Los Angeles. There's a noticeable dip followed by a sharp rise in temperature, which may indicate an unusual event.

- The bottom subplot shows the full historical time series used to find the most similar pattern (the optimal baseline).

We will split the full time series below into segments, and we will compare the segments with the top time series so that we can detect deviations that stand out from expected behavior.

By selecting our city, window and index, we have uniquely defined our **target segment/window**. All the remaining windows form your bank of candidates. For example, for our `day` window, our segments have 24 points (one per hour). For this reason the list of candidates will have shape `(number of days - 1, 24)`.

```python
data.list_of_candidates.shape
```

    (1853, 24)

Let's display some random candidates.

```python
data.plot_target_and_candidates()
```

![png](/images/optimal_baseline/OBS_Notebook_9_0.svg)

```chart-multiple
{
  "id": "chart-multiple-1",
  "title": "Target and Candidate Baselines",
  "dataSource": "chart-2.csv",
  "columns": 4,
  "xAxis": { "key": "hour" },
  "yAxis": { "label": "Temperature (K)" },
  "charts": [
    {
      "id": "chart-inner-1",
      "series": [
        { "column": "target_curve", "name": "Target Time Series", "type": "line" },
        { "column": "candidate_3", "name": "Possible Candidate", "type": "line" }
      ]
    },
    {
      "id": "chart-inner-2",
      "series": [
        { "column": "target_curve", "name": "Target Time Series", "type": "line" },
        { "column": "candidate_4", "name": "Possible Candidate", "type": "line" }
      ]
    },
    {
      "id": "chart-inner-3",
      "series": [
        { "column": "target_curve", "name": "Target Time Series", "type": "line" },
        { "column": "candidate_5", "name": "Possible Candidate", "type": "line" }
      ]
    },
    {
      "id": "chart-inner-4",
      "series": [
        { "column": "target_curve", "name": "Target Time Series", "type": "line" },
        { "column": "candidate_6", "name": "Possible Candidate", "type": "line" }
      ]
    },
    {
      "id": "chart-inner-5",
      "series": [
        { "column": "target_curve", "name": "Target Time Series", "type": "line" },
        { "column": "candidate_7", "name": "Possible Candidate", "type": "line" }
      ]
    },
    {
      "id": "chart-inner-6",
      "series": [
        { "column": "target_curve", "name": "Target Time Series", "type": "line" },
        { "column": "candidate_8", "name": "Possible Candidate", "type": "line" }
      ]
    },
    {
      "id": "chart-inner-7",
      "series": [
        { "column": "target_curve", "name": "Target Time Series", "type": "line" },
        { "column": "candidate_9", "name": "Possible Candidate", "type": "line" }
      ]
    },
    {
      "id": "chart-inner-8",
      "series": [
        { "column": "target_curve", "name": "Target Time Series", "type": "line" },
        { "column": "candidate_10", "name": "Possible Candidate", "type": "line" }
      ]
    }
  ]
}
```

As we can see above, each subplot compares the target curve (lime) with a possible candidate (blue). Some candidates match the target candidate closely, while others show very different patterns. This highlights how some segments are good matches and others are not.

### Selecting the Optimal Baseline

In this step, we will use the MAE metric to find the **optimal baseline** (i.e. time series that is the closest to our target in the list of candidates). The code to find the optimal baseline is the following, note that the result of the optimal baseline search is stored in a dictionary:

```python
optimal_baseline_data = data.find_optimal_baseline()
optimal_baseline_data
```

    {'optimal_baseline_curve': array([293.7 , 292.67, 291.21, 289.89, 288.9 , 288.12, 287.4 , 286.99,
            286.68, 286.2 , 285.71, 285.3 , 284.77, 284.55, 285.35, 287.65,
            290.15, 292.2 , 293.24, 295.04, 296.17, 296.54, 296.74, 296.53]),
     'optimal_baseline_diff': array([0.26      , 0.2485    , 0.51      , 0.1       , 0.02      ,
            0.11      , 0.42      , 0.14      , 0.24      , 0.07      ,
            0.25      , 0.35      , 0.38      , 0.93033333, 3.26066667,
            1.256     , 4.1655    , 0.79      , 1.46      , 0.4175    ,
            0.41      , 0.3       , 0.17      , 0.42      ]),
     'optimal_baseline_error': 0.694937499999997,
     'target_curve': array([293.96      , 292.4215    , 290.7       , 289.79      ,
            288.88      , 288.23      , 287.82      , 287.13      ,
            286.44      , 286.27      , 285.96      , 285.65      ,
            285.15      , 283.61966667, 282.08933333, 288.906     ,
            285.9845    , 291.41      , 294.7       , 294.6225    ,
            295.76      , 296.24      , 296.57      , 296.11      ])}

Our plot function can be used to display the optimal baseline vs the target curve and the **residual**, which is the absolute difference between the two divided by the maximum of the target curve (scaling factor):

```python
data.plot_target_and_optimal_baseline()
```

![png](/images/optimal_baseline/OBS_Notebook_13_0.svg)

```chart
{
  "id": "chart-3",
  "title": "Optimal Baseline And Target",
  "dataSource": "chart-3.csv",
  "xAxis": {
    "key": "hour"
  },
  "yAxis": {
    "label": "Temperature (K)"
  },
  "series": [
    {
      "column": "target_curve",
      "name": "Target Time Series",
      "type": "line"
    },
    {
      "column": "optimal_baseline",
      "name": "Optimal Baseline",
      "type": "line"
    }
  ]
}
```

```chart
{
  "id": "chart-4",
  "title": "Scaled Optimal Baseline Difference",
  "dataSource": "chart-3.csv",
  "xAxis": {
    "key": "hour"
  },
  "yAxis": {
    "label": "Temperature (K)"
  },
  "series": [
    {
      "column": "residual",
      "name": "Residual",
      "type": "line"
    }
  ]
}
```

The plot shows the comparison between the target curve and the optimal baseline:

- The top plot shows the target curve (lime) and the optimal baseline (cyan). The two curves track each other closely overall, but diverge slightly around the middle of the day.
- The bottom plot shows the scaled residual (white), which highlights where and when the difference is largest.

The peaks we can observe in the bottom plot are suspicious, and help us spot possible anomalies in the target segment.

We can see that the target time series is very in line with the Optimal Baseline, except for a small area (around the 15th hour), that we can consider as an **anomaly**. So now we can use a threshold to flag anomalies: any point where the residual exceeds this threshold is considered anomalous, as shown in the graph below.

```python
data.run_anomaly_detection(threshold=0.007, plot = True)
```

![png](/images/optimal_baseline/OBS_Notebook_15_1.svg)

```chart
{
  "id": "chart-5",
  "title": "Anomaly Detection Results",
  "dataSource": "chart-4.csv",
  "xAxis": {
    "key": "hour"
  },
  "yAxis": {
    "label": "Scale Residual (K)"
  },
  "series": [
    {
      "column": "residual",
      "name": "Residual",
      "type": "line"
    },
    {
      "column": "threshold",
      "name": "Threshold = 0.007",
      "type": "line",
      "strokeDashArray": "5,5"
    }
  ],
  "anomalies": {
    "column": "is_anomaly",
    "seriesColumn": "residual"
  }
}
```

The plot explains how to use the residual time series to flag anomalies. The residual curve is shown in white, with a threshold line at 0.007 (cyan). Two points exceed the threshold and are flagged as anomalies (lime dots). This method detects the earlier deviation we saw around hour 15 as an anomaly.

## Optimal Baseline Subtraction Considerations

The OBS method described above is a simple yet powerful approach for detecting anomalies in time series data using historical patterns within a statistical dataset.
Let's list some considerations:

- This method can be considered as a **preprocessing approach** and can be done before applying Machine Learning methods.
- This method is very versatile, and can be used as an **unsupervised approach**, as shown above. In presence of a labeled dataset (1/0 for anomaly/non anomaly), the choice of the threshold can be calibrated to achieve maximum accuracy
- The performance of this method increases when the historical dataset is **large**, as more optimal baseline (lower error) can be found.
- The bank of candidates can be personalized, for example by considering only time series that happen **before** the target one, or only time series that belong to a certain class (e.g. only time series from the same city in the example above).

## Application to Nixtla

Optimal Baseline Subtraction (OBS) can be seamlessly integrated into a forecasting pipeline using Nixtla’s [StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/index.html) library. After detecting and optionally replacing anomalies in your time series using OBS, you can feed the cleaned time series directly into models like AutoARIMA, AutoETS, or MSTL to improve forecasting accuracy. This is how you would do it:

### OBS as "Cleaner"

The OBS is here used as a "cleaner" for the time series: given your time series, you can replace the anomaly values of the target time series with the ones of the optimal baseline. For example, if the index 15 is an anomaly in the target time series, you replace the original value of the target time series with the one of the optimal baseline.

```python
from data import *
import pandas as pd
import numpy as np

# Step 1: Get target and baseline
data = TimeSeriesData()
data.isolate_city_and_time(city='Los Angeles', segment_class='day', segment_idx=377)
data.find_optimal_baseline()
threshold = 0.007
anomaly_data = data.run_anomaly_detection(threshold = threshold, plot = False)
fixed_target = anomaly_data['target_replaced_anomaly']
```

As we can see here, the OBS is fixing the areas where we have seen anomalies:

```python
plt.plot(data.optimal_baseline_data['target_curve'], label = 'Target Time Series', color ='k')
plt.plot(fixed_target, label = 'Target Time Series after OBS', color ='red')
plt.legend()
```

![png](/images/optimal_baseline/OBS_Notebook_20_1.svg)

```chart
{
  "id": "chart-6",
  "title": "Target Before and After OBS Cleaning",
  "dataSource": "chart-5.csv",
  "xAxis": {
    "key": "hour"
  },
  "yAxis": {
    "label": "Temperature (K)"
  },
  "series": [
    {
      "column": "target_original",
      "name": "Target Time Series",
      "type": "line"
    },
    {
      "column": "target_after_obs",
      "name": "Target Time Series after OBS",
      "type": "line"
    }
  ]
}
```

As we can see here, the OBS method smooths out the anomaly area by replacing the original values (green) with those from the optimal baseline, producing a "corrected" time series result (cyan). The corrected curve now follows a more consistent pattern, which helps reduce noise and improve downstream forecasting models.

### Nixtla Forecast Stage

Now we can use the Nixtla's [StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/index.html) library to run the forecast block with this code:

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
timestamps = pd.date_range(start='2020-01-01', periods=len(fixed_target), freq='H')
df = pd.DataFrame({
    'ds': timestamps,
    'y': fixed_target,
    'unique_id': 'los_angeles'  # Required by StatsForecast
})
sf = StatsForecast(models=[AutoARIMA(season_length=10)], freq='H')
sf.fit(df)
forecast = sf.predict(h=24)
forecast.head()

```

|     | unique_id   | ds                  | AutoARIMA  |
| --- | ----------- | ------------------- | ---------- |
| 0   | los_angeles | 2020-01-02 00:00:00 | 296.220402 |
| 1   | los_angeles | 2020-01-02 01:00:00 | 296.330804 |
| 2   | los_angeles | 2020-01-02 02:00:00 | 296.441205 |
| 3   | los_angeles | 2020-01-02 03:00:00 | 296.551607 |
| 4   | los_angeles | 2020-01-02 04:00:00 | 296.662009 |

The idea here is that we are doing the forecasting only on "clean" data, so that when we see clear differences between our prediction and the target time series we can spot anomalous points efficiently.

## Conclusions

In this post, we explored how Optimal Baseline Subtraction (OBS) can be used as a simple yet efficient **statistical method** for anomaly detection in time series. Some takeaways are:

- The time series segment of interest (**target**) is compared to its most similar historical counterpart, known as the **optimal baseline**.
- A similarity metric, like the **Mean Absolute Error**, is used to obtain the optimal baseline
- By subtracting the optimal baseline from the target, we obtain the **residual**.
- A **fixed threshold** is applied to the residual and points above the threshold are flagged as anomalies.

OBS can be a preprocessing stage to be integrated with the Nixtla’s StatsForecast library. By forecasting on "clean" (i.e. anomaly-filtered time series), we reduce the risk of bias or distortion caused by outliers. This makes the forecast more reliable, and, crucially, any large discrepancies between forecasted and observed values can now be interpreted as new potential anomalies.
