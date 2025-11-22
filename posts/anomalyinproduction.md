---
title: "Anomaly Detection for Cloud Cost Monitoring with Nixtla"
description: "Learn how to build a synthetic cloud cost dataset and use Nixtla's algorithms to detect spikes, drifts, and level shifts. This approach helps teams monitor performance and prevent unexpected billing surprises."
image: "/images/anomaly_detection_monitoring/CloudCostTimeSeries.svg"
categories: ["Anomaly Detection"]
tags:
  - TimeGPT
  - anomaly detection
  - cloud cost monitoring
  - Python
author_name: Piero Paialunga
author_image: "/images/authors/piero.jpg"
author_position: Data Scientist
publication_date: 2025-09-26
---

Monitoring the cost of the cloud operation is vital for every company. From a mathematical perspective, the cloud cost signal is a perfect example of a time series: the cost (dependent variable, y axis) is monitored against time (independent variable, x axis).

For the most part, this time series can be predicted with a certain level of accuracy. After all, we are the ones using the cloud, we know when we are going to launch products, and we are the ones using a specific cloud provider (AWS, GC or whatnot).

Nonetheless, there are various sources of uncertainty in this process. Some of them are:

- **Unexpected traffic spikes.** A sudden increase in users, seasonal demand, or an unplanned marketing campaign can cause workloads (and thus costs) to surge beyond forecasts.
- **Infrastructure misconfigurations**. A forgotten autoscaling rule, an oversized instance, or a misapplied storage class can quietly add costs.
- **Human error**. Engineers launching experimental clusters, data scientists forgetting to shut down GPUs, or simply misusing reserved instances can all introduce anomalies.

And beyond these, countless other random events can lead to irregular cost behavior. In the language of time series, we call such unexpected deviations **anomalies**. These anomalies often manifest as sudden spikes in the cost time series. In most organizations, a dedicated team or monitoring system is responsible for identifying these anomalies early and triggering alerts when they appear.

To help the monitoring team identify the anomalies, it is good practice to build an anomaly detection algorithm. This blog post wants to highlight how Nixtla can be used to develop such an algorithm. Let's dive in!

## Cloud Cost Model

So where does the cloud cost time series come from?

We can think of the cloud cost as coming mainly from three sources:

1. **The baseline infrastructure cost**: which represents the cost for your cloud infrastructure. This is usually a fixed value.
2. **The traffic cost**: every time someone makes a request, it represents a cost on our side. This is not constant and depends on the number of request at that given time
3. **The noise/random fluctuations**: small variations introduced by billing granularity, background services, data pipeline delays, or even provider-specific pricing quirks. These are not tied to business activity directly, but they add randomness to the time series.

These three sources of costs sum to the total cloud cost.

![Anomaly Example](/images/anomaly_detection_monitoring/Cost.svg)

The traffic itself has been modeled using the following assumptions:

1. There is a linear trend over time: we can expect our cost to grow with the company
2. The weekends are busier than the weekdays: we can expect people to spend more time on our apps when they are less busy.
3. The noise is modeled using a random walk.

In the traffic, there are sorts of "spikes" that should also be part of the model. In general, a **release of a product** leads to an increase of the cloud traffic. For example, when a new version of an LLM is productionized and released to the public, you can expect a large increase in the usage. Even without the release of a new product, a sudden increase in the promotion of old products can create the same effect.

Regardless of the specific reason, these spikes are injected by business choices, and we have a good level of control over them.

For this reason, they serve as a good test case for our time series **anomalies**: we know exactly when they happen, and we can check if our monitoring algorithms work in detecting them.

![Anomaly Example](/images/anomaly_detection_monitoring/Traffic.svg)

All these assumptions are modelled using code. Let's start by importing the necessary libraries and setting up utility functions:

```python
import numpy as np
import pandas as pd

def _rng(seed):
    return np.random.default_rng(None if seed is None else seed)

def _as_df_long(metric_id, ts, values, **extras):
    df = pd.DataFrame({"metric_id": metric_id, "timestamp": ts, "value": values})
    for k, v in extras.items():
        df[k] = v
    return df
```

Next, we'll create a function to generate the base traffic pattern. This incorporates the weekday/weekend behavior, linear growth trend, and random walk noise:

```python
def generate_traffic_pattern(idx, weekday_weekend_ratio, trend_growth, random_walk_std, base_traffic, rng):
    """Generate base traffic pattern with weekday factor, trend, and random walk."""
    n = len(idx)
    weekday = idx.weekday
    weekday_factor = np.where(weekday < 5, 1.0, weekday_weekend_ratio)
    trend = np.linspace(1.0, 1.0 + trend_growth, n)
    random_walk = np.cumsum(rng.normal(0, random_walk_std, n))
    traffic = (base_traffic * weekday_factor * trend * (1 + random_walk)).clip(min=base_traffic * 0.4)
    return traffic
```

To simulate business events like product launches or promotional campaigns, we apply traffic spikes on specific dates:

```python
def apply_promotions(traffic, idx, promo_days, promo_lift):
    """Apply promotional lift to traffic and return promo flags."""
    if promo_days is None:
        promo_days = []
    promo_days = pd.to_datetime(list(promo_days)) if promo_days else pd.to_datetime([])
    promo_flag = np.isin(idx, promo_days).astype(int)
    traffic = traffic * (1 + promo_lift * promo_flag)
    return traffic, promo_flag
```

With the traffic pattern established, we can now calculate the final cloud cost by adding baseline infrastructure costs and random noise:

```python
def calculate_cost_from_traffic(traffic, baseline_infra_usd, cost_per_request, noise_usd, n, rng):
    """Calculate final cost from traffic with baseline infrastructure cost and noise."""
    noise = rng.normal(0, noise_usd, n)
    cost = baseline_infra_usd + traffic * cost_per_request + noise
    return cost
```

Now we bring everything together in the main function that orchestrates the entire simulation:

```python
def make_cloud_cost_daily(
    start,
    end,
    baseline_infra_usd,
    cost_per_request,
    base_traffic,
    weekday_weekend_ratio=0.92,   # weekend traffic lower
    trend_growth=0.55,            # 55% growth across the period
    noise_usd=2.0,              # additive noise
    random_walk_std=0.002,        # slow drift in traffic
    promo_days=None,
    promo_lift=0.25,              # +25% traffic on promo days
    seed=42,
):
    """
    Returns a DataFrame with columns:
      metric_id, timestamp, value, traffic, deploy_flag, promo_flag, notes
    """
    rng = _rng(seed)
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    n = len(idx)

    # Generate base traffic pattern
    traffic = generate_traffic_pattern(
        idx, weekday_weekend_ratio, trend_growth, random_walk_std, base_traffic, rng
    )

    # Apply promotional events
    traffic, promo_flag = apply_promotions(traffic, idx, promo_days, promo_lift)

    # Calculate final cost
    cost = calculate_cost_from_traffic(
        traffic, baseline_infra_usd, cost_per_request, noise_usd, n, rng
    )

    return _as_df_long(
        "cloud_cost_usd",
        idx,
        np.round(cost, 2),
        traffic=traffic.astype(int),
        promo_flag = promo_flag
    )
```

Finally, let's generate the synthetic dataset and examine the first few rows:

```python
cloud_cost_df = make_cloud_cost_daily(
    start="2025-01-01",
    end="2025-08-31",
    baseline_infra_usd=2000.0,
    cost_per_request=8e-4,
    base_traffic=1_000_000,
    promo_days=("2025-03-15", "2025-05-10", "2025-07-04")
)
print(cloud_cost_df.head(3))
print("Rows:", len(cloud_cost_df))
```

|     | metric_id      | timestamp           |   value | traffic | promo_flag |
| --: | :------------- | :------------------ | ------: | ------: | ---------: |
|   0 | cloud_cost_usd | 2025-01-01 00:00:00 | 2797.55 | 1000609 |          0 |
|   1 | cloud_cost_usd | 2025-01-02 00:00:00 |  2804.9 | 1000798 |          0 |
|   2 | cloud_cost_usd | 2025-01-03 00:00:00 | 2801.09 | 1004575 |          0 |

If we display the time series using the following block we get this output:

![Anomaly Example](/images/anomaly_detection_monitoring/CloudCostTimeSeries.svg)

```chart
{
  "id": "chart-1",
  "title": "Cloud Cost Time Series",
  "dataSource": "chart-1.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "Target [y]"
  },
  "series": [
    {
      "column": "value",
      "name": "Time",
      "type": "line"
    }
  ]
}
```

A very important thing to notice is that, as stated above, this function does have **spikes** and an anomaly detection algorithm would typically detect them. Even though these anomalies are injected and they are the results of our business decision, we expect our monitoring algorithm to **detect them**.

## Anomaly Detection Algorithm using Nixtla

The **anomaly detection algorithm** that we will be testing is the `TimeGPT-1` model, developed by the [Nixtla](https://www.nixtla.io/) team. The idea behind TimeGPT-1 is to use the **transformer** algorithm and conformal probabilities to get accurate predictions and uncertainty boundaries. You can read more about it in the original [paper](https://arxiv.org/abs/2310.03589), while another application of anomaly detection through TimeGPT-1 can be found in this [blogpost](https://www.nixtla.io/blog/anomaly_detection).

We use Nixtla’s TimeGPT-1 to forecast tomorrow’s cloud cost and a 99% confidence band from our daily history. This prediction will be used to assess anomalies day by day. More precisely, we will follow this pipeline:

1. **We look at your daily cloud spend up to the current date.** This will be our training set for TimeGPT-1.
2. **TimeGPT-1 guesses tomorrow’s cost and gives a safe range around that guess.** This is our cloud cost estimate.
3. **When tomorrow arrives, we evaluate the expected and real cloud cost**. If the real cost is outside the range and the difference is not tiny, we call it an anomaly. If it’s inside the range or only a hair off, we don’t.
4. **We show a simple chart: recent costs, TimeGPT’s range, and a red mark when something’s off.** In a real world scenario, the data and the plot will be provided to the monitoring team.

We can make it stricter or looser by changing how wide the range is and what “not tiny” means. (In general, it is good practice to define what we are willing to accept as "expected fluctuation" and what isn't).

Here's a chart to display the anomaly detection process:

![Anomaly Example](/images/anomaly_detection_monitoring/anomaly_workflow.svg)

## Nixtla Forecasting Algorithm

Let's explore Nixtla's forecasting algorithm. In just a few lines, you can train TimeGPT-1 up to a given date. Then, you can predict the next day's cloud cost using the trained TimeGPT-1. Finally, you can integrate the real value for the next day into the training, and repeat the process.

To make things easier for you, I wrapped everything around a function named `plot_last_k_days_next_h_forecasts`. This function (and others) is included in the following GitHub Folder: [PieroPaialungaAI/AnomalyDetectionCloudCosts/](https://github.com/PieroPaialungaAI/AnomalyDetectionCloudCosts/tree/main)

> Note: You would need Nixtla's API Key. Follow the instructions [here](https://www.nixtla.io/docs/setup/setting_up_your_api_key).

```python
from simulate_timegpt_anomaly import *

cloud_cost_df.rename(columns= {'metric_id': 'unique_id','timestamp': 'ds', 'value': 'y'}, inplace = True)
plot_last_k_days_next_h_forecasts(
    df=cloud_cost_df,
    api_key=api_key,
    freq="D",
    level=99,
    k=5,                          # last 5 anchors
    h=7,                          # predict next 7 days
    model="timegpt-1",            # or "timegpt-1-long-horizon"
    title="Cloud Cost, 1-week forecasts"
)
```

The output of this function is the following plot:

1. The top plot represents, with the cyan color, the training data for Time GPT-1 (all data but last week). The other colors represent the forecasting using Time GPT-1. As we can see, we first forecast the next day, then we incorporate that day in the training and we forecast the day after and so on. The shaded area is the forecasting uncertainty.

2. The bottom plot is a zoomed in version of the top plot.

![Anomaly Example](/images/anomaly_detection_monitoring/ForecastingOnCloud.svg)

```chart
{
  "id": "chart-2",
  "title": "Cloud Cost, 1-week forecasts",
  "dataSource": "chart-2.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "Cloud Cost"
  },
  "series": [
    {
      "type": "area",
      "columns": {
        "high": "forecast_hi",
        "low": "forecast_lo"
      },
      "name": "Forecast interval (± 99%)"
    },
    {
      "column": "actual",
      "type": "line",
      "name": "Observed"
    },
    {
      "column": "forecast_mean",
      "type": "line",
      "name": "Forecast mean"
    }
  ]
}
```

```chart
{
  "id": "chart-3",
  "title": "Cloud Cost, 1-week forecasts (Zoom)",
  "dataSource": "chart-3.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "Cloud Cost (zoom)"
  },
  "series": [
    {
      "type": "area",
      "columns": {
        "high": "forecast_hi",
        "low": "forecast_lo"
      },
      "name": "Forecast interval (± 99%)"
    },
    {
      "column": "actual",
      "type": "line",
      "name": "Observed"
    },
    {
      "column": "forecast_mean",
      "type": "line",
      "name": "Forecast mean"
    }
  ]
}
```

This experiment leads us to two considerations:

1. **Time GPT-1 forecasting algorithm clearly does a very good and reliable job**. The sinusoidal behavior is obviously represented in the forecasting time series with great accuracy

2. **The monitoring algorithm can be reliably used.** The strategy of integrating one day at a time is promising and fairly straightforward to code.

## Monitoring algorithm

This algorithm is implemented using the function `simulate_and_plot_last_k_next_day_anomalies`, which uses TimeGPT's [`detect_anomalies_online`](https://www.nixtla.io/docs/anomaly_detection/real-time/introduction) method for real-time monitoring.

This is how to run the monitoring algorithm:

```python
results = simulate_and_plot_last_k_next_day_anomalies(
    df=cloud_cost_df[:-40],
    api_key=api_key,
    k=30,          # e.g., last 3 weeks
    level=99,
    model="timegpt-1",
    title="Cloud Cost — next-day anomalies (last 30 days)"
)
```

This is what it looks like:

![Monitoring Example](/images/anomaly_detection_monitoring/MonitoringAlgorithm.svg)

```chart
{
  "id": "chart-4",
  "title": "Cloud Cost — next-day anomalies (last 30 daysss)",
  "dataSource": "chart-4.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "y"
  },
  "series": [
    {
      "column": "actual",
      "type": "line",
      "name": "Observed"
    },
    {
      "column": "forecast_mean",
      "type": "line",
      "name": "Forecast mean",
      "showDots": true
    },
    {
      "type": "area",
      "columns": {
        "high": "forecast_hi",
        "low": "forecast_lo"
      },
      "name": "± 99% interval"
    }
  ],
  "anomalies": {
    "column": "is_anomaly",
    "seriesColumn": "actual"
  }
}
```

```chart
{
  "id": "chart-5",
  "title": "Cloud Cost — next-day anomalies (last 30 days) (zoom)",
  "dataSource": "chart-5.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "y (zoom)"
  },
  "series": [
    {
      "type": "area",
      "columns": {
        "high": "forecast_hi",
        "low": "forecast_lo"
      },
      "name": "± 99% interval"
    },
    {
      "column": "actual",
      "type": "line",
      "name": "Observed"
    },
    {
      "column": "forecast_mean",
      "type": "line",
      "name": "Forecast mean",
      "showDots": true
    }
  ],
  "anomalies": {
    "column": "is_anomaly",
    "seriesColumn": "actual"
  }
}
```

1. **The top plot** shows the full history of cloud costs over the last 30 days.

   - The **blue-green line** is the observed daily spend.
   - The **yellow line with shaded band** is TimeGPT-1’s predicted mean and the ±99% confidence interval.
   - The **yellow dots** are the realized (next-day) costs, which let us compare actuals with the forecast.
   - **Red X marks** highlight anomalies — days where actual costs fell well outside the expected range.

2. **The bottom plot** is a zoomed-in view of the top plot.

As we can see, the "**product spike**" we injected is perfectly recognized by the algorithm as an "anomaly". While this spike is expected, this is a great way to show that the algorithm does find the anomaly in our time series.

The team would monitor the detected anomaly (which happened the **2025-07-04** in our made up example) and assess whether or not an action is required for our system.

## Conclusion

Let’s recap what we covered in this post:

- **We built a synthetic cloud cost dataset** that mimics real-world dynamics: baseline infrastructure costs, traffic-driven costs, and random noise. We also modeled expected spikes from promotions or product launches.

- **We explicitly detect spikes (and drifts/level shifts) as anomalies by design.** Since we know exactly when promotions or product launches occur, we can verify that the model detects anomalies at those points.

- **We demonstrated a monitoring pipeline.** By forecasting one day ahead, then integrating the actual cost back into the training loop, we built a realistic monitoring system that adapts day by day.

- **We validated the approach.** TimeGPT-1 successfully captured the seasonal patterns, trends, and cost fluctuations, while reliably flagging injected anomalies such as product spikes.

Overall, this workflow shows how synthetic data, combined with Nixtla’s forecasting models, can provide a robust foundation for **cloud cost anomaly detection**—helping teams monitor spend, catch surprises early, and keep budgets under control.
