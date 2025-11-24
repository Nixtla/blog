---
title: "Anomaly Detection in Time Series with TimeGPT and Python"
description: "Discover how to use TimeGPT for scalable, accurate anomaly detection in Python Includes real-world time series, exogenous variables, and adjustable confidence levels."
image: "/images/anomaly_detection/anomaly_detection.svg"
categories: ["Anomaly Detection"]
tags:
  - TimeGPT
  - anomaly detection
  - Python
  - confidence intervals
author_name: Khuyen Tran
author_image: "/images/authors/khuyen.jpeg"
author_position: Developer Advocate - Nixtla
publication_date: 2025-08-26
---

Imagine that you're tracking daily website traffic. Some days show spikes in visits, but it's hard to tell which ones reflect real behavioral changes versus normal fluctuations. Manually spotting anomalies is time-consuming, unreliable, and impractical as your traffic data becomes more complex.

This is where anomaly detection becomes essential. Instead of relying on manual inspection or guesswork, a model like TimeGPT provides consistent, automated detection of unusual behavior based on the page's historical trends and known patterns.

This tutorial walks through using TimeGPT from Nixtla to detect anomalies in daily visits to Peyton Manning's Wikipedia page.

The source code of this article can be found [here](https://nixtla.github.io/nixtla_blog_examples/notebooks/anomaly_detection.html).

## What is TimeGPT?

**TimeGPT** is a generative time series model built by [Nixtla](https://www.nixtla.io/). It learns patterns from vast amounts of time series data to generate accurate forecasts and detect anomalies with minimal manual tuning.

Unlike traditional statistical models, TimeGPT adapts to different data behaviors (seasonality, trends, outliers) automatically and can integrate exogenous features like holidays, promotions, or events. This makes it highly effective for real-world anomaly detection tasks.

To install TimeGPT, simply use:

```bash
pip install nixtla
```

This installs the full Nixtla Python client, which includes support for TimeGPT-based forecasting and anomaly detection APIs.

## Initialize the Nixtla Client

To get started with anomaly detection using TimeGPT, begin by obtaining your API key by signing in at [Nixtla Dashboard](https://dashboard.nixtla.io/) to configure the [Nixtla client](https://www.nixtla.io/docs/intro).

This client provides access to TimeGPT's forecasting and anomaly detection models via a simple Python interface.

```python
import os

import pandas as pd
from nixtla import NixtlaClient


NIXTLA_API_KEY = os.environ["NIXTLA_API_KEY"]
nixtla_client = NixtlaClient(api_key=NIXTLA_API_KEY)
```

## Load and Prepare the Dataset

Next, you'll load the historical page view data from the Peyton Manning dataset. This dataset contains daily Wikipedia page views for Peyton Manning, a former NFL quarterback, spanning multiple years.

It's well-suited for anomaly detection because traffic naturally spikes around events like playoff games, retirement announcements, or media coverage. These real-world fluctuations provide a strong test case for detecting unexpected user behavior in time series data.

```python
# Read the dataset
wikipedia = pd.read_csv("https://datasets-nixtla.s3.amazonaws.com/peyton-manning.csv", parse_dates=["ds"])
wikipedia.head(10)
```

Output

| unique_id | ds                  | y           |
| --------- | ------------------- | ----------- |
| 0         | 2007-12-10T00:00:00 | 9.590761139 |
| 0         | 2007-12-11T00:00:00 | 8.519590316 |
| 0         | 2007-12-12T00:00:00 | 8.183676582 |
| 0         | 2007-12-13T00:00:00 | 8.072467369 |
| 0         | 2007-12-14T00:00:00 | 7.893572073 |
| 0         | 2007-12-15T00:00:00 | 7.783640596 |
| 0         | 2007-12-16T00:00:00 | 8.414052432 |
| 0         | 2007-12-17T00:00:00 | 8.829226354 |
| 0         | 2007-12-18T00:00:00 | 8.382518288 |
| 0         | 2007-12-19T00:00:00 | 8.069655307 |

Visualize the time series using the `plot` method:

```python
nixtla_client.plot(wikipedia)
```

The time series spans several years and shows clear seasonal spikes, likely tied to NFL events, alongside irregular, sharp peaks. These mixed patterns make it hard to judge by eye which spikes are expected and which are truly unusual, making TimeGPT essential for accurate anomaly detection.

## Detect Anomalies with TimeGPT

By default, TimeGPT uses a **99% confidence interval**. This means the model expects 99% of data points to fall within a predicted range based on historical behavior and learned patterns. If a value lies outside this range, it's flagged as an anomaly because it's statistically unlikely given the context.

```python
anomalies_df = nixtla_client.detect_anomalies(
    wikipedia,
    freq="D",
    model="timegpt-1",
)
anomalies_df.head()
```

| unique_id | ds                  | y             | TimeGPT  | TimeGPT-hi-99 | TimeGPT-lo-99 | anomaly |
| --------- | ------------------- | ------------- | -------- | ------------- | ------------- | ------- |
| 0         | 2008-01-10T00:00:00 | 8.2817239904  | 8.224187 | 9.503586      | 6.9447885     | false   |
| 0         | 2008-01-11T00:00:00 | 8.2927988582  | 8.151533 | 9.430932      | 6.8721347     | false   |
| 0         | 2008-01-12T00:00:00 | 8.1991893591  | 8.127243 | 9.406642      | 6.8478446     | false   |
| 0         | 2008-01-13T00:00:00 | 9.9965224185  | 8.917259 | 10.196658     | 7.637861      | false   |
| 0         | 2008-01-14T00:00:00 | 10.1270710071 | 9.002326 | 10.281725     | 7.7229276     | false   |

In this table:

- The **anomaly** column shows whether each data point is considered unusual.
- A value of **False** means the data falls within TimeGPT's expected range and is considered normal.
- A value of **True** means the data lies outside the model's confidence interval and is flagged as a potential anomaly for further investigation.

Plot the anomalies using the `plot` method:

```python
nixtla_client.plot(wikipedia, anomalies_df)
```

The plot reveals several key insights:

- While many anomalies occur during seasonal peaks, only a few are flagged with green dots, showing that TimeGPT can distinguish expected fluctuations from true outliers.
- The gray confidence band covers most of the variation, but the flagged points fall well outside this range, indicating significant deviations.

## Add Exogenous Features

[Exogenous features](https://www.nixtla.io/docs/tutorials-exogenous_variables) are additional inputs that can help explain what's driving changes in the target variable.

Because Peyton Manning's Wikipedia traffic consistently spikes during the NFL season, adding features like month and year helps TimeGPT recognize these seasonal trends and avoid flagging expected increases as anomalies.

```python
anomalies_df_exogenous = nixtla_client.detect_anomalies(
    wikipedia,
    freq="D",
    date_features=["month", "year"],
    date_features_to_one_hot=True,
    model="timegpt-1",
)
```

Plot the relative weights of exogenous features to see how much each one contributes to the model's understanding of the time series.

```python
nixtla_client.weights_x.plot.barh(
    x='features',
    y='weights'
)
```

In this plot:

- Each bar represents how strongly a particular feature (like a specific year or month) influenced TimeGPT's modeling. **Higher bars** indicate more influential features.
- In this dataset, **certain years and months stand out**. This makes sense given the strong seasonal and event-driven patterns in Peyton Manning's page traffic.

Let's compare the number of anomalies between with and without exogenous variables:

```python
# Without exogenous features
print("Number of anomalies without exogenous features:", anomalies_df.anomaly.sum())

# With exogenous features
print("Number of anomalies with exogenous features:", anomalies_df_exogenous.anomaly.sum())
```

Output:

```
Number of anomalies without exogenous features: 89
Number of anomalies with exogenous features: 92
```

Although the increase is small, the model detected more anomalies when exogenous features were included.

By giving the model more context about seasonality and long-term changes, these features sharpened its sense of what 'normal' looks like for different times of year, resulting in a more precise detection of truly unusual behavior. This suggests that month and year provide useful context, helping TimeGPT uncover subtle deviations that wouldn't otherwise be flagged.

## Modifying the Confidence Interval to 70%

You can also control how sensitive TimeGPT is to anomalies by adjusting the [confidence level](https://www.nixtla.io/docs/tutorials-uncertainty_quantification).

```python
anomalies_df_70 = nixtla_client.detect_anomalies(wikipedia, freq="D", level=70)

# Print and compare anomaly counts
print("Number of anomalies with 99% confidence interval:", anomalies_df.anomaly.sum())
print("Number of anomalies with 70% confidence interval:", anomalies_df_70.anomaly.sum())
```

Output:

```
Number of anomalies with 99% confidence interval: 89
Number of anomalies with 70% confidence interval: 505
```

Reducing the confidence interval from 99% to 70% narrows the range of expected values, making the model more sensitive to deviations and resulting in many more anomalies being flagged.

This is useful when you want to flag **more subtle deviations** or prefer a higher sensitivity in anomaly detection, especially for use cases where catching borderline anomalies is more important than minimizing false positives.

## Final Thoughts

In this walkthrough, you've seen how TimeGPT enables scalable, context-aware anomaly detection in time series data. You explored how to:

- Load and visualize real-world time series
- Apply TimeGPT for anomaly detection
- Adjust sensitivity using confidence levels
- Improve model precision with exogenous features like dates and holidays

This setup can be easily adapted for other traffic, sales, or event-based time series datasets
