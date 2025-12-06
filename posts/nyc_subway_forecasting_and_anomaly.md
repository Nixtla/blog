---
title: "NYC Subway Forecasting and Anomaly Detection using Nixtla"
description: "Learn how to forecast ridership and detect anomalies in NYC subway data using Nixtla's StatsForecast and TimeGPT."
image: "/images/nyc_subway_forecasting/title_image.svg"
categories: ["TimeGPT Forecasting", "Anomaly Detection"]
tags:
  - StatsForecast
  - TimeGPT
  - forecasting
  - anomaly detection
  - urban analytics
  - time series
  - AutoARIMA
  - public transportation
author_name: Piero Paialunga
author_image: "/images/authors/piero.jpg"
author_position: Data Scientist
publication_date: 2025-06-12
---

## Introduction

Urban environments are a great natural representation of **real world time series**. The time we leave the house, our commute to work, the road we take when we go to the gym: everything designs a complex time series environment. 

For example, **public transportation systems** handle millions of passengers daily, generating rich time series data that reveals patterns, trends, and unexpected disruptions. Many functional questions of interest arise from this environment:

- **Forecasting demand**: How many riders, customers, or users should we expect next week?
- **Detecting anomalies**: When does unusual behavior signal a problem or opportunity?
- **Understanding seasonality**: How do weekday vs. weekend patterns differ?
- **Planning resources**: Are we allocating staff, inventory, or capacity efficiently?

**Public transit ridership** provides an excellent case study for these patterns. Unlike simple metrics that might grow linearly, transit data exhibits complex characteristics:

- **Strong weekly seasonality**: Weekday ridership can be 2-3x higher than weekends
- **Multiple time scales**: Trends evolve over months while daily patterns remain consistent
- **Clear anomalies**: Service disruptions, weather events, and holidays create detectable deviations

These characteristics appear across industries: 
- In **retail**, customer traffic shows weekly patterns with holiday anomalies. 
- In **manufacturing**, production output follows shift schedules with equipment failure anomalies. 
- In **digital products**, user engagement peaks on specific days with viral event anomalies. 

Understanding how to forecast and detect anomalies in **seasonal, disruption-prone time series** has broad applications beyond transportation.

In this article, we'll use **NYC public transport system ridership data** to demonstrate how **Nixtla's forecasting and anomaly detection ecosystem** can both predict future demand and identify unusual patterns. We'll leverage two powerful tools:

1. [**StatsForecast with AutoARIMA**](https://nixtlaverse.nixtla.io/statsforecast/index.html): Automated statistical forecasting that captures seasonality
2. [**TimeGPT**](https://www.nixtla.io/blog/timegpt-2-announcement): AI-powered anomaly detection that identifies outliers in your data

Let's get started!

## The Workflow

To accomplish both forecasting and anomaly detection, we'll follow this systematic approach:

1. **Load and Explore the Data**: Import NYC MTA ridership data and understand its patterns
2. **Prepare the Time Series**: Transform data into the format Nixtla libraries expect
3. **Forecast with AutoARIMA**: Train on recent data, predict future ridership with confidence intervals
4. **Validate Forecast Accuracy**: Compare predictions against held-out test data
5. **Detect Anomalies with TimeGPT**: Identify unusual patterns in historical ridership
6. **Visualize and Interpret Results**: Present findings with publication-ready plots

The setup is summarized in the following workflow:

![Workflow](/images/workflow.svg)

Let's dive into each step!

## 1. Setup, Data Loading and Data Exploration

First, let's import our libraries and load the NYC MTA ridership data. The open data can be downloaded from the [MTA data source](https://data.ny.gov/Transportation/MTA-Daily-Ridership-Data-2020-2025/vxuj-8kew/about_data). 

However, both the data and the code are included in the [PieroAI/MTATimeSeries](https://github.com/PieroPaialungaAI/MTATimeSeries) GitHub folder. In this folder, we can find the custom module (`NYCSubway`) that handles data loading and provides easy access to Nixtla's forecasting and anomaly detection capabilities.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NYCSubway import DataLoader, TimeSeriesAnalyzer
from plotter import plot_timeseries, set_dark_theme, LIME, CYAN, BLUE, WHITE

# Load the data
loader = DataLoader()
df = loader.get_data()

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Total days: {len(df)} days")
```

**Output:**
```
Data shape: (1776, 15)
Date range: 2020-03-01 to 2025-01-09
Total days: 1776 days
```

The MTA dataset spans from March 2020 to January 2025, capturing nearly 5 years of daily ridership across multiple transit systems:

- Subways
- Buses  
- LIRR (Long Island Rail Road)
- Metro-North
- Access-A-Ride
- Bridges and Tunnels
- Staten Island Railway

```python
# Preview the data - all available columns
print("Available ridership columns:")
for col in DataLoader.RIDERSHIP_COLUMNS:
    print(f"  - {col}")

print(f"\nData preview:")
df[['Date'] + DataLoader.RIDERSHIP_COLUMNS[:3]].head()
```

**Output:**
```
Available ridership columns:
  - Subways: Total Estimated Ridership
  - Buses: Total Estimated Ridership
  - LIRR: Total Estimated Ridership
  - Metro-North: Total Estimated Ridership
  - Access-A-Ride: Total Scheduled Trips
  - Bridges and Tunnels: Total Traffic
  - Staten Island Railway: Total Estimated Ridership

Data preview:
```

| Date                |   Subways: Total Estimated Ridership |   Buses: Total Estimated Ridership |   LIRR: Total Estimated Ridership |
|:--------------------|-------------------------------------:|-----------------------------------:|----------------------------------:|
| 2020-03-01 00:00:00 |                              2212965 |                             984908 |                             86790 |
| 2020-03-02 00:00:00 |                              5329915 |                            2209066 |                            321569 |
| 2020-03-03 00:00:00 |                              5481103 |                            2228608 |                            319727 |
| 2020-03-04 00:00:00 |                              5498809 |                            2177165 |                            311662 |
| 2020-03-05 00:00:00 |                              5496453 |                            2244515 |                            307597 |

Let's explore the data a little bit.

```python
# Analyze weekly patterns in recent data
recent_data = df.tail(60).copy()
recent_data['day_of_week'] = recent_data['Date'].dt.day_name()
recent_data['is_weekend'] = recent_data['Date'].dt.dayofweek >= 5

# Calculate average ridership by day of week
dow_avg = recent_data.groupby('day_of_week')['Subways: Total Estimated Ridership'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

print("Average ridership by day of week (last 60 days):")
print(dow_avg.apply(lambda x: f"{x:,.0f}"))
print(f"\nWeekday average: {recent_data[~recent_data['is_weekend']]['Subways: Total Estimated Ridership'].mean():,.0f}")
print(f"Weekend average: {recent_data[recent_data['is_weekend']]['Subways: Total Estimated Ridership'].mean():,.0f}")
print(f"Ratio (Weekday/Weekend): {recent_data[~recent_data['is_weekend']]['Subways: Total Estimated Ridership'].mean() / recent_data[recent_data['is_weekend']]['Subways: Total Estimated Ridership'].mean():.2f}x")
```

**Output:**
```
Average ridership by day of week (last 60 days):
day_of_week
Monday       3,631,539
Tuesday      3,858,784
Wednesday    3,583,527
Thursday     3,681,107
Friday       3,654,454
Saturday     2,649,918
Sunday       2,091,374
Name: Subways: Total Estimated Ridership, dtype: object

Weekday average: 3,682,506
Weekend average: 2,370,646
Ratio (Weekday/Weekend): 1.55x
```


The ratio between weekday and weekend displays that the data exhibits **strong weekly seasonality**: weekday ridership is dramatically higher than weekends. This is somewhat expected: this load is presumably due to the commuters that work in Manhattan and use the public transportation.


From the following plot, where every color represents a column of the df, we can notice how all three systems exhibit the same weekly pattern. Each system has different magnitudes but shares the fundamental weekday-weekend oscillation.

```python
# Visualize multiple transit systems together
fig, ax = plot_timeseries(
    df,
    columns=[
        'Subways: Total Estimated Ridership',
        'Buses: Total Estimated Ridership',
        'LIRR: Total Estimated Ridership'
    ],
    start_date='2024-01-01',
    title='NYC Transit Ridership Comparison (2024-2025)',
    ylabel='Daily Ridership'
)
plt.savefig('images/title_image.svg')
plt.show()
```

![NYC Transit Ridership Comparison](/images/title_image.svg)

## 3. Preparing Data for Forecasting

We select our target column and set three key parameters:

- **Horizon**: 7 days (one week ahead). This means we are going to predict the next 7 days.

- **Training Window**: 200 days. This means we are only training on the last 200 days (minus the last 7). We train on recent data instead of the full 5 years because early pandemic patterns differ from current ridership.

- **Train/Test Split**: The last 7 days become test data for validation. Training uses the 200 days before that.

```python
# Select target and configure parameters
target_column = 'Subways: Total Estimated Ridership'
horizon = 7
train_window = 210

# Split into train/test
test_df = df[-horizon:].copy()
train_df = df[:-horizon].copy()
actual_train_df = train_df.tail(train_window).copy()

print(f"Training: {len(actual_train_df)} days ({actual_train_df['Date'].min().date()} to {actual_train_df['Date'].max().date()})")
print(f"Testing: {len(test_df)} days ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
```

**Output:**
```
Training: 210 days (2024-06-07 to 2025-01-02)
Testing: 7 days (2025-01-03 to 2025-01-09)
```

## 4. Forecasting with AutoARIMA

Now we train using StatsForecast's AutoARIMA with weekly seasonality (`season_length=7`).
Let's use the 70% of confidence, which is a decent balance of conservative and realistic boundaries.


> From now on, you will need Nixtla API key. Get started [here](https://www.nixtla.io/docs/setup/setting_up_your_api_key)

```python
# Initialize analyzer and generate forecast
api_key = "YOUR_API_KEY"  # Replace with your actual API key
analyzer = TimeSeriesAnalyzer(api_key=api_key)
level = [70]

forecasts = analyzer.forecast(
    train_df,
    target_columns=target_column,
    horizon=horizon,
    level=level,
    train_window=train_window
)

forecasts.head()
```

**Output:**

|   index | unique_id                          | ds                  |   AutoARIMA |   AutoARIMA-lo-70 |   AutoARIMA-hi-70 |
|--------:|:-----------------------------------|:--------------------|------------:|------------------:|------------------:|
|       0 | Subways: Total Estimated Ridership | 2025-01-03 00:00:00 | 3.39341e+06 |       3.01699e+06 |       3.76983e+06 |
|       1 | Subways: Total Estimated Ridership | 2025-01-04 00:00:00 | 2.44614e+06 |       1.99462e+06 |       2.89766e+06 |
|       2 | Subways: Total Estimated Ridership | 2025-01-05 00:00:00 | 2.03367e+06 |       1.55287e+06 |       2.51447e+06 |
|       3 | Subways: Total Estimated Ridership | 2025-01-06 00:00:00 | 3.51542e+06 |       3.02232e+06 |       4.00852e+06 |
|       4 | Subways: Total Estimated Ridership | 2025-01-07 00:00:00 | 3.76477e+06 |       3.26637e+06 |       4.26317e+06 |

The forecast DataFrame contains:
- **`ds`**: Forecast dates
- **`AutoARIMA`**: Point forecast. This is the forecast of the Estimated Ridership. 
- **`AutoARIMA-lo-70`**: Lower bound of 70% confidence interval. This is the lowest number of riders we can expect, with 70% of confidence.
- **`AutoARIMA-hi-70`**: Upper bound of 70% confidence interval. This is the highest number of riders we can expect, with 70% of confidence. 

## 5. Visualizing the Forecast

Let's create two views: a full context view showing training data, and a zoomed view focused on the forecast period.

```python
# Full context view
set_dark_theme()
fig, ax = plt.subplots(figsize=(14, 6))

# Plot training data
ax.plot(actual_train_df['Date'], actual_train_df[target_column],
        label=f'Training Data ({train_window} days)', linewidth=2, color=WHITE, alpha=0.7)

# Plot actual test data
ax.plot(test_df['Date'], test_df[target_column],
        label=f'Actual Test Data ({horizon} days)', linewidth=2, color=LIME)

# Plot forecast
ax.plot(forecasts['ds'], forecasts['AutoARIMA'],
        label='Forecast (AutoARIMA)', linewidth=2, linestyle='--', color=CYAN)

# Plot confidence intervals
ax.fill_between(
    forecasts['ds'],
    forecasts[f'AutoARIMA-lo-{level[0]}'],
    forecasts[f'AutoARIMA-hi-{level[0]}'],
    alpha=0.3,
    color=CYAN,
    label=f'{level[0]}% Confidence Interval'
)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Daily Ridership', fontsize=12)
ax.set_title('NYC Subway Ridership: Forecast vs Actual', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig('images/forecasting.svg')
plt.show()
```

![NYC Subway Ridership Forecast](/images/forecasting.svg)

And the zoomed in version:

```python
# Zoomed view - forecast period only
fig, ax = plt.subplots(figsize=(14, 6))

# Plot actual test data
ax.plot(test_df['Date'], test_df[target_column],
        label=f'Actual Test Data ({horizon} days)', linewidth=3, color=LIME, marker='o', markersize=8)

# Plot forecast
ax.plot(forecasts['ds'], forecasts['AutoARIMA'],
        label='Forecast (AutoARIMA)', linewidth=3, linestyle='--', color=CYAN, marker='s', markersize=8)

# Plot confidence intervals
ax.fill_between(
    forecasts['ds'],
    forecasts[f'AutoARIMA-lo-{level[0]}'],
    forecasts[f'AutoARIMA-hi-{level[0]}'],
    alpha=0.3,
    color=CYAN,
    label=f'{level[0]}% Confidence Interval'
)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Daily Ridership', fontsize=12)
ax.set_title('NYC Subway Ridership: Forecast Period (Zoomed)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('images/forecasting_zoomed.svg')
plt.show()
```

![NYC Subway Ridership Forecast Zoomed](/images/forecasting_zoomed.svg)

From the two plots, we can see that the `Statsforecast` library does a great forecasting job, but we can be more quantitative in the evalution.

## 6. Evaluating Forecast Accuracy

We'll calculate three standard metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Emphasizes larger errors more than MAE  
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error for interpretability

```python
# Calculate accuracy metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test_df[target_column], forecasts['AutoARIMA'])
rmse = np.sqrt(mean_squared_error(test_df[target_column], forecasts['AutoARIMA']))
mape = np.mean(np.abs((test_df[target_column].values - forecasts['AutoARIMA'].values) / 
                       test_df[target_column].values)) * 100

print(f"Forecast Accuracy Metrics:")
print(f"MAE: {mae:,.0f} riders")
print(f"RMSE: {rmse:,.0f} riders")
print(f"MAPE: {mape:.2f}%")
```

**Output:**
```
Forecast Accuracy Metrics:
MAE: 119,925 riders
RMSE: 158,439 riders
MAPE: 4.53%
```

A MAPE of around 5-6% indicates strong forecast accuracy. This means our predictions perform very well for a complex time series with strong weekly seasonality and potential disruptions.

## 7. Multi-Series Forecasting

One of Nixtla's powerful features is the ability to forecast multiple time series simultaneously. This is particularly useful when you need to forecast across different categories, regions, or products. Let's forecast ridership for three transit systems at once: Subways, Buses, and LIRR.

```python
# Forecast multiple transit systems at once
multi_forecasts = analyzer.forecast(
    df,
    target_columns=[
        'Subways: Total Estimated Ridership',
        'Buses: Total Estimated Ridership',
        'LIRR: Total Estimated Ridership'
    ],
    horizon=14,
    level=[70],
    train_window=200
)

print(f"Multi-series forecast shape: {multi_forecasts.shape}")
print(f"Unique series: {multi_forecasts['unique_id'].unique().tolist()}")
multi_forecasts.head(10)
```

**Output:**
```
Multi-series forecast shape: (42, 6)
Unique series: ['Buses: Total Estimated Ridership', 'LIRR: Total Estimated Ridership', 'Subways: Total Estimated Ridership']
```

|   index | unique_id                        | ds                  |        AutoARIMA |   AutoARIMA-lo-70 |   AutoARIMA-hi-70 |
|--------:|:---------------------------------|:--------------------|-----------------:|------------------:|------------------:|
|       0 | Buses: Total Estimated Ridership | 2025-01-10 00:00:00 | 959506           |  806388           |       1.11263e+06 |
|       1 | Buses: Total Estimated Ridership | 2025-01-11 00:00:00 | 619183           |  447223           |  791142           |
|       2 | Buses: Total Estimated Ridership | 2025-01-12 00:00:00 | 496851           |  318918           |  674783           |
|       3 | Buses: Total Estimated Ridership | 2025-01-13 00:00:00 |      1.11759e+06 |  937710           |       1.29748e+06 |
|       4 | Buses: Total Estimated Ridership | 2025-01-14 00:00:00 |      1.19128e+06 |       1.00955e+06 |       1.37302e+06 |
|       5 | Buses: Total Estimated Ridership | 2025-01-15 00:00:00 |      1.15673e+06 |  973237           |       1.34022e+06 |
|       6 | Buses: Total Estimated Ridership | 2025-01-16 00:00:00 |      1.04808e+06 |  862922           |       1.23324e+06 |
|       7 | Buses: Total Estimated Ridership | 2025-01-17 00:00:00 |      1.05944e+06 |  867568           |       1.25131e+06 |
|       8 | Buses: Total Estimated Ridership | 2025-01-18 00:00:00 | 662593           |  467183           |  858002           |
|       9 | Buses: Total Estimated Ridership | 2025-01-19 00:00:00 | 500000           |  302156           |  697843           |

The `TimeSeriesAnalyzer` automatically handles multiple series by creating a unified forecast with the `unique_id` column identifying each system. This approach is efficient because it processes all series in a single call rather than requiring separate forecasts for each system.

## 8. Anomaly Detection with TimeGPT

Beyond forecasting, identifying anomalies is critical for understanding when ridership deviates significantly from expected patterns. Anomalies can signal:

- **Service disruptions**: Track closures, equipment failures
- **Special events**: Concerts, sports games, parades
- **Weather events**: Snowstorms, heat waves  
- **Holidays**: Thanksgiving, Christmas, New Year's

Nixtla's TimeGPT uses deep learning to detect these anomalies automatically. Let's apply it to our subway data.

```python
# Detect anomalies using TimeGPT
anomalies = analyzer.detect_anomalies(
    df,
    target_columns='Subways: Total Estimated Ridership'
)

print(f"Anomaly detection results shape: {anomalies.shape}")
print(f"Number of anomalies detected: {anomalies['anomaly'].sum()}")
print(f"Anomaly rate: {(anomalies['anomaly'].sum() / len(anomalies)) * 100:.2f}%")
anomalies.head(10)
```

**Output:**

| unique_id                          | ds                  |      y |   TimeGPT |    TimeGPT-hi-99 |   TimeGPT-lo-99 | anomaly   |
|:-----------------------------------|:--------------------|-------:|----------:|-----------------:|----------------:|:----------|
| Subways: Total Estimated Ridership | 2020-04-03 00:00:00 | 483357 |    416334 |      1.16844e+06 |         -335768 | False     |
| Subways: Total Estimated Ridership | 2020-04-04 00:00:00 | 282278 |    246368 | 998470           |         -505734 | False     |
| Subways: Total Estimated Ridership | 2020-04-05 00:00:00 | 227303 |    171350 | 923452           |         -580752 | False     |
| Subways: Total Estimated Ridership | 2020-04-06 00:00:00 | 445454 |    477128 |      1.22923e+06 |         -274974 | False     |
| Subways: Total Estimated Ridership | 2020-04-07 00:00:00 | 434986 |    558203 |      1.31031e+06 |         -193899 | False     |
| Subways: Total Estimated Ridership | 2020-04-08 00:00:00 | 427485 |    530036 |      1.28214e+06 |         -222066 | False     |
| Subways: Total Estimated Ridership | 2020-04-09 00:00:00 | 404635 |    552341 |      1.30444e+06 |         -199761 | False     |
| Subways: Total Estimated Ridership | 2020-04-10 00:00:00 | 400275 |    383190 |      1.13529e+06 |         -368912 | False     |
| Subways: Total Estimated Ridership | 2020-04-11 00:00:00 | 257061 |    263966 |      1.01607e+06 |         -488136 | False     |
| Subways: Total Estimated Ridership | 2020-04-12 00:00:00 | 198399 |    227805 | 979907           |         -524297 | False     |

The anomaly detector returns the original time series with an additional `anomaly` column (1 for anomalies, 0 for normal points). Let's visualize these anomalies to understand when unusual ridership patterns occurred.

```python
# Visualize anomalies
set_dark_theme()
fig, ax = plt.subplots(figsize=(14, 6))

# Plot full time series
ax.plot(anomalies['ds'], anomalies['y'], label='Subway Ridership', 
        linewidth=1.5, color=WHITE, alpha=0.7)

# Highlight anomalies
anomaly_points = anomalies[anomalies['anomaly'] == 1]
ax.scatter(anomaly_points['ds'], anomaly_points['y'], 
           color=LIME, s=50, zorder=5, label=f'Anomalies ({len(anomaly_points)})')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Daily Ridership', fontsize=12)
ax.set_title('NYC Subway Ridership - Anomaly Detection with TimeGPT', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig('images/anomalies.svg')
plt.show()
```

![NYC Subway Anomaly Detection](/images/anomalies.svg)

The green points highlight days where ridership significantly deviated from expected patterns. Notice how anomalies cluster around specific periods, likely corresponding to major holidays, weather events, or service disruptions.

## 9. Multi-Series Anomaly Detection

Just like with forecasting, we can detect anomalies across multiple transit systems simultaneously. This helps us understand whether anomalies are system-specific or affect all NYC transportation.

```python
# Detect anomalies across all transit systems
all_anomalies = analyzer.detect_anomalies(df, target_columns='all')

print(f"Multi-series anomaly detection shape: {all_anomalies.shape}")
print(f"\nAnomalies by transit system:")
for series in sorted(all_anomalies['unique_id'].unique()):
    n_anomalies = all_anomalies[
        (all_anomalies['unique_id'] == series) & 
        (all_anomalies['anomaly'] == 1)
    ].shape[0]
    total_points = all_anomalies[all_anomalies['unique_id'] == series].shape[0]
    pct = (n_anomalies / total_points) * 100
    print(f"  {series}: {n_anomalies} anomalies ({pct:.1f}%)")
```

**Output**

```
Anomalies by transit system:
  Access-A-Ride: Total Scheduled Trips: 51 anomalies (2.9%)
  Bridges and Tunnels: Total Traffic: 39 anomalies (2.2%)
  Buses: Total Estimated Ridership: 51 anomalies (2.9%)
  LIRR: Total Estimated Ridership: 37 anomalies (2.1%)
  Metro-North: Total Estimated Ridership: 45 anomalies (2.6%)
  Staten Island Railway: Total Estimated Ridership: 51 anomalies (2.9%)
  Subways: Total Estimated Ridership: 40 anomalies (2.3%)
```

Different transit systems show varying anomaly rates. Systems with more consistent patterns (like LIRR commuter rail) may have fewer anomalies, while systems with more variable usage (like Access-A-Ride paratransit) may show more.

In a real-world scenario, this implementation enables teams to promptly react to system overload or retrospectively analyze anomalies to understand their root causes.

## Conclusions

In this article, we showcased Nixtla's capabilities on a real-world time series: **NYC transit ridership data**. Using **StatsForecast** for forecasting and **TimeGPT** for anomaly detection, we demonstrated how these tools handle complex seasonal patterns, multiple transit systems, and unexpected disruptions.

Here's what we accomplished:

- **Automated forecasting with AutoARIMA**: We trained on 210 days of subway ridership data and predicted the next 7 days with 5% MAPE accuracy. AutoARIMA automatically captured the weekly seasonality (weekday vs. weekend patterns) without manual parameter tuning, making it practical for production systems.

- **Multi-series processing at scale**: By organizing data in panel format (`unique_id`, `ds`, `y`), we forecasted and detected anomalies across all seven NYC transit systems in a single workflow. This approach scales efficiently whether you're monitoring transit lines, retail stores, or manufacturing plants.

- **Confidence intervals for better decisions**: By using AutoARIMA, we don't just output a single point prediction, but also **prediction intervals**. Instead of saying "ridership will be 3.4M," we can say "ridership will likely be between 3.0M and 3.8M with 70% confidence." This range-based approach leads to more confident and actionable decisions for resource planning. 

- **AI-powered anomaly detection**: TimeGPT identified unusual ridership patterns, possibly corresponding to holidays, weather events, and service disruptions. This enables both real-time response to system issues and retrospective analysis of what caused past anomalies.

This workflow applies beyond transportation to any domain with seasonal patterns and unexpected disruptions: retail foot traffic, manufacturing output, website engagement, or energy consumption. Nixtla provides the tools to forecast future demand and detect anomalies in a unified, automated way.

