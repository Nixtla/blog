---
title: "Production-Ready Forecasting Pipeline with TimeGPT and Polars"
description: "Learn how TimeGPT's native DataFrame compatibility lets you leverage Polars' blazing-fast performance for time series forecasting without data conversion overhead."
categories: ["Time Series Forecasting"]
tags:
  - TimeGPT
  - Polars
  - zero-shot forecasting
  - DataFrame libraries
  - scalable forecasting
image: "/images/production-ready-forecasting-pipeline-with-timegpt-and-polars/featured_image.png"
author_name: Khuyen Tran
author_image: "/images/authors/khuyen.jpeg"
author_position: Developer Advocate - Nixtla
publication_date: 2025-08-26
---

Have you ever hit memory limits while working with time series data? pandas DataFrames struggle with large datasets due to eager evaluation that loads entire datasets into memory.

[Polars](https://pola-rs.github.io/polars/user-guide/concepts/lazy-vs-eager/) offers a superior alternative with lazy evaluation and memory-efficient columnar storage. TimeGPT supports Polars natively, eliminating conversion overhead between data processing and forecasting.

## Introduction to TimeGPT

[TimeGPT](https://www.nixtla.io) is a foundation model for time series that provides zero-shot forecasting capabilities. Unlike traditional models that require training on your specific data, TimeGPT comes pre-trained on millions of time series patterns and generates predictions instantly through API calls.

The key differentiator is its universal DataFrame support. While most forecasting libraries force you into pandas, TimeGPT works natively with:

- Pandas DataFrames for compatibility
- Polars DataFrames for speed and memory efficiency
- Spark DataFrames for distributed computing
- Any DataFrame implementing the DataFrame Interchange Protocol

The complete source code and Jupyter notebook for this tutorial are available on [GitHub](https://github.com/Nixtla/nixtla_blog_examples/blob/main/notebooks/production-ready-forecasting-pipeline-with-timegpt-and-polars.ipynb). Clone it to follow along!

## Loading Data with Polars

Let's start by loading into a Polars DataFrame the [M4 competition dataset](https://nixtlaverse.nixtla.io/datasetsforecast), which contains over 100,000 time series.

```python
import polars as pl
import pandas as pd
import os
from dotenv import load_dotenv
from nixtla import NixtlaClient
from datasetsforecast.m4 import M4
import time
```

First, load a subset of the M4 hourly dataset to demonstrate the basics:

```python
# Load M4 hourly data
m4_data = M4.load(directory='data/', group='Hourly')
train_df = m4_data[0]
```

Converting pandas DataFrames to Polars is straightforward with `pl.from_pandas()`.

```python
# Convert to Polars for better performance
train_pl = pl.from_pandas(train_df)

print(f"Dataset shape: {train_pl.shape}")
print(train_pl.head())
```

Output:

```bash
Dataset shape: (373372, 3)
shape: (5, 3)
┌───────────┬─────┬───────┐
│ unique_id ┆ ds  ┆ y     │
│ ---       ┆ --- ┆ ---   │
│ str       ┆ i64 ┆ f64   │
╞═══════════╪═════╪═══════╡
│ H1        ┆ 1   ┆ 605.0 │
│ H1        ┆ 2   ┆ 586.0 │
│ H1        ┆ 3   ┆ 586.0 │
│ H1        ┆ 4   ┆ 559.0 │
│ H1        ┆ 5   ┆ 511.0 │
└───────────┴─────┴───────┘
```

Select 10 time series for initial demo.

```python
# Select 10 time series for initial demo
sample_ids = train_pl.select("unique_id").unique().limit(10)["unique_id"].implode()

# Filter the main dataframe using the sample IDs
demo_df = train_pl.filter(pl.col("unique_id").is_in(sample_ids))
```

In this code:

- The `select()` method chooses specific columns, `unique()` removes duplicates, and `limit()` restricts the results.
- `implode()` converts the `unique_id` column to a list.
- `filter()` filters the dataframe using sample IDs, while `is_in()` checks if values exist in the list.

Next, we'll convert the integer timestamps to datetime.

```python
# Convert integer timestamps to datetime
base_datetime = pl.datetime(2020, 1, 1)
demo_long = demo_df.with_columns([
    (base_datetime + pl.duration(hours=pl.col("ds") - 1)).alias("ds")
])

# Keep only required columns and filter out missing values
demo_long = demo_long.select(["unique_id", "ds", "y"]).filter(pl.col("y").is_not_null())
print(demo_long.head())
```

Output:

```bash
shape: (5, 3)
┌───────────┬─────────────────────┬──────┐
│ unique_id ┆ ds                  ┆ y    │
│ ---       ┆ ---                 ┆ ---  │
│ str       ┆ datetime[μs]        ┆ f64  │
╞═══════════╪═════════════════════╪══════╡
│ H188      ┆ 2020-01-01 00:00:00 ┆ 12.4 │
│ H188      ┆ 2020-01-01 01:00:00 ┆ 11.9 │
│ H188      ┆ 2020-01-01 02:00:00 ┆ 11.5 │
│ H188      ┆ 2020-01-01 03:00:00 ┆ 11.2 │
│ H188      ┆ 2020-01-01 04:00:00 ┆ 11.0 │
└───────────┴─────────────────────┴──────┘
```

In this code:

- `pl.datetime()` creates a base datetime and `pl.duration()` calculates time offsets from integer values.
- `pl.col("ds") - 1` converts 1-indexed timestamps to 0-indexed for proper hour calculation.
- `select()` keeps only required columns and `is_not_null()` filters out missing values.

## Performance at Scale: Polars vs Pandas

Before diving into forecasting, let's demonstrate why Polars matters for larger datasets. We'll compare performance between pandas and Polars using the full M4 hourly dataset we already loaded.

Start with creating a timing decorator:

```python
# Create timing decorator for accurate performance measurement
def time_it(n_runs=10):
    """Decorator that runs a function n_runs times and returns average time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            times = []
            result = None
            for _ in range(n_runs):
                start = time.time()
                result = func(*args, **kwargs)
                times.append(time.time() - start)
            avg_time = sum(times) / len(times)
            return result, avg_time
        return wrapper
    return decorator
```

Apply the decorator to the functions and run 100 times:

```python
# Define operations as decorated functions
@time_it(n_runs=100)
def pandas_aggregation(df, ids):
    return (
        df[df["unique_id"].isin(ids)]
        .groupby("unique_id")["y"]
        .agg(["count", "mean", "std"])
    )

@time_it(n_runs=100)
def polars_aggregation(df, ids):
    return df.filter(pl.col("unique_id").is_in(ids)).group_by("unique_id").agg([
        pl.col("y").count().alias("count"),
        pl.col("y").mean().alias("mean"),
        pl.col("y").std().alias("std"),
    ])
```

Compare performance between pandas and Polars with some sample IDs:

```python
# Compare performance with accurate timing
sample_ids = ["H1", "H2", "H3", "H4", "H5"]

pandas_stats, pandas_time = pandas_aggregation(train_df, sample_ids)
polars_stats, polars_time = polars_aggregation(train_pl, sample_ids)

print(f"Pandas: {pandas_time:.4f}s | Polars: {polars_time:.4f}s | Speedup: {pandas_time / polars_time:.1f}x")
```

Output:

```bash
Pandas: 0.0087s | Polars: 0.0027s | Speedup: 3.2x
```

Polars is 3.2x faster than pandas for this operation.

Next, let's compare memory usage between pandas and Polars.

```python
# Compare memory usage
import sys

pandas_memory = sys.getsizeof(train_df) / 1024 / 1024  # MB
polars_memory = train_pl.estimated_size('mb')

print(f"Pandas DataFrame: {pandas_memory:.1f} MB")
print(f"Polars DataFrame: {polars_memory:.1f} MB")
print(f"Memory savings: {((pandas_memory - polars_memory) / pandas_memory * 100):.1f}%")
```

Output:

```bash
Pandas DataFrame: 37.3 MB
Polars DataFrame: 7.0 MB
Memory savings: 81.1%
```

Polars' columnar storage delivers 81% memory savings, enabling larger time series datasets without requiring distributed computing.

## Basic Forecasting

Now let's generate our first forecast using TimeGPT with a Polars DataFrame.

First, we need to initialize the TimeGPT client:

```python
# Load the API key from the .env file
load_dotenv()

# Initialize the TimeGPT client
nixtla_client = NixtlaClient(
    api_key=os.environ['NIXTLA_API_KEY']
)
```

TimeGPT's `forecast()` method accepts Polars DataFrames directly. Key parameters: `h` for forecast horizon, `freq` for data frequency, `time_col` and `target_col` to specify column names.

```python
# Generate forecasts directly from Polars DataFrame
forecast_df = nixtla_client.forecast(
    df=demo_long,
    h=24,  # Forecast 24 hours ahead
    freq='1h',
    time_col='ds',
    target_col='y'
)

print(f"Generated {len(forecast_df)} forecasts for {len(forecast_df['unique_id'].unique())} series")
print(forecast_df.head())
```

Output:

```bash
Generated 240 forecasts for 10 series

shape: (5, 3)
┌───────────┬─────────────────────┬───────────┐
│ unique_id ┆ ds                  ┆ TimeGPT   │
│ ---       ┆ ---                 ┆ ---       │
│ str       ┆ datetime[μs]        ┆ f64       │
╞═══════════╪═════════════════════╪═══════════╡
│ H123      ┆ 2020-02-01 04:00:00 ┆ 1256.7139 │
│ H123      ┆ 2020-02-01 05:00:00 ┆ 1147.082  │
│ H123      ┆ 2020-02-01 06:00:00 ┆ 1034.873  │
│ H123      ┆ 2020-02-01 07:00:00 ┆ 939.2955  │
│ H123      ┆ 2020-02-01 08:00:00 ┆ 829.92975 │
└───────────┴─────────────────────┴───────────┘
```

## Visualizing Forecasts

TimeGPT's built-in plotting functionality makes it easy to visualize both historical data and forecasts. The `plot()` method automatically handles Polars DataFrames and creates professional time series visualizations.

```python
# Plot the forecast with historical data
nixtla_client.plot(
    df=demo_long,
    forecasts_df=forecast_df,
    time_col='ds',
    target_col='y',
    max_insample_length=100  # Show last 100 historical points
)
```

```chart-multiple
{
  "id": "chart-1",
  "title": "Multi-Series Forecast with TimeGPT",
  "dataSource": "chart-1.csv",
  "columns": 2,
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Target (y)" },
  "charts": [
    {
      "id": "chart-1-H148",
      "title": "H148",
      "series": [
        { "column": "H148_y", "name": "Actual", "type": "line" },
        { "column": "H148_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" }
      ]
    },
    {
      "id": "chart-1-H206",
      "title": "H206",
      "series": [
        { "column": "H206_y", "name": "Actual", "type": "line" },
        { "column": "H206_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" }
      ]
    },
    {
      "id": "chart-1-H22",
      "title": "H22",
      "series": [
        { "column": "H22_y", "name": "Actual", "type": "line" },
        { "column": "H22_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" }
      ]
    },
    {
      "id": "chart-1-H240",
      "title": "H240",
      "series": [
        { "column": "H240_y", "name": "Actual", "type": "line" },
        { "column": "H240_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" }
      ]
    },
    {
      "id": "chart-1-H334",
      "title": "H334",
      "series": [
        { "column": "H334_y", "name": "Actual", "type": "line" },
        { "column": "H334_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" }
      ]
    },
    {
      "id": "chart-1-H42",
      "title": "H42",
      "series": [
        { "column": "H42_y", "name": "Actual", "type": "line" },
        { "column": "H42_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" }
      ]
    },
    {
      "id": "chart-1-H59",
      "title": "H59",
      "series": [
        { "column": "H59_y", "name": "Actual", "type": "line" },
        { "column": "H59_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" }
      ]
    },
    {
      "id": "chart-1-H65",
      "title": "H65",
      "series": [
        { "column": "H65_y", "name": "Actual", "type": "line" },
        { "column": "H65_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" }
      ]
    }
  ]
}
```

The plot shows 8 different time series (H314, H188, H355, H390, H406, H414, H277, H76) with clear patterns:

- **Historical data** appears in cyan lines showing various seasonal and trend patterns
- **Forecasts** extend into the future (bright green lines) with different prediction patterns for each series
- **Time series separation** displays each `unique_id` in its own subplot for easy comparison
- **Prediction accuracy**: TimeGPT accurately captures each series' unique patterns, which includes seasonality, steady trends, and declines

## Adding Confidence Intervals

Understanding forecast uncertainty is crucial for inventory planning. TimeGPT provides prediction intervals without additional computation.

To add [confidence intervals](https://www.nixtla.io/docs/forecasting/probabilistic/prediction_intervals), we can use the `level` parameter in the `forecast()` method. `level=[80, 95]` will generate 80% and 95% confidence intervals.

```python
# Add confidence intervals
forecast_with_intervals = nixtla_client.forecast(
    df=demo_long,
    h=24,
    freq='1h',
    time_col='ds',
    target_col='y',
    level=[80, 95]  # 80% and 95% confidence levels
)
```

Next, let's analyze the forecast uncertainty. In the code below, we group by `unique_id` and calculate the mean of the 95% and 80% confidence intervals.

```python
# Analyze forecast uncertainty
uncertainty_stats = forecast_with_intervals.group_by('unique_id').agg([
    (pl.col('TimeGPT-hi-95') - pl.col('TimeGPT-lo-95')).mean().alias('avg_95_interval'),
    (pl.col('TimeGPT-hi-80') - pl.col('TimeGPT-lo-80')).mean().alias('avg_80_interval'),
    pl.col('TimeGPT').std().alias('forecast_volatility')
])

print("Forecast Uncertainty Analysis:")
print(uncertainty_stats)
```

Output:

```bash
Forecast Uncertainty Analysis:
shape: (10, 4)
┌───────────┬─────────────────┬─────────────────┬─────────────────────┐
│ unique_id ┆ avg_95_interval ┆ avg_80_interval ┆ forecast_volatility │
│ ---       ┆ ---             ┆ ---             ┆ ---                 │
│ str       ┆ f64             ┆ f64             ┆ f64                 │
╞═══════════╪═════════════════╪═════════════════╪═════════════════════╡
│ H360      ┆ 68.412796       ┆ 34.549354       ┆ 18.647164           │
│ H38       ┆ 580.504963      ┆ 418.681237      ┆ 531.209935          │
│ H188      ┆ 0.292824        ┆ 0.237932        ┆ 2.876808            │
│ H55       ┆ 58.820318       ┆ 41.017378       ┆ 73.065779           │
│ H277      ┆ 0.471367        ┆ 0.407173        ┆ 3.049113            │
│ H334      ┆ 40.401254       ┆ 26.365327       ┆ 33.550085           │
│ H76       ┆ 329.445405      ┆ 216.768719      ┆ 271.65608           │
│ H390      ┆ 17.135066       ┆ 12.570395       ┆ 4.982014            │
│ H414      ┆ 125.905222      ┆ 80.358579       ┆ 39.360953           │
│ H406      ┆ 55.226392       ┆ 31.820611       ┆ 30.963349           │
└───────────┴─────────────────┴─────────────────┴─────────────────────┘
```

Now let's visualize the forecasts with confidence intervals to see the uncertainty bands:

```python
# Plot forecasts with confidence intervals
nixtla_client.plot(
    df=demo_long,
    forecasts_df=forecast_with_intervals,
    time_col='ds',
    target_col='y',
    max_insample_length=100,
    level=[80, 95]  # Display 80% and 95% confidence intervals
)
```

```chart-multiple
{
  "id": "chart-2",
  "title": "Forecast with Confidence Intervals",
  "dataSource": "chart-2.csv",
  "columns": 2,
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Target (y)" },
  "charts": [
    {
      "id": "chart-2-H148",
      "title": "H148",
      "series": [
        { "column": "H148_y", "name": "Actual", "type": "line" },
        { "column": "H148_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" },
        { "type": "area", "columns": { "high": "H148_TimeGPT-hi-95", "low": "H148_TimeGPT-lo-95" }, "name": "95% CI", "color": "chart-2" },
        { "type": "area", "columns": { "high": "H148_TimeGPT-hi-80", "low": "H148_TimeGPT-lo-80" }, "name": "80% CI", "color": "chart-3" }
      ]
    },
    {
      "id": "chart-2-H206",
      "title": "H206",
      "series": [
        { "column": "H206_y", "name": "Actual", "type": "line" },
        { "column": "H206_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" },
        { "type": "area", "columns": { "high": "H206_TimeGPT-hi-95", "low": "H206_TimeGPT-lo-95" }, "name": "95% CI", "color": "chart-2" },
        { "type": "area", "columns": { "high": "H206_TimeGPT-hi-80", "low": "H206_TimeGPT-lo-80" }, "name": "80% CI", "color": "chart-3" }
      ]
    },
    {
      "id": "chart-2-H22",
      "title": "H22",
      "series": [
        { "column": "H22_y", "name": "Actual", "type": "line" },
        { "column": "H22_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" },
        { "type": "area", "columns": { "high": "H22_TimeGPT-hi-95", "low": "H22_TimeGPT-lo-95" }, "name": "95% CI", "color": "chart-2" },
        { "type": "area", "columns": { "high": "H22_TimeGPT-hi-80", "low": "H22_TimeGPT-lo-80" }, "name": "80% CI", "color": "chart-3" }
      ]
    },
    {
      "id": "chart-2-H240",
      "title": "H240",
      "series": [
        { "column": "H240_y", "name": "Actual", "type": "line" },
        { "column": "H240_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" },
        { "type": "area", "columns": { "high": "H240_TimeGPT-hi-95", "low": "H240_TimeGPT-lo-95" }, "name": "95% CI", "color": "chart-2" },
        { "type": "area", "columns": { "high": "H240_TimeGPT-hi-80", "low": "H240_TimeGPT-lo-80" }, "name": "80% CI", "color": "chart-3" }
      ]
    },
    {
      "id": "chart-2-H334",
      "title": "H334",
      "series": [
        { "column": "H334_y", "name": "Actual", "type": "line" },
        { "column": "H334_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" },
        { "type": "area", "columns": { "high": "H334_TimeGPT-hi-95", "low": "H334_TimeGPT-lo-95" }, "name": "95% CI", "color": "chart-2" },
        { "type": "area", "columns": { "high": "H334_TimeGPT-hi-80", "low": "H334_TimeGPT-lo-80" }, "name": "80% CI", "color": "chart-3" }
      ]
    },
    {
      "id": "chart-2-H42",
      "title": "H42",
      "series": [
        { "column": "H42_y", "name": "Actual", "type": "line" },
        { "column": "H42_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" },
        { "type": "area", "columns": { "high": "H42_TimeGPT-hi-95", "low": "H42_TimeGPT-lo-95" }, "name": "95% CI", "color": "chart-2" },
        { "type": "area", "columns": { "high": "H42_TimeGPT-hi-80", "low": "H42_TimeGPT-lo-80" }, "name": "80% CI", "color": "chart-3" }
      ]
    },
    {
      "id": "chart-2-H59",
      "title": "H59",
      "series": [
        { "column": "H59_y", "name": "Actual", "type": "line" },
        { "column": "H59_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" },
        { "type": "area", "columns": { "high": "H59_TimeGPT-hi-95", "low": "H59_TimeGPT-lo-95" }, "name": "95% CI", "color": "chart-2" },
        { "type": "area", "columns": { "high": "H59_TimeGPT-hi-80", "low": "H59_TimeGPT-lo-80" }, "name": "80% CI", "color": "chart-3" }
      ]
    },
    {
      "id": "chart-2-H65",
      "title": "H65",
      "series": [
        { "column": "H65_y", "name": "Actual", "type": "line" },
        { "column": "H65_TimeGPT", "name": "TimeGPT", "type": "line", "strokeDashArray": "5 5" },
        { "type": "area", "columns": { "high": "H65_TimeGPT-hi-95", "low": "H65_TimeGPT-lo-95" }, "name": "95% CI", "color": "chart-2" },
        { "type": "area", "columns": { "high": "H65_TimeGPT-hi-80", "low": "H65_TimeGPT-lo-80" }, "name": "80% CI", "color": "chart-3" }
      ]
    }
  ]
}
```

The plot reveals key insights about forecast uncertainty:

- **Confidence bands**: The subtle green shaded areas represent 80% (darker) and 95% (lighter) prediction intervals
- **Variable uncertainty**: Series like H360 and H38 show wider bands indicating higher volatility
- **Stable patterns**: H188 and H277 display tighter intervals, suggesting more predictable behavior
- **Forecast horizon effect**: Uncertainty generally increases further into the future for most series

## Cross-Validation for Model Validation

While single forecasts are useful for demonstration, production systems require robust validation. TimeGPT's cross-validation feature lets you test forecast accuracy across multiple time windows.

[Cross-validation](https://www.nixtla.io/docs/forecasting/evaluation/cross_validation) works by creating multiple train-test splits at different points in your time series, simulating how the model would have performed historically.

```python
# Perform cross-validation with multiple windows
cv_results = nixtla_client.cross_validation(
    df=demo_long,  # Convert from Polars to pandas
    h=24,           # 24-hour forecast horizon
    n_windows=3,    # Test on 3 different time periods
    step_size=24,   # Move forward 24 hours between windows
    freq='1h',
    time_col='ds',
    target_col='y'
)

print(cv_results.head())
```

Output:

```bash
┌───────────┬─────────────────────┬─────────────────────┬──────┬───────────┐
│ unique_id ┆ ds                  ┆ cutoff              ┆ y    ┆ TimeGPT   │
│ ---       ┆ ---                 ┆ ---                 ┆ ---  ┆ ---       │
│ str       ┆ datetime[μs]        ┆ datetime[μs]        ┆ f64  ┆ f64       │
╞═══════════╪═════════════════════╪═════════════════════╪══════╪═══════════╡
│ H188      ┆ 2020-02-09 00:00:00 ┆ 2020-02-08 23:00:00 ┆ 18.5 ┆ 18.417282 │
│ H188      ┆ 2020-02-09 01:00:00 ┆ 2020-02-08 23:00:00 ┆ 17.9 ┆ 17.932968 │
│ H188      ┆ 2020-02-09 02:00:00 ┆ 2020-02-08 23:00:00 ┆ 17.5 ┆ 17.614662 │
│ H188      ┆ 2020-02-09 03:00:00 ┆ 2020-02-08 23:00:00 ┆ 17.3 ┆ 17.318157 │
│ H188      ┆ 2020-02-09 04:00:00 ┆ 2020-02-08 23:00:00 ┆ 16.8 ┆ 16.850842 │
└───────────┴─────────────────────┴─────────────────────┴──────┴───────────┘
```

The results include:

- `cutoff`: The point where training data ends for each validation window
- `y`: Actual observed values
- `TimeGPT`: Forecasted values

Now let's calculate error metrics for each time series and sort them by accuracy to identify the best and worst performing forecasts:

```python
# Calculate cross-validation performance metrics
cv_performance = cv_results.with_columns([
    (pl.col('y') - pl.col('TimeGPT')).abs().alias('MAE'),
    ((pl.col('y') - pl.col('TimeGPT'))**2).alias('RMSE')
]).group_by('unique_id').agg([
    pl.col('MAE').mean().alias('avg_MAE'),
    pl.col('RMSE').mean().sqrt().alias('avg_RMSE')
]).sort('avg_MAE')

print("Cross-validation performance summary:")
print(cv_performance)
```

Output:

```bash
Cross-validation performance summary:
shape: (10, 3)
┌───────────┬───────────┬────────────┐
│ unique_id ┆ avg_MAE   ┆ avg_RMSE   │
│ ---       ┆ ---       ┆ ---        │
│ str       ┆ f64       ┆ f64        │
╞═══════════╪═══════════╪════════════╡
│ H188      ┆ 0.081965  ┆ 0.09732    │
│ H277      ┆ 0.132735  ┆ 0.185227   │
│ H390      ┆ 4.577264  ┆ 5.998636   │
│ H360      ┆ 6.798993  ┆ 9.141901   │
│ H334      ┆ 9.595661  ┆ 12.957023  │
│ H406      ┆ 11.129589 ┆ 14.894546  │
│ H414      ┆ 13.771655 ┆ 18.916914  │
│ H55       ┆ 16.83572  ┆ 23.639961  │
│ H76       ┆ 69.668117 ┆ 85.634201  │
│ H38       ┆ 164.39661 ┆ 265.494055 │
└───────────┴───────────┴────────────┘
```

The results show clear performance differences across series:

- **Top performers**: H188 (MAE: 0.08) and H277 (MAE: 0.13) achieve excellent accuracy
- **Challenging series**: H38 and H76 show higher errors, indicating complex seasonal patterns
- **Performance range**: MAE varies from 0.08 to 164, demonstrating TimeGPT adapts to different data characteristics

## Next Steps

Ready to scale your forecasting pipeline? Here are practical next steps:

1. **Try the code examples** with your own data, which starts with 100 series and scale up
2. **Build production pipelines** using Polars for preprocessing and TimeGPT for forecasting
3. **Monitor performance gains** by comparing your current pipeline against the Polars + TimeGPT combination
