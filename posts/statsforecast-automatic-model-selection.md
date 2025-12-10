---
title: "Automatic Model Selection with StatsForecast for Time Series Forecasting"
seo_title: Automated Model Selection with StatsForecast
description: "Stop testing statistical models manually. Use StatsForecast to automatically fit AutoARIMA, AutoETS, AutoCES, and AutoTheta models, then select the best performer for each series through cross-validation."
categories: ["Time Series Forecasting"]
tags:
  - StatsForecast
  - automatic model selection
  - AutoARIMA
  - AutoETS
  - cross-validation
image: "/images/statsforecast-automatic-model-selection/featured-image.svg"
author_name: Khuyen Tran
author_image: "/images/authors/khuyen.jpeg"
author_position: Developer Advocate - Nixtla
publication_date: 2025-11-20
---

Imagine you have multiple time series to forecast. Which model should you use for each one? ARIMA? ETS? Theta? Or just a simple Naive baseline?

Using one model for all series is easy but sacrifices accuracy since each series has different patterns. However, manually testing different models for each series creates problems:

- **Time consuming**: Hours spent fitting and comparing individual models
- **Expertise required**: Each algorithm needs different parameter configurations
- **Inconsistent evaluation**: Different validation approaches across models
- **Doesn't scale**: Manual testing for hundreds or thousands of series is impractical

StatsForecast automates this process by fitting multiple statistical models simultaneously, then using cross-validation to select the best performer for each time series.

This article demonstrates how to use StatsForecast's automatic model selection with the M4 hourly competition data, then compares it against TimeGPT, Nixtla's foundation model.

> The source code of this article can be found in the [interactive Jupyter notebook](https://github.com/Nixtla/nixtla_blog/blob/main/examples/notebooks/statsforecast-automatic-model-selection/statsforecast_demo.ipynb).

## Introduction to StatsForecast

[StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/) is an open-source Python library that automates statistical forecasting through:

- **Automatic model selection**: AutoARIMA, AutoETS, AutoCES, and AutoTheta optimize parameters automatically
- **Speed**: 20x faster than pmdarima, leveraging Numba for performance
- **Scale**: Handles hundreds or thousands of series with parallel processing

The library follows scikit-learn's familiar `.fit()` and `.predict()` API pattern.

To install StatsForecast, run:

```bash
pip install statsforecast utilsforecast
```

Additional dependencies for this tutorial:

```bash
pip install pandas numpy
```

## Setup - M4 Hourly Competition Data

Import the necessary libraries:

```python
import numpy as np
import pandas as pd

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoCES, AutoTheta, Naive, SeasonalNaive
from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, smape, mase
```

Load the M4 hourly benchmark dataset, which contains 414 hourly time series ranging from 700 to 960 observations each:

```python
# Load M4 hourly competition data
Y_train_df = pd.read_csv('https://auto-arima-results.s3.amazonaws.com/M4-Hourly.csv')
Y_test_df = pd.read_csv('https://auto-arima-results.s3.amazonaws.com/M4-Hourly-test.csv')

# Convert hour indices to datetime
Y_train_df['ds'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(Y_train_df['ds'], unit='h')
Y_test_df['ds'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(Y_test_df['ds'], unit='h')
```

Sample 8 series for demonstration:

```python
# Randomly select 8 series
n_series = 8
uids = Y_train_df['unique_id'].drop_duplicates().sample(8, random_state=23).values
df_train = Y_train_df.query('unique_id in @uids')
df_test = Y_test_df.query('unique_id in @uids')

print(f"Training observations: {len(df_train)}")
print(f"Test observations: {len(df_test)}")
```

```
Training observations: 6640
Test observations: 384
```

Define evaluation metrics with hourly seasonality:

- **MASE (Mean Absolute Scaled Error)**: Scaled error vs seasonal baseline (< 1.0 = beats baseline, = 1.0 = matches baseline, > 1.0 = worse than baseline)
- **RMSE (Root Mean Squared Error)**: Measures the magnitude of prediction errors
- **SMAPE (Symmetric Mean Absolute Percentage Error)**: Calculates percentage-based accuracy

```python
# Define error metrics with 24-hour seasonality
from functools import partial
hourly_mase = partial(mase, seasonality=24)
metrics = [hourly_mase, rmse, smape]
```

Visualize the selected time series:

```python
# Plot the selected series
plot_series(df_train, df_test.rename(columns={"y": "actual"}), max_ids=4)
```

```chart-multiple
{
  "id": "chart-multiple-1",
  "title": "Selected Series",
  "dataSource": "chart-1.csv",
  "columns": 2,
  "legendConfig": {
    "displaySeries": [{ "name": "Y", "color": "blue-500" }, { "name": "Actual", "color": "cyan-500" }]
  },
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Target (y)" },
  "charts": [
    {
      "id": "chart-inner-1",
      "series": [
        { "column": "y_H51", "name": "y_H51", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "actual_H51", "name": "actual_H51", "type": "line", "color": "cyan-500", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-2",
      "series": [
        { "column": "y_H263", "name": "y_H263", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "actual_H263", "name": "actual_H263", "type": "line", "color": "cyan-500", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-3",
      "series": [
        { "column": "y_H25", "name": "y_H25", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "actual_H25", "name": "actual_H25", "type": "line", "color": "cyan-500", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-4",
      "series": [
        { "column": "y_H69", "name": "y_H69", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "actual_H69", "name": "actual_H69", "type": "line", "color": "cyan-500", "strokeWidth": 1 }
      ]
    }
  ]
}
```

Each series shows different patterns. Some have strong daily cycles, others trend up or down over time, and some are quite volatile. This variety is why selecting the right model for each series matters.

## Baseline Models - Naive and SeasonalNaive

Before diving into complex models, start with simple baselines:

- **Naive**: Uses the last observed value as the forecast
- **SeasonalNaive**: Captures seasonal patterns by repeating values from the previous cycle (24 hours ago for hourly data)

These baselines provide a performance floor. Any sophisticated model should beat these simple approaches.

```python
# Configure baseline models
sf_base = StatsForecast(
    models=[Naive(), SeasonalNaive(season_length=24)],
    freq='H',
    n_jobs=-1
)

# Generate 48-hour forecasts
fcst_base = sf_base.forecast(df=df_train, h=48)

# Merge with test data for evaluation
eval_base = df_test.merge(fcst_base, on=['unique_id', 'ds'])
```

Visualize baseline predictions:

```python
# Plot baseline forecasts
plot_series(df_train, eval_base, max_ids=4, max_insample_length=5*24)
```

```chart-multiple
{
  "id": "chart-multiple-2",
  "title": "Selected Series",
  "dataSource": "chart-2.csv",
  "columns": 2,
  "legendConfig": {
    "displaySeries": [
        { "name": "Y", "color": "blue-500" },
        { "name": "Naive", "color": "purple-500" },
        { "name": "Seasonal", "color": "cyan-500" }
    ]
  },
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Target (y)" },
  "charts": [
    {
      "id": "chart-inner-1",
      "series": [
        { "column": "y_H165", "name": "y_H165", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "naive_H165", "name": "naive_H165", "type": "line", "color": "purple-500", "strokeWidth": 1 },
        { "column": "seasonal_H165", "name": "seasonal_H165", "type": "line", "color": "cyan-500", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-2",
      "series": [
        { "column": "y_H263", "name": "y_H263", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "naive_H263", "name": "naive_H263", "type": "line", "color": "purple-500", "strokeWidth": 1 },
        { "column": "seasonal_H263", "name": "seasonal_H263", "type": "line", "color": "cyan-500", "strokeWidth": 1}
      ]
    },
    {
      "id": "chart-inner-3",
      "series": [
        { "column": "y_H25", "name": "y_H25", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "naive_H25", "name": "naive_H25", "type": "line", "color": "purple-500", "strokeWidth": 1 },
        { "column": "seasonal_H25", "name": "seasonal_H25", "type": "line", "color": "cyan-500", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-4",
      "series": [
        { "column": "y_H299", "name": "y_H299", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "naive_H299", "name": "naive_H299", "type": "line", "color": "purple-500", "strokeWidth": 1 },
        { "column": "seasonal_H299", "name": "seasonal_H299", "type": "line", "color": "cyan-500", "strokeWidth": 1 }
      ]
    }
  ]
}
```

The plot shows 5 days of historical data followed by 48-hour forecasts. The cyan line shows SeasonalNaive following the daily rhythm, while the pink Naive line stays flat at the last value.

Evaluate baseline performance:

```python
# Calculate metrics for baseline models
metrics_base = evaluate(
    df=eval_base,
    train_df=df_train,
    metrics=metrics,
    agg_fn='mean',
).set_index('metric')

metrics_base
```

|       | Naive      | SeasonalNaive |
| ----- | ---------- | ------------- |
| mase  | 8.029174   | 0.993421      |
| rmse  | 179.520049 | 66.529088     |
| smape | 0.252074   | 0.065754      |

The SeasonalNaive model performs significantly better than Naive, achieving MASE close to 1.0. This suggests strong seasonal patterns in the hourly data that repeating values from 24 hours ago captures effectively.

## Advanced Statistical Models

Let's move beyond baselines with more sophisticated models:

- **AutoARIMA**: Handles autocorrelation, trends, and seasonality
- **AutoETS**: Auto-selects exponential smoothing components
- **AutoCES**: Offers flexible cyclical pattern modeling
- **AutoTheta**: Fast, robust forecasting that often wins competitions

These automatic models eliminate manual parameter tuning while ensuring forecasts meet statistical standards for accuracy and reliability.

Configure the automatic models:

```python
# Define automatic statistical models
models = [
    AutoARIMA(season_length=24),
    AutoETS(season_length=24),
    AutoCES(season_length=24),
    AutoTheta(season_length=24)
]
```

Fit all models and generate forecasts in one step:

```python
# Initialize StatsForecast with all models
sf = StatsForecast(
    models=models,
    freq='H',
    n_jobs=-1
)

# Fit and forecast with 90% prediction intervals
fcst_sf_models = sf.forecast(df=df_train, h=48, level=[90])

# Merge with test data
eval_sf_models = df_test.merge(fcst_sf_models, on=['unique_id', 'ds'])
```

The `forecast()` method automatically fits each model to every series, optimizes parameters, and generates predictions with uncertainty intervals in a single function call.

Visualize predictions from all models:

```python
# Plot forecasts from all automatic models
plot_series(df_train, eval_sf_models, max_ids=4, max_insample_length=5*24)
```

```chart-multiple
{
  "id": "chart-multiple-3",
  "title": "Selected Series",
  "dataSource": "chart-3.csv",
  "columns": 2,
  "legendConfig": {
    "displaySeries": [
        { "name": "Y", "color": "blue-500" },
        { "name": "AutoARIMA", "color": "green-500" },
        { "name": "AutoETS", "color": "purple-500" },
        { "name": "CES", "color": "pink-500" },
        { "name": "AutoTheta", "color": "cyan-500" }
    ]
  },
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Target (y)" },
  "charts": [
    {
      "id": "chart-inner-1",
      "series": [
        { "column": "y_H165", "name": "y_H165", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "AutoARIMA_H165", "name": "AutoARIMA_H165", "type": "line", "color": "green-500", "strokeWidth": 1 },
        { "column": "AutoETS_H165", "name": "AutoETS_H165", "type": "line", "color": "purple-500", "strokeWidth": 1 },
        { "column": "CES_H165", "name": "CES_H165", "type": "line", "color": "pink-500", "strokeWidth": 1 },
        { "column": "AutoTheta_H165", "name": "AutoTheta_H165", "type": "line", "color": "cyan-500", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-2",
      "series": [
        { "column": "y_H263", "name": "y_H263", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "AutoARIMA_H263", "name": "AutoARIMA_H263", "type": "line", "color": "green-500", "strokeWidth": 1 },
        { "column": "AutoETS_H263", "name": "AutoETS_H263", "type": "line", "color": "purple-500", "strokeWidth": 1 },
        { "column": "CES_H263", "name": "CES_H263", "type": "line", "color": "pink-500", "strokeWidth": 1 },
        { "column": "AutoTheta_H263", "name": "AutoTheta_H263", "type": "line", "color": "cyan-500", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-3",
      "series": [
        { "column": "y_H25", "name": "y_H25", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "AutoARIMA_H25", "name": "AutoARIMA_H25", "type": "line", "color": "green-500", "strokeWidth": 1 },
        { "column": "AutoETS_H25", "name": "AutoETS_H25", "type": "line", "color": "purple-500", "strokeWidth": 1 },
        { "column": "CES_H25", "name": "CES_H25", "type": "line", "color": "pink-500", "strokeWidth": 1 },
        { "column": "AutoTheta_H25", "name": "AutoTheta_H25", "type": "line", "color": "cyan-500", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-4",
      "series": [
        { "column": "y_H299", "name": "y_H299", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "AutoARIMA_H299", "name": "AutoARIMA_H299", "type": "line", "color": "green-500", "strokeWidth": 1 },
        { "column": "AutoETS_H299", "name": "AutoETS_H299", "type": "line", "color": "purple-500", "strokeWidth": 1 },
        { "column": "CES_H299", "name": "CES_H299", "type": "line", "color": "pink-500", "strokeWidth": 1 },
        { "column": "AutoTheta_H299", "name": "AutoTheta_H299", "type": "line", "color": "cyan-500", "strokeWidth": 1 }
      ]
    }
  ]
}
```

Unlike the simple baselines, these models adapt to the data's complexity and follow the actual patterns much more closely.

Evaluate performance across all models:

```python
# Calculate metrics for automatic models
metrics_sf_models = evaluate(
    df=eval_sf_models,
    metrics=metrics,
    train_df=df_train,
    agg_fn='mean',
).set_index('metric')

metrics_sf_models
```

|       | AutoARIMA | AutoETS    | CES       | AutoTheta |
| ----- | --------- | ---------- | --------- | --------- |
| mase  | 0.803407  | 1.331669   | 0.729921  | 1.868366  |
| rmse  | 71.456734 | 122.784231 | 60.979897 | 65.105242 |
| smape | 0.063307  | 0.075775   | 0.079244  | 0.076261  |

AutoCES achieves the lowest MASE (0.73) and RMSE (60.98), outperforming both AutoARIMA and baseline models. All automatic models beat the Naive baseline, demonstrating that automated parameter optimization works effectively.

Compare all models visually:

```python
# Compare baseline and automatic models
from utils import plot_metric_bar_multi
plot_metric_bar_multi(dfs=[metrics_sf_models, metrics_base], metric='mase')
```

```chart
{
  "id": "chart-1",
  "title": "MASE Comparison Accross Models",
  "dataSource": "chart-4.csv",
  "xAxis": {
    "key": "model"
  },
  "yAxis": {
    "label": "MASE"
  },
  "series": [
    {
      "column": "mase",
      "type": "bar"
    }
  ],
  "showLabels": true
}
```

This bar chart shows MASE (Mean Absolute Scaled Error), where values below 1.0 beat the baseline and values above 1.0 perform worse:

- **Naive (MASE 8.03)**: Performs worst, far below the seasonal baseline
- **AutoTheta (MASE 1.87) and AutoETS (MASE 1.33)**: Above 1.0, don't beat SeasonalNaive
- **AutoARIMA (MASE 0.80) and AutoCES (MASE 0.73)**: Below 1.0, successfully outperform the baseline

## Cross-Validation for Model Selection

While AutoCES performs best on average, different models might work better for individual series. Cross-validation helps identify the optimal model for each time series.

Traditional cross-validation uses random splits, which doesn't work for time series. Randomly shuffling would let the model peek into the future, creating data leakage. StatsForecast fixes this by ensuring models only train on historical data, never future values.

[Time series cross-validation](https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html#cross-validation) uses a rolling window approach:

1. Train all models on initial window (yellow in the visualization below)
2. Forecast the next period (red region) and evaluate forecast accuracy
3. Slide the window forward in time and add new data to training set
4. Repeat: train, forecast, evaluate
5. Compare models across all windows and select the best for each series

Run cross-validation with rolling windows:

```python
# Cross-validation with 2 rolling windows
cv_df = sf.cross_validation(
    df=df_train,
    h=24,          # Forecast next 24 hours
    step_size=24,  # Step forward 24 hours
    n_windows=2    # 2 evaluation windows
)

print(f"Cross-validation results: {cv_df.shape}")
cv_df.head()
```

```bash
Cross-validation results: (384, 11)
```

| unique_id | ds                  | cutoff              | y       | AutoARIMA | AutoETS  | CES      | AutoTheta |
| --------- | ------------------- | ------------------- | ------- | --------- | -------- | -------- | --------- |
| H10       | 2024-01-29 05:00:00 | 2024-01-29 04:00:00 | 14502.0 | 14478.23  | 14512.45 | 14489.12 | 14501.67  |
| H10       | 2024-01-29 06:00:00 | 2024-01-29 04:00:00 | 14547.0 | 14523.78  | 14558.19 | 14534.87 | 14547.42  |
| H10       | 2024-01-29 07:00:00 | 2024-01-29 04:00:00 | 14595.0 | 14571.34  | 14606.01 | 14582.63 | 14595.18  |

Evaluate model performance and select the best for each series:

```python
# Evaluate models using MAE across cross-validation windows
from utils import evaluate_cv, get_best_model_forecast

evaluation_df = evaluate_cv(cv_df, mae)

# Count how many times each model was selected
evaluation_df['best_statsforecast_model'].value_counts().to_frame().reset_index()
```

| best_statsforecast_model | count |
| ------------------------ | ----- |
| AutoARIMA                | 3     |
| AutoETS                  | 2     |
| AutoTheta                | 2     |
| CES                      | 1     |

AutoARIMA was selected as the best model for 3 out of 8 series, while AutoETS and AutoTheta each won 2 series. This demonstrates that different models excel on different time series patterns.

## Best Model Forecasts with Prediction Intervals

After selecting the best model for each series, let's generate final forecasts with uncertainty intervals:

```python
# Extract forecasts from the best model for each series
best_fcst_sf = get_best_model_forecast(fcst_sf_models, evaluation_df)
eval_best_sf = df_test.merge(best_fcst_sf, on=['unique_id', 'ds'])

# Plot forecasts with 90% prediction intervals
plot_series(df_train, eval_best_sf, level=[90], max_insample_length=5*24, max_ids=4)
```

```chart-multiple
{
  "id": "chart-multiple-4",
  "title": "Forecasts with Prediction Intervals",
  "dataSource": "chart-5.csv",
  "columns": 2,
  "legendConfig": {
    "displaySeries": [
        { "name": "Y", "color": "blue-500" },
        { "name": "StatsForecast Model", "color": "cyan-500" },
        { "name": "Model Level 90", "color": "cyan-800" }
    ]
  },
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Target (y)" },
  "charts": [
    {
      "id": "chart-inner-1",
      "series": [
        { "column": "y_H165", "name": "y_H165", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "best_statsforecast_model_H165", "name": "best_statsforecast_model_H165", "type": "line", "color": "cyan-500", "strokeWidth": 1 },
        { "columns": { "high": "best_statsforecast_model-hi-90_H165", "low": "best_statsforecast_model-lo-90_H165" }, "name": "best_statsforecast_model_level_90", "type": "area", "color": "cyan-800", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-2",
      "series": [
        { "column": "y_H263", "name": "y_H263", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "best_statsforecast_model_H263", "name": "best_statsforecast_model_H263", "type": "line", "color": "cyan-500", "strokeWidth": 1 },
        { "columns": { "high": "best_statsforecast_model-hi-90_H263", "low": "best_statsforecast_model-lo-90_H263" }, "name": "best_statsforecast_model_level_90", "type": "area", "color": "cyan-800", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-3",
      "series": [
        { "column": "y_H25", "name": "y_H25", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "best_statsforecast_model_H25", "name": "best_statsforecast_model_H25", "type": "line", "color": "cyan-500", "strokeWidth": 1 },
        { "columns": { "high": "best_statsforecast_model-hi-90_H25", "low": "best_statsforecast_model-lo-90_H25" }, "name": "best_statsforecast_model_level_90", "type": "area", "color": "cyan-800", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-4",
      "series": [
        { "column": "y_H299", "name": "y_H299", "type": "line", "color": "blue-500", "strokeWidth": 1 },
        { "column": "best_statsforecast_model_H299", "name": "best_statsforecast_model_H299", "type": "line", "color": "cyan-500", "strokeWidth": 1 },
        { "columns": { "high": "best_statsforecast_model-hi-90_H299", "low": "best_statsforecast_model-lo-90_H299" }, "name": "best_statsforecast_model_level_90", "type": "area", "color": "cyan-800", "strokeWidth": 1 }
      ]
    }
  ]
}
```

These are the forecasts from the best model for each series. The shaded bands represent 90% prediction intervals:

- **Tighter bands**: More confidence in the forecast
- **Wider bands**: More uncertainty in the forecast

Calculate metrics for the best model selection approach:

```python
# Calculate metrics for best model selection
metrics_sf_best = evaluate(
    df=eval_best_sf,
    train_df=df_train,
    metrics=metrics,
    agg_fn='mean',
).set_index('metric')
```

Compare baseline, individual models, and best model selection:

```python
# Compare baseline, individual models, and best selection
plot_metric_bar_multi(dfs=[metrics_sf_models, metrics_base, metrics_sf_best], metric='mase')
```

```chart
{
  "id": "chart-2",
  "title": "MASE Comparison Accross Model Groups",
  "dataSource": "chart-6.csv",
  "xAxis": {
    "key": "model"
  },
  "yAxis": {
    "label": "MASE"
  },
  "series": [
    {
      "column": "mase",
      "type": "bar"
    }
  ],
  "showLabels": true
}
```

The green bar, which represents the best model selection, has the lowest MASE. By selecting the best model for each series, we get stronger performance than applying a single model to every time series.

## TimeGPT - Foundation Model for Time Series

Now let's try TimeGPT, Nixtla's foundation model for time series forecasting. It's pre-trained on millions of diverse series and requires minimal tuning.

Why TimeGPT?

- Strong out-of-the-box accuracy with minimal tuning
- Handles trend/seasonality/holidays automatically
- Scales to many series with simple APIs

### Setup TimeGPT Client

First, install the Nixtla package:

```bash
pip install nixtla
```

Initialize the TimeGPT client using an API key:

```python
# Import necessary packages
import os
from dotenv import load_dotenv
from nixtla import NixtlaClient

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("NIXTLA_API_KEY")

# Initialize the client
nixtla_client = NixtlaClient(api_key=api_key)
```

This connects to Nixtla's cloud service where the foundation model runs. The setup is simple: just import and authenticate.

### Zero-Shot Forecast with TimeGPT

Zero-shot forecasting means using the pre-trained model directly without any training on your specific data. This approach saves time by eliminating the training step while still leveraging patterns learned from millions of diverse time series.

The code below generates 48-hour forecasts with 80% and 90% prediction intervals in a single API call:

```python
# Simple zero-shot TimeGPT forecast
fcst_timegpt = nixtla_client.forecast(
    df=df_train,
    h=48,         # forecast horizon (next 48 hours)
    freq='H',     # hourly frequency
    level=['80', '90']
)
fcst_timegpt.head()
```

|     | unique_id | ds                  | TimeGPT    | TimeGPT-hi-80 | TimeGPT-hi-90 | TimeGPT-lo-80 | TimeGPT-lo-90 |
| --- | --------- | ------------------- | ---------- | ------------- | ------------- | ------------- | ------------- |
| 0   | H165      | 2024-01-30 05:00:00 | 20.847889  | 26.581411     | 27.379568     | 15.114367     | 14.316211     |
| 1   | H165      | 2024-01-30 06:00:00 | 25.407340  | 34.487167     | 34.707325     | 16.327513     | 16.107355     |
| 2   | H165      | 2024-01-30 07:00:00 | 49.702620  | 75.707430     | 79.375336     | 23.697813     | 20.029905     |
| 3   | H165      | 2024-01-30 08:00:00 | 134.175630 | 152.434750    | 153.382050    | 115.916504    | 114.969210    |
| 4   | H165      | 2024-01-30 09:00:00 | 366.113680 | 379.054170    | 381.407230    | 353.173200    | 350.820130    |

### Fine-Tuned Forecast with TimeGPT

Fine-tuning adapts the pre-trained model to your specific data patterns by running additional training steps on your dataset. This often improves accuracy for domain-specific patterns that differ from the general patterns TimeGPT learned during pre-training.

To fine-tune TimeGPT, simply add the `finetune_steps` parameter. Here we use 10 finetune steps:

```python
# Add finetune steps to make it more accurate
fcst_timegpt_ft = nixtla_client.forecast(
    df=df_train,
    h=48,
    freq='H',
    level=['80', '90'],
    finetune_steps=10
)
```

### TimeGPT-2 - The Latest Foundation Model

[TimeGPT-2](https://www.nixtla.io/blog/timegpt-2-announcement), Nixtla's latest foundation model, brings enhanced forecasting accuracy and performance improvements. The model is currently available by invitation only.

To use TimeGPT-2, initialize the client with the preview API endpoint:

```python
# Initialize client with TimeGPT-2 credentials
nixtla_client = NixtlaClient(
    api_key=api_key,
    base_url='https://api-preview.nixtla.io'
)
```

To use TimeGPT-2, simply add the `model='timegpt-2'` parameter to the API call:

```python
# Forecast with TimeGPT-2
fcst_timegpt_2 = nixtla_client.forecast(
    df=df_train,
    h=48,
    freq='H',
    level=['80', '90'],
    model='timegpt-2'
)

# Merge with test data
eval_tgpt_2 = df_test.merge(fcst_timegpt_2, on=['unique_id', 'ds'])
```

Visualize TimeGPT-2 predictions:

```python
# Plot TimeGPT-2 forecasts
fig = nixtla_client.plot(
    df_train,
    eval_tgpt_2,
    level=['80', '90'],
    max_insample_length=5*24,
    max_ids=4
)
```

```chart-multiple
{
  "id": "chart-multiple-5",
  "title": "TimeGPT-2 - The Latest Foundation Model",
  "dataSource": "chart-7.csv",
  "columns": 2,
  "legendConfig": {
    "displaySeries": [
        { "name": "Y", "color": "blue-800" },
        { "name": "TimeGPT", "color": "pink-400" },
        { "name": "TimeGPT Level 80", "color": "pink-300" },
        { "name": "TimeGPT Level 90", "color": "pink-200" }
    ]
  },
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Target (y)" },
  "charts": [
    {
      "id": "chart-inner-1",
      "series": [
        { "column": "H165_y", "name": "y_H165", "type": "line", "color": "blue-800", "strokeWidth": 1 },
        { "column": "H165_TimeGPT", "name": "H165_TimeGPT", "type": "line", "color": "pink-400", "strokeWidth": 1 },
        { "columns": { "high": "H165_TimeGPT-hi-80", "low": "H165_TimeGPT-lo-80" }, "name": "TimeGPT_level_80", "type": "area", "color": "pink-300", "strokeWidth": 1 },
        { "columns": { "high": "H165_TimeGPT-hi-90", "low": "H165_TimeGPT-lo-90" }, "name": "TimeGPT_level_90", "type": "area", "color": "pink-200", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-2",
      "series": [
        { "column": "H263_y", "name": "y_H263", "type": "line", "color": "blue-800", "strokeWidth": 1 },
        { "column": "H263_TimeGPT", "name": "H263_TimeGPT", "type": "line", "color": "pink-400", "strokeWidth": 1 },
        { "columns": { "high": "H263_TimeGPT-hi-80", "low": "H263_TimeGPT-lo-80" }, "name": "TimeGPT_level_80", "type": "area", "color": "pink-300", "strokeWidth": 1 },
        { "columns": { "high": "H263_TimeGPT-hi-90", "low": "H263_TimeGPT-lo-90" }, "name": "TimeGPT_level_90", "type": "area", "color": "pink-200", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-3",
      "series": [
        { "column": "H25_y", "name": "y_H25", "type": "line", "color": "blue-800", "strokeWidth": 1 },
        { "column": "H25_TimeGPT", "name": "H25_TimeGPT", "type": "line", "color": "pink-400", "strokeWidth": 1 },
        { "columns": { "high": "H25_TimeGPT-hi-80", "low": "H25_TimeGPT-lo-80" }, "name": "TimeGPT_level_80", "type": "area", "color": "pink-300", "strokeWidth": 1 },
        { "columns": { "high": "H25_TimeGPT-hi-90", "low": "H25_TimeGPT-lo-90" }, "name": "TimeGPT_level_90", "type": "area", "color": "pink-200", "strokeWidth": 1 }
      ]
    },
    {
      "id": "chart-inner-4",
      "series": [
        { "column": "H299_y", "name": "y_H299", "type": "line", "color": "blue-800", "strokeWidth": 1 },
        { "column": "H299_TimeGPT", "name": "H299_TimeGPT", "type": "line", "color": "pink-400", "strokeWidth": 1 },
        { "columns": { "high": "H299_TimeGPT-hi-80", "low": "H299_TimeGPT-lo-80" }, "name": "TimeGPT_level_80", "type": "area", "color": "pink-300", "strokeWidth": 1 },
        { "columns": { "high": "H299_TimeGPT-hi-90", "low": "H299_TimeGPT-lo-90" }, "name": "TimeGPT_level_90", "type": "area", "color": "pink-200", "strokeWidth": 1 }
      ]
    }
  ]
}
```

Compare the performance of the different TimeGPT variants:

```python
# Evaluate all three TimeGPT variants
metrics_tgpt = evaluate(
    df=df_test
        .merge(fcst_timegpt.rename(columns={'TimeGPT': 'TimeGPT_zero_shot'}), on=['unique_id', 'ds'])
        .merge(fcst_timegpt_ft.rename(columns={'TimeGPT': 'TimeGPT_finetuned'}), on=['unique_id', 'ds'])
        .merge(fcst_timegpt_2.rename(columns={'TimeGPT': 'TimeGPT_2'}), on=['unique_id', 'ds']),
    train_df=df_train,
    metrics=metrics,
    agg_fn='mean',
).set_index('metric')

metrics_tgpt
```

|       | TimeGPT_zero_shot | TimeGPT_finetuned | TimeGPT_2 |
| ----- | ----------------- | ----------------- | --------- |
| mase  | 1.119019          | 0.831670          | 0.428494  |
| rmse  | 53.228989         | 53.755869         | 27.029288 |
| smape | 0.056423          | 0.064954          | 0.035599  |

The progression shows dramatic improvements:

- **Zero-shot baseline**: TimeGPT-1 achieves MASE of 1.12
- **Fine-tuning benefit**: Reduces error to 0.83 (26% improvement)
- **TimeGPT-2 breakthrough**: Achieves 0.43 (48% better than fine-tuned)
- **No tuning required**: TimeGPT-2 delivers superior accuracy out-of-the-box

### Final Model Comparison

Compare all approaches including statistical models and TimeGPT-2:

```python
# Compare all models including TimeGPT-2
plot_metric_bar_multi(dfs=[metrics_base, metrics_sf_best, metrics_tgpt], metric='mase')
```

```chart
{
  "id": "chart-3",
  "title": "MASE Comparison Accross Model Groups",
  "dataSource": "chart-8.csv",
  "xAxis": {
    "key": "model"
  },
  "yAxis": {
    "label": "MASE"
  },
  "series": [
    {
      "column": "mase",
      "type": "bar"
    }
  ],
  "showLabels": true
}
```

The results show both accuracy and computational performance:

| Model                         | MASE | Inference Time |
| ----------------------------- | ---- | -------------- |
| TimeGPT-2                     | 0.43 | 2.5 s          |
| Best StatsForecast            | 0.71 | 180 s          |
| TimeGPT-1 (10 finetune steps) | 0.83 | 1.7 s          |
| SeasonalNaive                 | 0.99 | -              |
| Naive                         | 8.03 | -              |

**Key findings:**

- **TimeGPT-2 dominates**: Best accuracy (MASE 0.43) while being 72x faster than StatsForecast
- **StatsForecast trades time for interpretability**: Strong results (MASE 0.71) with transparent statistical models
- **Fine-tuning is efficient**: Improves TimeGPT-1 from 1.12 to 0.83 with only 0.5 seconds overhead
- **Model choice matters**: Proper selection reduces errors by 18x compared to Naive baseline

## Conclusion

This article demonstrated two powerful approaches to automatic forecasting:

| Aspect          | TimeGPT                                                                                              | StatsForecast                                                                 |
| --------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Type**        | Foundation model                                                                                     | Classical statistical models                                                  |
| **Speed**       | Fast, no training required: zero-shot or fine-tuning via API                                         | Medium, requires fitting models locally or in batch                           |
| **Scalability** | Handles thousands of series instantly via API, no need for local compute                             | Scales efficiently with local compute using parallel processing (n_jobs, ray) |
| **Accuracy**    | High accuracy result from strong generalization, especially for complex, noisy, or non-seasonal data | Accurate for structured, seasonal, or stable patterns                         |

**Choose TimeGPT when you:**

- Need instant forecasts with minimal setup
- Want to leverage foundation model intelligence for complex patterns
- Prefer API-based forecasting without local infrastructure

**Choose StatsForecast when you:**

- Prefer interpretable statistical methods
- Need full control over model selection
- Want to run everything locally with transparent optimization

## Related Resources

For SQL-native forecasting in Snowflake, see [TimeGPT in Snowflake](https://www.nixtla.io/blog/timegpt_in_snowflake) to generate forecasts directly in your data warehouse without Python or ML infrastructure.

For automated feature engineering and gradient boosting methods, see our [MLforecast guide](https://www.nixtla.io/blog/automated-time-series-feature-engineering-with-mlforecast) which complements the statistical modeling approaches covered in this article.
