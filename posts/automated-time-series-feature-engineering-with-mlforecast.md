---
title: "Automated Time Series Feature Engineering with MLforecast"
seo_title: Automated Time Series Feature Engineering with MLforecast
description: "Replace hours of custom feature engineering code with MLforecast's automated lag features, rolling statistics, and target transformations for faster, more reliable time series forecasting."
categories: ["Time Series Forecasting"]
tags:
  - MLforecast
  - automated feature engineering
  - lag features
  - target transformations
image: "/images/automated-time-series-feature-engineering-with-mlforecast/automated-feature-engineering-rolling-expanding-comparison.svg"
author_name: Khuyen Tran
author_image: "/images/authors/khuyen.jpeg"
author_position: Developer Advocate - Nixtla
publication_date: 2025-08-26
---

Building effective time series models requires creating dozens of engineered features: lag values from previous periods, rolling averages over different windows, seasonal indicators, and [target transformations](https://nixtlaverse.nixtla.io/mlforecast/target_transforms.html). The traditional approach involves writing hundreds of lines of custom [pandas](https://pandas.pydata.org/) code, handling edge cases manually, and maintaining separate pipelines for each use case.

This manual process creates several problems:

- **Time consuming**: 1-2 hours per model just for feature creation
- **Error prone**: Missing values, incorrect window calculations, transformation reversals
- **Inconsistent**: Different feature implementations across models and teams
- **Maintenance heavy**: Breaking when data schemas or patterns change

MLforecast eliminates this pain with automated feature engineering that's faster, more reliable, and consistent across all your forecasting models.

> Before diving into automated feature engineering, establishing [baseline model performance](https://www.nixtla.io/blog/baseline_forecasts) provides essential context for measuring improvement and avoiding over-engineering.

> The source code of this article can be found in the [interactive Jupyter notebook](https://github.com/Nixtla/nixtla_blog_examples/blob/main/notebooks/automated-time-series-feature-engineering-with-mlforecast.ipynb).

## Introduction to MLforecast

[MLforecast](https://nixtlaverse.nixtla.io/mlforecast/) is Nixtla's machine learning forecasting library that handles the complete time series modeling pipeline:

- **Automated feature engineering**: Lag features, rolling statistics, date features
- **Model training**: Multiple algorithms with unified API
- **Cross-validation**: Time series-aware validation splits
- **Prediction generation**: Forecasts with automatic transformation handling

You define models and features through simple parameters while MLforecast handles the complex implementation details.

To install MLforecast, run:

```bash
pip install mlforecast
```

Other dependencies for the examples in this article:

```bash
pip install pandas numpy lightgbm
```

We'll use [LightGBM](https://lightgbm.readthedocs.io/) for our machine learning models throughout this tutorial.

## Setup - Installation and Basic Configuration

Import the necessary libraries:

```python
import pandas as pd
import numpy as np
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, ExpandingMean
from mlforecast.target_transforms import Differences
import lightgbm as lgb
```

Let's start with a simple e-commerce demand forecasting scenario:

```python
# Generate sample e-commerce sales data
np.random.seed(42)
dates = pd.date_range("2023-01-01", "2024-12-01", freq="D")
products = ["product_1", "product_2", "product_3"]

data = []
for product in products:
    # Create realistic sales patterns with trend and seasonality
    trend = np.linspace(100, 200, len(dates))
    seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly pattern
    noise = np.random.normal(0, 20, len(dates))
    sales = np.maximum(0, trend + seasonal + noise)

    product_data = pd.DataFrame({"unique_id": product, "ds": dates, "y": sales})
    data.append(product_data)

sales_data = pd.concat(data, ignore_index=True)
print(f"Dataset shape: {sales_data.shape}")
sales_data.head()
```

```bash
Dataset shape: (2046, 3)
```

| unique_id | ds         | y          |
| --------- | ---------- | ---------- |
| product_1 | 2023-01-01 | 149.967142 |
| product_1 | 2023-01-02 | 194.064742 |
| product_1 | 2023-01-03 | 156.073594 |
| product_1 | 2023-01-04 | 169.276074 |
| product_1 | 2023-01-05 | 135.228628 |

Now let's configure MLforecast with basic automated features:

```python
# Basic MLforecast configuration with automated features
fcst = MLForecast(
    models=lgb.LGBMRegressor(verbosity=-1),
    freq="D",
    lags=[1, 7, 14],  # Previous day, week, and two weeks
    date_features=["dayofweek", "month"],  # Automatic date features
)

print("Configured features:")
print(f"Lags: {fcst.ts.lags}")
print(f"Date features: {fcst.ts.date_features}")
```

```bash
Configured features:
Lags: [1, 7, 14]
Date features: ['dayofweek', 'month']
```

## Automated Lag Feature Engineering - Replacing Manual Lag Creation

Lag features use previous time periods' values to predict future outcomes. A lag-1 feature contains yesterday's sales value, lag-7 contains last week's value, and so on. These historical values often predict future patterns better than raw timestamps alone.

Creating lag features manually with pandas requires dozens of lines of custom code. Here's what you would typically write manually:

```python
# Traditional manual approach
def create_features_manually(df, lags, date_features):
    """Manual feature creation - replicates MLforecast preprocessing"""
    df_with_features = df.copy()

    # Create lag features with MLforecast naming
    for lag in lags:
        df_with_features[f"lag{lag}"] = df_with_features.groupby("unique_id")[
            "y"
        ].shift(lag)

    # Create date features
    for feature in date_features:
        if feature == "dayofweek":
            df_with_features["dayofweek"] = df_with_features["ds"].dt.dayofweek
        elif feature == "month":
            df_with_features["month"] = df_with_features["ds"].dt.month

    # Remove rows where any lag feature is NaN
    lag_columns = [f"lag{lag}" for lag in lags]
    df_with_features = df_with_features.dropna(subset=lag_columns)

    return df_with_features


# Manual approach demonstration
manual_result = create_features_manually(
    sales_data, lags=[1, 7, 14], date_features=["dayofweek", "month"]
)
```

MLforecast handles all this complexity automatically. The [`preprocess()` method](https://nixtlaverse.nixtla.io/mlforecast/forecast.html#preprocess):

- Reads your lag configuration (`lags=[1, 7, 14]`)
- Creates lag columns using efficient pandas operations
- Adds configured date features automatically
- Filters out rows where lag values cannot be calculated

```python
# MLforecast automated approach
# Lags are created automatically when preprocessing
preprocessed_data = fcst.preprocess(sales_data)

print("Automatically created features:")
print(preprocessed_data.columns.tolist())

# Show lag features for one product
product_sample = preprocessed_data[preprocessed_data["unique_id"] == "product_1"]
print(f"\nLag features for product_1 (first 5 rows):")
print(product_sample[["ds", "y", "lag1", "lag7", "lag14"]].head(5))
```

```bash
Automatically created features:
['unique_id', 'ds', 'y', 'lag1', 'lag7', 'lag14', 'dayofweek', 'month']

Lag features for product_1 (first 5 rows):
           ds           y        lag1        lag7       lag14
14 2023-01-15   67.501643   24.499964  116.348695  109.934283
15 2023-01-16  129.988681   67.501643  130.844944  136.469145
16 2023-01-17  130.775487  129.988681  160.883311  161.985881
17 2023-01-18  130.407705  130.775487  113.854405  152.583356
18 2023-01-19   62.716760  130.407705   70.562647   74.194174
```

## Advanced Lag Features - Rolling Statistics and Expanding Means

Beyond basic lag values, MLforecast can apply [lag transformations](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/lag_transforms_guide.html) to lag features for richer patterns.

Lag transforms work in two steps:

1. Create raw historical values with `lags=[1, 7, 14]`

   - `lag1` = yesterday's exact sales (150 units)
   - `lag7` = last week's exact sales (120 units)

2. Apply statistics to those lag features with `lag_transforms`
   - `RollingMean(window_size=7)` on `lag1` = 7-day average of yesterday's values (145 units)
   - `ExpandingMean()` on `lag7` = growing average of weekly values (from 120 to 135 units over time)

```python
# Enhanced MLforecast with lag transforms
fcst_enhanced = MLForecast(
    models=lgb.LGBMRegressor(verbosity=-1),
    freq="D",
    lags=[1, 7, 14],
    lag_transforms={
        1: [RollingMean(window_size=7)],  # 7-day rolling mean of yesterday's values
        7: [ExpandingMean()],  # Expanding mean of weekly values
    },
    date_features=["dayofweek", "month"],
)

# Process data with enhanced lag features
enhanced_data = fcst_enhanced.preprocess(sales_data)
```

Now let's examine what enhanced features were automatically created and view the transformed data:

```python
print("Enhanced lag features:")
print(enhanced_data.columns.tolist())

# Show enhanced features for one product
enhanced_sample = enhanced_data[enhanced_data["unique_id"] == "product_1"].head(10)
print(f"\nEnhanced features for product_1 (first 5 rows):")
print(
    enhanced_sample[
        ["ds", "y", "rolling_mean_lag1_window_size7", "expanding_mean_lag7"]
    ].head()
)
```

```bash
Enhanced lag features:
['unique_id', 'ds', 'y', 'lag1', 'lag7', 'lag14', 'rolling_mean_lag1_window_size7', 'expanding_mean_lag7', 'dayofweek', 'month']

Enhanced features for product_1 (first 5 rows):
           ds           y  rolling_mean_lag1_window_size7  expanding_mean_lag7
14 2023-01-15   67.501643                       96.400157           111.518814
15 2023-01-16  129.988681                       89.422007           113.666161
16 2023-01-17  130.775487                       89.299684           118.387876
17 2023-01-18  130.407705                       84.998566           117.975743
18 2023-01-19   62.716760                       87.363323           114.024651
```

Let's prepare some data to visualize how these transforms work:

```python
# Prepare data for visualization comparison
product_viz = sales_data[sales_data["unique_id"] == "product_1"].tail(60)
product_viz["rolling_7"] = product_viz["y"].rolling(7).mean()
product_viz["expanding"] = product_viz["y"].expanding().mean()
```

Now create a visualization to compare the different patterns:

```python
# Visualize the different patterns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(product_viz["ds"], product_viz["y"], label="Original Sales", alpha=0.6)
ax.plot(product_viz["ds"], product_viz["rolling_7"], label="7-day Rolling Mean")
ax.plot(product_viz["ds"], product_viz["expanding"], label="Expanding Mean")
ax.legend()
plt.show()
```

```chart
{
  "id": "chart-1",
  "title": "Rolling Mean vs Expanding Mean Comparison",
  "dataSource": "chart-1.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "Sales Units"
  },
  "series": [
    {
      "column": "original_sales",
      "name": "Original Sales",
      "type": "line"
    },
    {
      "column": "rolling_7_mean",
      "name": "7 Day Rolling Mean",
      "type": "line"
    },
    {
      "column": "expanding_mean",
      "name": "Expanding Mean",
      "type": "line"
    }
  ]
}
```

The visualization shows how each transform reveals different patterns:

- **Original sales** (white): Shows all daily fluctuations and noise
- **7-day rolling mean** (cyan): Smooths short-term volatility while following trends closely
- **Expanding mean** (lime): Reveals long-term directional changes, less responsive to recent spikes

## Target Transformations - Automatic Preprocessing and Postprocessing

Target transformations improve forecasting accuracy by preprocessing the target variable. For example, differencing transforms trending sales data from [100, 110, 125, 140] into changes [+10, +15, +15], making patterns easier for models to learn.

MLforecast automatically handles both directions: it applies transformations during training (raw values → differences) and reverses them during prediction (model output → original scale). This eliminates the error-prone manual process of transformation reversal.

```python
# Configure MLforecast with target transformations
fcst_with_transforms = MLForecast(
    models=lgb.LGBMRegressor(verbosity=-1),
    freq="D",
    target_transforms=[Differences([1])],  # First difference transformation
    date_features=["dayofweek", "month"],
)

# Preprocessing automatically applies transformations
preprocessed_with_transforms = fcst_with_transforms.preprocess(sales_data)
```

Let's examine the transformed features and see how the target variable has been processed:

```python
print("Features with transformations:")
print(preprocessed_with_transforms.columns.tolist())

# Show transformation results
sample_transformed = preprocessed_with_transforms[
    preprocessed_with_transforms["unique_id"] == "product_1"
].head(10)

print(f"\nTransformed features for product_1:")
sample_transformed[["ds", "y"]].head()
```

```bash
Features with transformations:
['unique_id', 'ds', 'y', 'dayofweek', 'month']

Transformed features for product_1:
|    | ds         | y         |
|----|------------|-----------|
| 15 | 2023-01-16 | 62.487037 |
| 16 | 2023-01-17 | 0.786807  |
| 17 | 2023-01-18 | -0.367782 |
| 18 | 2023-01-19 | -67.690945|
| 19 | 2023-01-20 | -36.994944|

```

The `target_transforms=[Differences([1])]` transforms the `y` column in-place, converting raw sales values into differences. Notice how the `y` values are now small positive/negative changes rather than the original 100-200 range sales figures.

## Cross-Validation for Time Series - Proper Model Evaluation

Standard cross-validation uses random data splits, creating data leakage by training on future data:

- Train: Jan, Mar, May, Jul → Test: Feb, Apr, Jun
- Problem: Uses July data to predict February (impossible in real forecasting)

MLforecast's [`cross_validation()` method](https://nixtlaverse.nixtla.io/mlforecast/forecast.html#cross_validation) creates multiple training/validation splits that respect temporal order. Each validation window trains on all historical data up to a cutoff date, then tests predictions on the following period. For example:

- Window 1: Train Jan-Mar → Test Apr
- Window 2: Train Jan-Apr → Test May
- Window 3: Train Jan-May → Test Jun

The parameters control the validation setup:

- `n_windows=3`: Creates 3 separate validation periods
- `h=7`: Forecasts 7 days ahead for each window
- `step_size=7`: Moves each window forward by 7 days

Let's set up time series cross-validation with a simplified model:

```python
# Fit the model for cross-validation
fcst_cv = MLForecast(
    models=lgb.LGBMRegressor(verbosity=-1),
    freq="D",
    lags=[7, 14],
    lag_transforms={7: [RollingMean(window_size=14)]},
    date_features=["dayofweek"],
)

# Time series cross-validation with multiple windows
cv_results = fcst_cv.cross_validation(
    df=sales_data,
    n_windows=3,  # Number of validation windows
    h=7,  # Forecast horizon (7 days)
    step_size=7,  # Step between windows
)

print("Cross-validation results shape:", cv_results.shape)
print("\nCV results sample:")
print(cv_results.head(5))
```

```bash
Cross-validation results shape: (63, 5)

CV results sample:
   unique_id         ds     cutoff           y  LGBMRegressor
0  product_1 2024-11-11 2024-11-10  250.667873     223.959992
1  product_1 2024-11-12 2024-11-10  223.451074     232.576272
2  product_1 2024-11-13 2024-11-10  208.632353     186.442924
3  product_1 2024-11-14 2024-11-10  185.664733     171.521580
4  product_1 2024-11-15 2024-11-10  124.525334     146.330637
```

The results show 63 total predictions (3 products × 3 windows × 7 days = 63 rows). Each row contains the actual sales value (`y`) and the model's prediction (`LGBMRegressor`) for a specific product and date. Notice how the predictions are reasonably close to actual values, indicating the model is learning meaningful patterns from the lag and date features.

The cross-validation results show predictions for each validation window. To evaluate model performance, we need to calculate error metrics across all windows and products:

```python
# Evaluate performance across windows
from mlforecast.utils import PredictionIntervals
import numpy as np


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


cv_summary = (
    cv_results.groupby(["unique_id", "cutoff"])
    .apply(
        lambda x: mean_absolute_error(x["y"], x["LGBMRegressor"]), include_groups=False
    )
    .reset_index(name="mae")
)

print(f"\nMAE by product and validation window:")
print(cv_summary.head(5))
```

```bash
MAE by product and validation window:
   unique_id     cutoff        mae
0  product_1 2024-11-10  17.970560
1  product_1 2024-11-17  13.236885
2  product_1 2024-11-24  12.433266
3  product_2 2024-11-10  24.969107
4  product_2 2024-11-17  18.825011
```

The MAE values show consistent performance across different validation windows, with errors around 12-25 units. This indicates the model generalizes well across time periods rather than overfitting to specific patterns.

> Once you've mastered automated feature engineering, apply similar time series techniques to [anomaly detection workflows](https://nixtla.io/blog/anomaly_detection) for comprehensive data monitoring.

## Complete Automated Workflow - End-to-End Pipeline Without Manual Features

Now let's put all the concepts together in a complete workflow. This MLforecast configuration combines the lag features, transformations, and cross-validation techniques we've explored:

```python
# Complete automated MLforecast workflow
final_fcst = MLForecast(
    models=[
        lgb.LGBMRegressor(verbosity=-1, random_state=42),
    ],
    freq="D",
    lags=[1, 7, 14, 21],  # Multiple lag periods
    lag_transforms={
        1: [RollingMean(window_size=7), ExpandingMean()],  # Short-term patterns
        7: [RollingMean(window_size=14)],  # Weekly patterns
    },
    target_transforms=[Differences([1])],  # Handle trend
    date_features=["dayofweek", "month", "quarter"],  # Seasonal features
    num_threads=2,  # Parallel processing
)
```

Next, let's prepare our data for training and testing:

```python
# Split data for training and testing
split_date = "2024-11-01"
train_data = sales_data[sales_data["ds"] < split_date]
test_data = sales_data[sales_data["ds"] >= split_date]

print(f"Training data: {train_data.shape}")
print(f"Test data: {test_data.shape}")

# Fit the model (automatically creates features and trains)
final_fcst.fit(train_data)
```

```bash
Training data: (2010, 3)
Test data: (93, 3)
```

With the model trained, we can generate forecasts that automatically apply and reverse all transformations:

```python
# Generate forecasts (automatically applies transformations and reverses them)
forecasts = final_fcst.predict(h=30)  # 30-day forecast

print(f"\nForecast results:")
print(forecasts.head(5))
```

```bash
Forecast results:
   unique_id         ds  LGBMRegressor
0  product_1 2024-11-01     142.816836
1  product_1 2024-11-02     158.925252
2  product_1 2024-11-03     196.809486
3  product_1 2024-11-04     236.261782
4  product_1 2024-11-05     246.399197
```

Let's visualize how well our automated predictions align with actual sales patterns:

```python
# Visualize forecast vs actual values
import matplotlib.pyplot as plt

# Get actual and forecast data for one product
viz_data = sales_data[sales_data["unique_id"] == "product_1"].tail(60)
forecast_data = forecasts[forecasts["unique_id"] == "product_1"]
```

Now visualize how our automated predictions compare to actual sales:

```python
# Compare predictions vs actual sales
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(viz_data["ds"], viz_data["y"], label="Actual Sales")
ax.plot(forecast_data["ds"], forecast_data["LGBMRegressor"], label="Predictions")
ax.axvline(pd.Timestamp("2024-11-01"), linestyle="--", alpha=0.7, label="Train/Test Split")
ax.legend()
plt.show()
```

```chart
{
  "id": "chart-2",
  "title": "MLforecast Predictions vs Actual Sales",
  "dataSource": "chart-2.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": ""
  },
  "series": [
    {
      "column": "actual_sales",
      "name": "Actual Sales",
      "type": "line"
    },
    {
      "column": "mlforecast_predictions",
      "name": "MLforecast Predictions",
      "type": "line"
    }
  ],
  "thresholds": {
    "enabled": true,
    "column": "threshold",
    "label": "Anomaly Time Step"
  }
}
```

The visualization shows how MLforecast's automated predictions align with actual sales patterns. The model successfully captures trends and seasonality using only the automatically generated features.

Finally, let's examine which automatically created features were most important for the model's predictions:

```python
# Show feature importance (automatically created features)
feature_importance = final_fcst.models_["LGBMRegressor"].feature_importances_
feature_names = final_fcst.ts.features

importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": feature_importance}
).sort_values("importance", ascending=False)

print(f"\nTop 10 most important automatically created features:")
print(importance_df.head(10))
```

```bash
Top 10 most important automatically created features:
                           feature  importance
4   rolling_mean_lag1_window_size7         458
1                             lag7         420
3                            lag21         409
6  rolling_mean_lag7_window_size14         387
5              expanding_mean_lag1         363
2                            lag14         344
0                             lag1         322
7                        dayofweek         183
8                            month         114
9                          quarter           0
```

The rolling mean transformations dominate the top features, with `rolling_mean_lag1_window_size7` being most important. This shows MLforecast's automated feature engineering created more predictive features than raw lag values alone.

## Conclusion

MLforecast eliminates the manual feature engineering bottleneck in time series forecasting. By replacing hundreds of lines of custom code with simple declarative configuration, you get:

- **80% time reduction** in feature engineering
- **Consistent features** across all models and environments
- **Automatic handling** of edge cases and missing values
- **Built-in optimizations** for performance and reliability

Stop spending hours on manual feature engineering. With MLforecast, you can focus on the valuable parts of forecasting: understanding your data, interpreting results, and making better business decisions.

## Related Resources

For production-scale implementations, consider [TimeGPT's performance advantages](https://nixtla.io/blog/timegpt_in_snowflake) when deploying automated forecasting pipelines at enterprise scale.

For different forecasting scenarios, explore [multi-horizon forecasting approaches](https://nixtla.io/blog/multilevelforecasting) that complement MLforecast's machine learning methods. When working with sparse or irregular data patterns, our [intermittent demand forecasting guide](https://nixtla.io/blog/intermittent_demand) provides specialized techniques.
