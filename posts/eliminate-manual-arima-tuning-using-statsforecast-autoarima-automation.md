---
title: "Eliminate Manual ARIMA Tuning Using StatsForecast AutoARIMA Automation"
description: "Eliminate weeks of manual ARIMA parameter tuning with StatsForecast's AutoARIMA. Automatically select optimal model parameters for 50+ time series with confidence intervals in under 30 minutes."
categories: ["Time Series Forecasting"]
tags:
  - StatsForecast
  - AutoARIMA
  - automatic model selection
  - ARIMA parameters
  - prediction intervals
image: "/images/eliminate-manual-arima-tuning-using-statsforecast-autoarima-automation/autoarima-forecast-confidence-intervals-featured-image.svg"
author_name: Khuyen Tran
author_image: "/images/authors/khuyen.jpeg"
author_position: Developer Advocate - Nixtla
publication_date: 2025-08-26
---

How much time does your team waste on parameter tuning? Manual [ARIMA modeling](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) requires weeks of statistical testing and expertise that doesn't scale beyond single series.

[`AutoARIMA`](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoarima) eliminates manual parameter testing through automated optimization. It handles parameter selection, seasonality detection, and uncertainty quantification automatically. Complete 50-series forecasting in minutes, not weeks.

## Introduction to StatsForecast

StatsForecast provides fast classical statistical models with automatic parameter selection. The library offers:

- **Automatic model selection**: `AutoARIMA`, `AutoETS`, `AutoTheta` with intelligent parameter optimization
- **Scalable processing**: Handle millions of time series with distributed computing and optimized algorithms
- **Unified interface**: Consistent API across different forecasting methods
- **Built-in validation**: Time series cross-validation for reliable model evaluation
- **Prediction intervals**: Confidence intervals for uncertainty quantification

Install StatsForecast with a single command:

```bash
pip install statsforecast
```

For this tutorial, install additional dependencies:

```bash
pip install pandas numpy matplotlib
```

The source code for this tutorial is available on [GitHub](https://github.com/Nixtla/nixtla_blog_examples/tree/main/notebooks/eliminate-manual-arima-tuning-using-statsforecast-autoarima-automation.ipynb).

## Setup - Retail Sales Scenario

Import the necessary libraries and create our retail forecasting scenario:

```python
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
import matplotlib.pyplot as plt
```

We will create a controlled retail dataset with realistic business patterns:

- **5 product categories**: electronics, clothing, home & garden, sports, books
- **4-year timespan**: 2020-2023 monthly sales data (240 observations)
- **Realistic patterns**: trend growth, seasonal cycles, and category-specific scaling factors

View the code to create the dataset in the [source code](https://github.com/Nixtla/nixtla_blog_examples/tree/main/notebooks/eliminate-manual-arima-tuning-using-statsforecast-autoarima-automation.ipynb).

Here is a sample of the dataset:

| unique_id   | ds         | y            |
| ----------- | ---------- | ------------ |
| electronics | 2020-01-01 | 62980.284918 |
| electronics | 2020-02-01 | 65936.371640 |
| electronics | 2020-03-01 | 75810.350968 |
| electronics | 2020-04-01 | 83436.051479 |
| electronics | 2020-05-01 | 72051.214384 |

## AutoARIMA for Automatic Model Selection

Traditional ARIMA modeling requires extensive manual parameter testing. For example, the nested loops below test every ARIMA parameter combination:

- Six nested loops generate all possible `(p,d,q)(P,D,Q)` combinations
- Each iteration instantiates a new ARIMA object
- AIC comparison tracks the lowest information criterion score

The exhaustive parameter search fits 144 models per series (3×2×3×2×2×2 combinations). This time-intensive process doesn't scale to multiple series.

```python
from statsmodels.tsa.arima.model import ARIMA

def manual_arima_approach(data, max_p=2, max_q=2, max_d=2, max_P=1, max_Q=1):
    """Conceptual overview of manual ARIMA selection"""
    best_aic = float('inf')
    best_params = None

    # Test all parameter combinations manually with nested loops
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(2):
                        for Q in range(max_Q + 1):
                            model = ARIMA(
                                data,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, 12),
                            )
                            aic = model.fit().aic

                            if aic < best_aic:
                                best_aic = aic
                                best_params = (p, d, q, P, D, Q)

    return best_params, best_aic
```

```plaintext
Manual ARIMA parameter selection for electronics category:
Testing parameter combinations manually...
==================================================
ARIMA(0,0,0)(0,0,0)[12]: AIC = 1045.73
ARIMA(0,0,0)(0,0,1)[12]: AIC = 1034.88
ARIMA(0,0,0)(0,1,0)[12]: AIC = 777.79
ARIMA(0,0,0)(0,1,1)[12]: AIC = 787.59
ARIMA(0,0,0)(1,0,0)[12]: AIC = 1037.25
... (tested 144 combinations)
Best: ARIMA(0, 1, 0)x(0, 1, 1)[12], AIC = 740.28
```

The manual approach shows rapid AIC improvements from 1045.73 to 777.79 in early iterations, but requires testing all 144 combinations to ensure the optimal model isn't missed.

`AutoARIMA` transforms weeks of manual parameter tuning into seconds of automatic optimization. One simple `fit()` call replaces complex nested loops with blazing-fast algorithms.

```python
# AutoARIMA with constrained search space for fair comparison with manual approach
sf_fair = StatsForecast(
    models=[
        AutoARIMA(
            season_length=12,      # Annual seasonality
            max_p=2, max_q=2,      # Match manual search limits
            max_P=1, max_Q=1,      # Match manual seasonal limits
            seasonal=True          # Enable seasonal components
        )
    ],
    freq='MS'
)
```

Let's compare the performance of the manual approach with the `AutoARIMA` approach with constrained search space.

```python
electronics_series = retail_df[retail_df["unique_id"] == "electronics"]

start_time = time.time()
sf_fair.fit(electronics_series)
fair_execution_time = time.time() - start_time

print(f"Manual approach (statsmodels): {manual_execution_time:.2f}s for 144 combinations")
print(f"AutoARIMA (statsforecast):   {fair_execution_time:.2f}s for 144 combinations")
print(f"Algorithm efficiency gain:   {manual_execution_time/fair_execution_time:.1f}x faster")
```

```plaintext
Manual approach (statsmodels): 7.02s for 144 combinations
AutoARIMA (statsforecast):   0.36s for 144 combinations
Algorithm efficiency gain:   19.5x faster
```

AutoARIMA achieves 19.5x speed improvements over the manual approach.

## Multiple Model Comparison and Ensemble

Compare `AutoARIMA` with other automatic methods to validate performance:

```python
# Configure multiple automatic models for comparison
sf_comparison = StatsForecast(
    models=[
        AutoARIMA(season_length=12, stepwise=True),
        AutoETS(season_length=12),  # Exponential smoothing
        SeasonalNaive(season_length=12)  # Simple seasonal baseline
    ],
    freq='MS'
)
```

Split the data into training and testing sets:

```python
# Split data for training and testing (use 80% for training)
split_date = retail_df["ds"].quantile(0.8)
train_data = retail_df[retail_df["ds"] <= split_date]
test_data = retail_df[retail_df["ds"] > split_date]

print(f"Training period: {train_data['ds'].min()} to {train_data['ds'].max()}")
print(f"Test period: {test_data['ds'].min()} to {test_data['ds'].max()}")
```

Fit all models on the training data:

```python
# Fit all models simultaneously
sf_comparison.fit(train_data)
print("All models fitted for comparison")
```

```plaintext
Training period: 2020-01-01 00:00:00 to 2023-03-01 00:00:00
Test period: 2023-04-01 00:00:00 to 2023-12-01 00:00:00
All models fitted for comparison
```

Generate forecasts from all models:

```python
# Generate forecasts for comparison
forecasts = sf_comparison.predict(h=7)  # 7-month ahead forecasts
print(f"Forecast shape: {forecasts.shape}")
forecasts.head()
```

```plaintext
Forecast shape: (35, 5)
```

| unique_id | ds         | AutoARIMA    | AutoETS      | SeasonalNaive |
| --------- | ---------- | ------------ | ------------ | ------------- |
| books     | 2023-04-01 | 60523.191642 | 55534.897267 | 54350.389275  |
| books     | 2023-05-01 | 61275.648592 | 55534.897267 | 61674.121108  |
| books     | 2023-06-01 | 59179.501052 | 55534.897267 | 44922.018634  |
| books     | 2023-07-01 | 55284.722061 | 55534.897267 | 50806.165985  |
| books     | 2023-08-01 | 48753.993447 | 55534.897267 | 39706.558281  |

## Prediction Intervals and Uncertainty Quantification

[Prediction intervals](https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html#predict) quantify forecast uncertainty by providing upper and lower bounds around point predictions. They answer critical business questions: "What's the worst-case scenario for inventory planning?" and "How confident should we be in these sales projections?"

Unlike point forecasts that give single values, prediction intervals capture the inherent uncertainty in future outcomes. A 95% confidence interval means that if you repeated the forecasting process 100 times, 95 of those intervals would contain the actual future value.

`AutoARIMA` generates prediction intervals automatically through its `predict()` method. To generate prediction intervals, specify confidence levels using the `level` parameter (e.g., `level=[80, 95]`).

First, create and configure the AutoARIMA model for generating prediction intervals:

```python
# Create AutoARIMA model for prediction intervals
sf_auto = StatsForecast(
    models=[AutoARIMA(season_length=12, stepwise=True)],
    freq='MS'
)
```

Next, fit the model on training data and generate forecasts with multiple confidence levels:

```python
# Fit on training data and generate forecasts with multiple confidence levels
sf_auto.fit(train_data)
forecasts_with_intervals = sf_auto.predict(
    h=12,                    # 12-month forecast horizon
    level=[50, 80, 90, 95]  # Multiple confidence levels
)

print(f"Forecast columns: {forecasts_with_intervals.columns.tolist()}")
```

```plaintext
Forecast columns: ['unique_id', 'ds', 'AutoARIMA', 'AutoARIMA-lo-95', 'AutoARIMA-lo-90', 'AutoARIMA-lo-80', 'AutoARIMA-lo-50', 'AutoARIMA-hi-50', 'AutoARIMA-hi-80', 'AutoARIMA-hi-90', 'AutoARIMA-hi-95']
```

Finally, examine the forecast results with confidence intervals for the electronics category:

```python
sample_forecast = forecasts_with_intervals[
    forecasts_with_intervals["unique_id"] == "electronics"
].head()

sample_forecast[["ds", "AutoARIMA", "AutoARIMA-lo-95", "AutoARIMA-hi-95"]]
```

| ds         | AutoARIMA    | AutoARIMA-lo-95 | AutoARIMA-hi-95 |
| ---------- | ------------ | --------------- | --------------- |
| 2023-04-01 | 99350.422989 | 87049.266213    | 111651.579766   |
| 2023-05-01 | 94747.079991 | 82445.923214    | 107048.236767   |
| 2023-06-01 | 97033.851200 | 84732.694424    | 109335.007977   |
| 2023-07-01 | 86302.421665 | 74001.264889    | 98603.578442    |
| 2023-08-01 | 83997.249910 | 71696.093133    | 96298.406686    |

Visualize predictions with uncertainty bands:

```python
# Visualize forecasts with confidence intervals
def plot_forecasts_with_intervals(category_name):
    # Get data
    historical = train_data[train_data["unique_id"] == category_name].tail(24)
    forecast = forecasts_with_intervals[forecasts_with_intervals["unique_id"] == category_name]

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot data
    plt.plot(historical["ds"], historical["y"], label="Historical Sales")
    plt.plot(forecast["ds"], forecast["AutoARIMA"], label="AutoARIMA Forecast")
    plt.fill_between(forecast["ds"], forecast["AutoARIMA-lo-95"], forecast["AutoARIMA-hi-95"],
                     alpha=0.2, label="95% Confidence")
    plt.fill_between(forecast["ds"], forecast["AutoARIMA-lo-80"], forecast["AutoARIMA-hi-80"],
                     alpha=0.3, label="80% Confidence")

    # Labels and legend
    plt.title(f"{category_name.title()} Sales Forecast with Confidence Intervals")
    plt.ylabel("Sales ($)")
    plt.xlabel("Date")
    plt.legend()
    plt.show()

# Plot forecasts for electronics category
plot_forecasts_with_intervals("electronics")
```

![autoarima-forecast-confidence-intervals](/images/eliminate-manual-arima-tuning-using-statsforecast-autoarima-automation/autoarima-forecast-confidence-intervals.svg)

The electronics forecast captures seasonal patterns with confidence intervals that widen over time, reflecting increasing uncertainty in longer-term predictions. AutoARIMA successfully identifies the cyclical sales patterns while providing realistic uncertainty bounds for inventory planning decisions.

## Cross-Validation for Model Evaluation

[Time series cross-validation](https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html#cross-validation) tests model performance by training on historical data and validating on future periods. This approach respects temporal order by using multiple validation windows that simulate how models perform as new data arrives.

`AutoARIMA` integrates cross-validation through the `cross_validation()` method. This automatically handles the complex temporal splitting required for reliable time series validation.

```python
# Perform time series cross-validation
cv_results = sf_auto.cross_validation(
    df=train_data,
    h=6,           # 6-month forecast horizon
    step_size=3,   # Move validation window by 3 months
    n_windows=4    # Use 4 validation windows
)

print(f"Cross-validation results shape: {cv_results.shape}")
cv_results.head()
```

```plaintext
Cross-validation results shape: (120, 5)
```

| unique_id | ds         | cutoff     | y            | AutoARIMA    |
| --------- | ---------- | ---------- | ------------ | ------------ |
| books     | 2022-01-01 | 2021-12-01 | 43018.516004 | 43226.769208 |
| books     | 2022-02-01 | 2021-12-01 | 48841.347642 | 42457.344848 |
| books     | 2022-03-01 | 2021-12-01 | 50980.426686 | 41966.543477 |
| books     | 2022-04-01 | 2021-12-01 | 54350.389275 | 41653.470486 |
| books     | 2022-05-01 | 2021-12-01 | 61674.121108 | 41453.767096 |

Calculate [MAPE (Mean Absolute Percentage Error)](https://nixtlaverse.nixtla.io/statsforecast/src/utils.html#evaluation) for each product category to measure forecasting accuracy:

```python
# Calculate mean absolute percentage error (MAPE) for each category
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

cv_performance = cv_results.groupby("unique_id").apply(
    lambda x: calculate_mape(x["y"], x["AutoARIMA"]),
    include_groups=False
).reset_index(name="MAPE")

print("AutoARIMA Cross-Validation Performance (MAPE):")
print(cv_performance.sort_values("MAPE"))
```

```plaintext
AutoARIMA Cross-Validation Performance (MAPE):
     unique_id       MAPE
4       sports   9.252036
2  electronics  11.865445
3  home_garden  12.132121
1     clothing  13.172707
0        books  14.270602
```

The MAPE values show consistent performance across categories, with errors under 15% demonstrating reliable forecasting capability.

## Complete Retail Forecasting Workflow

Let's put it all together in a complete retail forecasting pipeline, including:

1. Split data for validation
2. Configure AutoARIMA with essential parameters
3. Generate forecasts with confidence intervals
4. Validate performance on holdout data

```python
# Split data for validation
split_date = retail_df["ds"].quantile(0.85)  # Use 85% for training
train_data = retail_df[retail_df["ds"] <= split_date]
test_data = retail_df[retail_df["ds"] > split_date]

print(f"Training data: {len(train_data)} observations")
print(f"Training period: {train_data['ds'].min()} to {train_data['ds'].max()}")
```

```plaintext
Training data: 205 observations
Training period: 2020-01-01 00:00:00 to 2023-05-01 00:00:00
```

Configure the AutoARIMA model:

```python
# Configure AutoARIMA with essential parameters
sf_production = StatsForecast(
    models=[
        AutoARIMA(
            season_length=12,  # Annual seasonality
            stepwise=True,  # Efficient search
            seasonal=True,  # Enable seasonal detection
        )
    ],
    freq='MS'
)

# Fit model on training data
sf_production.fit(train_data)
```

Generate forecasts with confidence intervals:

```python
# Generate forecasts with confidence intervals
forecasts = sf_production.predict(
    h=12,                    # 12-month forecast horizon
    level=[80, 95]          # Confidence levels
)

print(f"Final forecasts shape: {forecasts.shape}")
```

```plaintext
Final forecasts shape: (60, 7)
```

Validate model performance on holdout data:

```python
# Generate predictions for validation period
validation_horizon = len(test_data["ds"].unique())
validation_forecasts = sf_production.predict(h=validation_horizon)

# Calculate validation metrics
validation_mape = calculate_mape(
    test_data["y"].values,
    validation_forecasts["AutoARIMA"].values
)
print(f"Validation MAPE: {validation_mape:.2f}%")
```

```plaintext
Validation MAPE: 22.42%
```

The 22.42% MAPE demonstrates reasonable predictive accuracy across product categories.

## Conclusion

`AutoARIMA` transforms weeks of manual parameter testing into a single, fast function call. This simple automation delivers reliable forecasts in minutes instead of months.

This time savings redirects analyst expertise toward business insights and strategic planning. Teams focus on interpreting forecasts and making data-driven decisions rather than manual model tuning.
