---
title: "Automatic Model Selection with StatsForecast for Time Series Forecasting"
description: "Stop testing statistical models manually. Use StatsForecast to automatically fit AutoARIMA, AutoETS, AutoCES, and AutoTheta models, then select the best performer for each series through cross-validation."
categories: ["Time Series Forecasting"]
tags:
  - StatsForecast
  - automatic model selection
  - AutoARIMA
  - AutoETS
  - cross-validation
image: "/images/statsforecast-automatic-model-selection/model-comparison-featured-image.svg"
author_name: Khuyen Tran
author_image: "/images/authors/khuyen.jpeg"
author_position: Developer Advocate - Nixtla
publication_date: 2025-11-18
---

Imagine you have multiple time series to forecast. Which model should you use for each one? ARIMA? ETS? Theta? Or just a simple Naive baseline?

Using one model for all series is easy but sacrifices accuracy since each series has different patterns. However, manually testing different models for each series creates problems:

- **Time consuming**: Hours spent fitting and comparing individual models
- **Expertise required**: Each algorithm needs different parameter configurations
- **Inconsistent evaluation**: Different validation approaches across models
- **Doesn't scale**: Manual testing for hundreds or thousands of series is impractical

StatsForecast automates this process by fitting multiple statistical models simultaneously, then using cross-validation to select the best performer for each time series.

This article demonstrates how to use StatsForecast's automatic model selection with the M4 hourly competition data, then compares it against TimeGPT, Nixtla's foundation model.

> The source code of this article can be found in the [interactive Jupyter notebook](https://github.com/Nixtla/nixtla_blog_examples/blob/main/notebooks/statsforecast-automatic-model-selection.ipynb).

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

```{python}
import numpy as np
import pandas as pd

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoCES, AutoTheta, Naive, SeasonalNaive
from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, smape, mase
```

Load the M4 hourly benchmark dataset:

```{python}
# Load M4 hourly competition data
Y_train_df = pd.read_csv('https://auto-arima-results.s3.amazonaws.com/M4-Hourly.csv')
Y_test_df = pd.read_csv('https://auto-arima-results.s3.amazonaws.com/M4-Hourly-test.csv')

# Convert hour indices to datetime
Y_train_df['ds'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(Y_train_df['ds'], unit='h')
Y_test_df['ds'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(Y_test_df['ds'], unit='h')
```

Sample 8 series for demonstration:

```{python}
# Randomly select 8 series
n_series = 8
uids = Y_train_df['unique_id'].drop_duplicates().sample(8, random_state=23).values
df_train = Y_train_df.query('unique_id in @uids')
df_test = Y_test_df.query('unique_id in @uids')

print(f"Training observations: {len(df_train)}")
print(f"Test observations: {len(df_test)}")
```

```bash
Training observations: 5600
Test observations: 384
```

Define evaluation metrics with hourly seasonality:

```{python}
# Define error metrics with 24-hour seasonality
from functools import partial
hourly_mase = partial(mase, seasonality=24)
metrics = [hourly_mase, rmse, smape]
```

Visualize the selected time series:

```{python}
# Plot the selected series
plot_series(df_train, df_test.rename(columns={"y": "actual"}), max_ids=4)
```

![Selected Time Series](/images/statsforecast-automatic-model-selection/selected-series.svg)

Each series shows different patterns. Some have strong daily cycles, others trend up or down over time, and some are quite volatile. These diverse patterns make model selection important.

## Baseline Models - Naive and SeasonalNaive

Before diving into complex models, start with simple baselines:

- **Naive**: Uses the last observed value as the forecast
- **SeasonalNaive**: Captures seasonal patterns by repeating values from the previous cycle (24 hours ago for hourly data)

These baselines provide a performance floor. Any sophisticated model should beat these simple approaches.

```{python}
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

```{python}
# Plot baseline forecasts
plot_series(df_train, eval_base, max_ids=4, max_insample_length=5*24)
```

![Baseline Forecasts](/images/statsforecast-automatic-model-selection/baseline-forecasts.svg)

The plot shows 5 days of historical data followed by 48-hour forecasts. The cyan line shows SeasonalNaive following the daily rhythm, while the pink Naive line stays flat at the last value.

Evaluate baseline performance:

```{python}
# Calculate metrics for baseline models
metrics_base = evaluate(
    df=eval_base,
    train_df=df_train,
    metrics=metrics,
    agg_fn='mean',
).set_index('metric')

metrics_base
```

|       | Naive     | SeasonalNaive |
|-------|-----------|---------------|
| mase  | 8.029174  | 0.993421      |
| rmse  | 179.520049| 66.529088     |
| smape | 0.252074  | 0.065754      |

The SeasonalNaive model performs significantly better than Naive, achieving MASE close to 1.0. This suggests strong seasonal patterns in the hourly data that repeating values from 24 hours ago captures effectively.

## Advanced Statistical Models

Move beyond baselines with more sophisticated models:

- **AutoARIMA**: Handles autocorrelation, trends, and seasonality
- **AutoETS**: Auto-selects exponential smoothing components
- **AutoCES**: Offers flexible cyclical pattern modeling
- **AutoTheta**: Fast, robust forecasting that often wins competitions

These automatic models eliminate manual parameter tuning while maintaining statistical rigor.

Configure the automatic models:

```{python}
# Define automatic statistical models
models = [
    AutoARIMA(season_length=24),
    AutoETS(season_length=24),
    AutoCES(season_length=24),
    AutoTheta(season_length=24)
]
```

Fit all models and generate forecasts in one step:

```{python}
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

```{python}
# Plot forecasts from all automatic models
plot_series(df_train, eval_sf_models, max_ids=4, max_insample_length=5*24)
```

![StatsForecast Model Predictions](/images/statsforecast-automatic-model-selection/statsforecast-predictions.svg)

Unlike the simple baselines, these models adapt to the data's complexity and follow the actual patterns much more closely.

Evaluate performance across all models:

```{python}
# Calculate metrics for automatic models
metrics_sf_models = evaluate(
    df=eval_sf_models,
    metrics=metrics,
    train_df=df_train,
    agg_fn='mean',
).set_index('metric')

metrics_sf_models
```

|       | AutoARIMA | AutoETS   | CES       | AutoTheta |
|-------|-----------|-----------|-----------|-----------|
| mase  | 0.803407  | 1.331669  | 0.729921  | 1.868366  |
| rmse  | 71.456734 | 122.784231| 60.979897 | 65.105242 |
| smape | 0.063307  | 0.075775  | 0.079244  | 0.076261  |

AutoCES achieves the lowest MASE (0.73) and RMSE (60.98), outperforming both AutoARIMA and baseline models. All automatic models beat the Naive baseline, demonstrating that automated parameter optimization works effectively.

Compare all models visually:

```{python}
# Compare baseline and automatic models
from utils import plot_metric_bar_multi
plot_metric_bar_multi(dfs=[metrics_sf_models, metrics_base])
```

![Model Comparison](/images/statsforecast-automatic-model-selection/model-comparison-bar-chart.svg)

This bar chart shows MASE (Mean Absolute Scaled Error), which measures how each model performs compared to SeasonalNaive. When MASE is less than 1, the model beats the baseline. Naive performs worst with MASE of 8.03. AutoTheta and AutoETS have MASE above 1, meaning they don't beat SeasonalNaive. However, AutoCES and AutoARIMA both achieve MASE below 1, outperforming the baseline.

## Cross-Validation for Model Selection

While AutoCES performs best on average, different models might work better for individual series. Cross-validation helps identify the optimal model for each time series.

Traditional cross-validation uses random splits, which doesn't work for time series. Randomly shuffling would let the model peek into the future, creating data leakage. StatsForecast fixes this by ensuring models only train on historical data, never future values.

[Time series cross-validation](https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html#cross-validation) uses a rolling window approach:

1. Train all models on initial window (yellow in the visualization below)
2. Forecast the next period (red region) and evaluate forecast accuracy
3. Slide the window forward in time and add new data to training set
4. Repeat: train, forecast, evaluate
5. Compare models across all windows and select the best for each series

![Rolling-window cross-validation](https://raw.githubusercontent.com/Nixtla/statsforecast/main/nbs/imgs/ChainedWindows.gif)

Run cross-validation with rolling windows:

```{python}
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

| unique_id | ds                  | cutoff              | y       | AutoARIMA | AutoETS   | CES       | AutoTheta |
|-----------|---------------------|---------------------|---------|-----------|-----------|-----------|-----------|
| H10       | 2024-01-29 05:00:00 | 2024-01-29 04:00:00 | 14502.0 | 14478.23  | 14512.45  | 14489.12  | 14501.67  |
| H10       | 2024-01-29 06:00:00 | 2024-01-29 04:00:00 | 14547.0 | 14523.78  | 14558.19  | 14534.87  | 14547.42  |
| H10       | 2024-01-29 07:00:00 | 2024-01-29 04:00:00 | 14595.0 | 14571.34  | 14606.01  | 14582.63  | 14595.18  |

Evaluate model performance and select the best for each series:

```{python}
# Evaluate models using MAE across cross-validation windows
from utils import evaluate_cv, get_best_model_forecast

evaluation_df = evaluate_cv(cv_df, mae)

# Count how many times each model was selected
evaluation_df['best_statsforecast_model'].value_counts().to_frame().reset_index()
```

| best_statsforecast_model | count |
|--------------------------|-------|
| AutoARIMA                | 3     |
| AutoETS                  | 2     |
| AutoTheta                | 2     |
| CES                      | 1     |

AutoARIMA was selected as the best model for 3 out of 8 series, while AutoETS and AutoTheta each won 2 series. This demonstrates that different models excel on different time series patterns.

## Best Model Forecasts with Prediction Intervals

After selecting the best model for each series, generate final forecasts with uncertainty intervals:

```{python}
# Extract forecasts from the best model for each series
best_fcst_sf = get_best_model_forecast(fcst_sf_models, evaluation_df)
eval_best_sf = df_test.merge(best_fcst_sf, on=['unique_id', 'ds'])

# Plot forecasts with 90% prediction intervals
plot_series(df_train, eval_best_sf, level=[90], max_insample_length=5*24, max_ids=4)
```

![Best Model Forecasts with Intervals](/images/statsforecast-automatic-model-selection/best-model-forecasts.svg)

These are the forecasts from the best model for each series. The shaded bands represent 90% prediction intervals. The interval widths vary across series - tighter bands mean more confidence, wider bands mean more uncertainty.

Evaluate the best model ensemble:

```{python}
# Calculate metrics for best model selection
metrics_sf_best = evaluate(
    df=eval_best_sf,
    train_df=df_train,
    metrics=metrics,
    agg_fn='mean',
).set_index('metric')

metrics_sf_best
```

|       | best_statsforecast_model |
|-------|--------------------------|
| mase  | 0.710028                 |
| rmse  | 66.165369                |
| smape | 0.057053                 |

Selecting the best model for each series improves MASE from 0.73 (best single model) to 0.71, and reduces SMAPE to 0.057. This per-series optimization delivers measurable accuracy gains.

Compare all approaches:

```{python}
# Compare baseline, individual models, and best selection
plot_metric_bar_multi(dfs=[metrics_sf_models, metrics_base, metrics_sf_best])
```

![Complete Model Comparison](/images/statsforecast-automatic-model-selection/complete-comparison.svg)

The green bar, which represents the best model selection, has the lowest MASE. By selecting the best model for each series, we get stronger performance than applying a single model to every time series.

## TimeGPT - Foundation Model for Time Series

Now let's try TimeGPT, Nixtla's foundation model for time series forecasting. It's pre-trained on millions of diverse series and requires minimal tuning.

Why TimeGPT?

- Strong out-of-the-box accuracy with minimal tuning
- Handles trend/seasonality/holidays automatically
- Scales to many series with simple APIs

### Setup TimeGPT Client

Initialize the TimeGPT client using an API key:

```{python}
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

This connects to Nixtla's cloud service where the foundation model runs. The setup is simple - just import and authenticate.

### Zero-Shot Forecast with TimeGPT

Generate forecasts without any fine-tuning, using the pre-trained model as-is:

```{python}
# Simple zero-shot TimeGPT forecast
fcst_timegpt = nixtla_client.forecast(
    df=df_train,
    h=48,         # forecast horizon (next 48 hours)
    freq='H',     # hourly frequency
    level=['80', '90']
)
```

The API handles everything automatically. We request 48-hour forecasts with both 80% and 90% prediction intervals.

### Fine-Tuned Forecast with TimeGPT

Add fine-tuning to adapt the pre-trained model to our specific data:

```{python}
# Add finetune steps to make it more accurate
fcst_timegpt_ft = nixtla_client.forecast(
    df=df_train,
    h=48,
    freq='H',
    level=['80', '90'],
    finetune_steps=10
)
```

With 10 fine-tuning steps, TimeGPT adjusts its parameters based on our 8 series. This often improves accuracy on domain-specific data.

### Evaluate TimeGPT Performance

Compare zero-shot and fine-tuned TimeGPT variants:

```{python}
# Evaluate both TimeGPT variants
metrics_tgpt = evaluate(
    df=df_test
        .merge(fcst_timegpt.rename(columns={'TimeGPT': 'TimeGPT_zero_shot'}), on=['unique_id', 'ds'])
        .merge(fcst_timegpt_ft.rename(columns={'TimeGPT': 'TimeGPT_finetuned'}), on=['unique_id', 'ds']),
    train_df=df_train,
    metrics=metrics,
    agg_fn='mean',
).set_index('metric')

metrics_tgpt
```

|       | TimeGPT_zero_shot | TimeGPT_finetuned |
|-------|-------------------|-------------------|
| mase  | 1.119019          | 0.831670          |
| rmse  | 53.228989         | 53.755869         |
| smape | 0.056423          | 0.064954          |

Both TimeGPT variants achieve excellent performance. Zero-shot gets MASE of 1.12, and fine-tuning improves it to 0.83. These results are competitive with the best StatsForecast models.

Visualize TimeGPT predictions:

```{python}
# Plot TimeGPT forecasts
eval_tgpt_ft = df_test.merge(fcst_timegpt_ft, on=['unique_id', 'ds'])
nixtla_client.plot(df_train, eval_tgpt_ft, level=['80', '90'], max_insample_length=5*24, max_ids=4)
```

![TimeGPT Forecasts](/images/statsforecast-automatic-model-selection/timegpt-forecasts.svg)

The TimeGPT forecasts show smooth, reasonable predictions that capture the underlying patterns. The nested prediction intervals (80% inside 90%) show increasing uncertainty further into the future.

### Final Model Comparison

Compare all approaches including TimeGPT:

```{python}
# Compare all models including TimeGPT
plot_metric_bar_multi(dfs=[metrics_base, metrics_sf_best, metrics_tgpt])
```

![Final Comparison with TimeGPT](/images/statsforecast-automatic-model-selection/final-comparison-timegpt.svg)

TimeGPT fine-tuned (MASE 0.83) comes close to the best StatsForecast model selection (MASE 0.71). Even simple SeasonalNaive (MASE 0.99) beats Naive significantly (MASE 8.03). Model choice matters enormously - the right model can reduce errors by over 10x.

## Conclusion

StatsForecast's automatic model selection eliminates hours of manual model testing and parameter tuning. By fitting multiple statistical models simultaneously and using cross-validation to select the best performer for each series, you get:

- **Automatic parameter optimization** across ARIMA, ETS, CES, and Theta models
- **Per-series model selection** through time series cross-validation
- **Consistent evaluation** with proper temporal splitting
- **Prediction intervals** for uncertainty quantification

Stop testing statistical models manually. With StatsForecast, you can evaluate multiple algorithms, identify the best performer for each time series, and generate reliable forecasts with confidence intervals.

## Related Resources

For foundation model approaches that eliminate parameter tuning entirely, explore [TimeGPT's zero-shot forecasting capabilities](https://nixtla.github.io/web/blog/timegpt_in_snowflake) that achieve strong accuracy without cross-validation.

When working with machine learning methods, our [automated feature engineering guide with MLforecast](https://nixtla.github.io/web/blog/automated-time-series-feature-engineering-with-mlforecast) complements StatsForecast's statistical models with gradient boosting approaches.
