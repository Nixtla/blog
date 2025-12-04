---
title: "Supercharge Your Sales Forecasts: A Complete Guide to Exogenous Variables in MLForecast"
seo_title: Exogenous Variables in MLForecast for Sales
description: Learn how to incorporate external factors like prices, promotions, and calendar patterns into your time series forecasts using MLForecast's exogenous variables.
image: "/images/mlforecast-exogenous-variables/calendar-features-forecast.svg"
categories: ["MLForecast"]
tags:
  - mlforecast
  - time-series
  - forecasting
  - python
  - lightgbm
  - nixtla
author_name: Khuyen Tran
author_image: "/images/authors/khuyen.jpeg"
author_position: Developer Advocate - Nixtla
publication_date: 2025-12-05
---

## Introduction

Time series forecasting rarely depends on historical values alone. External variables, such as prices, promotions, and calendar events, capture the context that shapes your predictions.

These variables aren't just "extra columns." They fall into three distinct categories that determine how you use them:

- **Static**: Store ID, product category. Constant, replicated across time.
- **Dynamic**: Prices, promotions. Time-varying but known ahead.
- **Calendar**: Weekday, holiday. Derived from the timestamp itself.

Mishandle these categories and you'll either leak future data or waste predictive signal. Traditional approaches require you to engineer each type manually.

[MLForecast](https://github.com/Nixtla/mlforecast) simplifies this workflow with a unified API that handles all three types of exogenous variables automatically. You specify which columns are static, provide future values for dynamic features, and let the library handle the rest.

::: {.callout-note appearance="simple"}
**Get the Code**: The complete source code and Jupyter notebook for this tutorial are available on [GitHub](https://github.com/Nixtla/blog/blob/main/examples/notebooks/mlforecast-exogenous-variables.ipynb). Clone it to follow along!
:::

## Introduction to MLForecast Exogenous Variables

MLForecast brings machine learning models to time series forecasting. You can use LightGBM, XGBoost, scikit-learn regressors, or any model with a fit/predict interface. The library handles feature engineering, lag creation, and multi-series alignment automatically.

When working with [exogenous variables](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/exogenous_features.html), MLForecast distinguishes between two categories:

- **[Static features](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/exogenous_features.html)** stay constant across a series: store metadata, product categories, geographic regions.
- **[Dynamic features](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/exogenous_features.html)** vary over time but are known ahead: prices, promotional flags, weather forecasts.

In the following sections, we'll walk through each category and how to use them with MLForecast.

## Setup

Install the required libraries for this article:

```bash
pip install mlforecast lightgbm utilsforecast
```

We'll use a subset of the [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset. This dataset contains daily sales data from Corporación Favorita, a large Ecuadorian grocery retailer, with rich exogenous variables including store metadata, promotions, oil prices, and holidays.

The subset contains 5 stores and 3 product families (GROCERY I, BEVERAGES, PRODUCE) from 2016-2017, merged with store metadata, oil prices, and holiday information.

```python
import pandas as pd

# DATA_URL = 'https://raw.githubusercontent.com/Nixtla/nixtla_blog/main/examples/data/mlforecast_exogenous/store_sales_subset.csv'
DATA_URL = 'data/mlforecast_exogenous/store_sales_subset.csv'
series = pd.read_csv(DATA_URL, parse_dates=['ds'])
series.head()
```

|     | unique_id   | ds         | y      | store_nbr | family    | city  | state     | type | cluster | onpromotion | oil_price | is_holiday |
| --- | ----------- | ---------- | ------ | --------- | --------- | ----- | --------- | ---- | ------- | ----------- | --------- | ---------- |
| 0   | 1_BEVERAGES | 2016-01-01 | 0.0    | 1         | BEVERAGES | Quito | Pichincha | D    | 13      | 0           | 37.13     | 1          |
| 1   | 1_BEVERAGES | 2016-01-02 | 1856.0 | 1         | BEVERAGES | Quito | Pichincha | D    | 13      | 7           | NaN       | 0          |
| 2   | 1_BEVERAGES | 2016-01-03 | 1048.0 | 1         | BEVERAGES | Quito | Pichincha | D    | 13      | 1           | NaN       | 0          |
| 3   | 1_BEVERAGES | 2016-01-04 | 3005.0 | 1         | BEVERAGES | Quito | Pichincha | D    | 13      | 3           | 36.81     | 0          |
| 4   | 1_BEVERAGES | 2016-01-05 | 2374.0 | 1         | BEVERAGES | Quito | Pichincha | D    | 13      | 9           | 35.97     | 0          |

The dataset contains 15 time series (5 stores × 3 product families) with:

- **Static features**: `store_nbr`, `family`, `city`, `state`, `type`, `cluster`
- **Dynamic features**: `onpromotion` (number of items on promotion), `oil_price` (daily oil price)
- **Calendar feature**: `is_holiday` (whether the date is a holiday)

Split the data into training and test sets. We'll hold out the last 7 days for evaluation:

```python
from utilsforecast.losses import mae
from utilsforecast.plotting import plot_series

horizon = 7
test = series.groupby('unique_id').tail(horizon).copy()
train = series.drop(test.index).copy()
print(f"Train: {len(train)} rows, Test: {len(test)} rows")
```

```text
Train: 8775 rows, Test: 105 rows
```

## Baseline Forecast

Before adding exogenous variables, let's establish a baseline. This model uses only a single lag feature (yesterday's value), giving us a reference point to measure the impact of each exogenous variable type.

```python
import lightgbm as lgb
from mlforecast import MLForecast

fcst_baseline = MLForecast(
    models=lgb.LGBMRegressor(n_jobs=1, random_state=0, verbosity=-1),
    freq='D',
    lags=[1],
    num_threads=2,
)
```

For a complete list of parameters including `lags`, `date_features`, and `target_transforms`, see the [MLForecast API documentation](https://nixtlaverse.nixtla.io/mlforecast/forecast.html#mlforecast).

Now fit the model to the training data. For the baseline, we use only the core time series columns without exogenous variables:

```python
train_baseline = train[['unique_id', 'ds', 'y']]
fcst_baseline.fit(train_baseline)
preds_baseline = fcst_baseline.predict(h=horizon)
```

Evaluate the baseline model:

```python
eval_baseline = test.merge(preds_baseline, on=['unique_id', 'ds'])
baseline_mae = mae(eval_baseline, models=['LGBMRegressor'])['LGBMRegressor'].mean()
print(f"Baseline MAE: {baseline_mae:.2f}")
```

```text
Baseline MAE: 694.14
```

Visualize how the baseline model performs without exogenous variables:

```python
plot_series(
    train,
    forecasts_df=preds_baseline,
    max_ids=4,
    plot_random=False,
    max_insample_length=50,
    engine='matplotlib'
)
```

```chart-multiple
{
  "id": "chart-1",
  "title": "Baseline Forecast",
  "dataSource": "chart-1.csv",
  "columns": 2,
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Sales" },
  "charts": [
    { "id": "chart-1-1", "title": "1_BEVERAGES", "series": [{ "column": "1_BEVERAGES_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_BEVERAGES_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-1-2", "title": "1_GROCERY I", "series": [{ "column": "1_GROCERY I_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_GROCERY I_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-1-3", "title": "1_PRODUCE", "series": [{ "column": "1_PRODUCE_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_PRODUCE_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-1-4", "title": "2_BEVERAGES", "series": [{ "column": "2_BEVERAGES_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "2_BEVERAGES_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] }
  ]
}
```

The baseline forecasts flatten quickly after the first step. With only yesterday's value as input, the model can't anticipate the weekly spikes and dips visible in the historical data.

## Static Features

Static features represent time-invariant characteristics of each series. In retail forecasting, these might include store type, product category, or geographic region. Our dataset includes store metadata like `city`, `state`, `type`, and `cluster`.

Without MLForecast, you would need to manually replicate these values when constructing features for prediction:

```python
# Manual approach: merge static features for prediction
static_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']
static_df = series.groupby('unique_id')[static_cols].first().reset_index()
future_dates = pd.DataFrame({'unique_id': ids, 'ds': future_timestamps})
future_with_static = future_dates.merge(static_df, on='unique_id')
```

MLForecast handles this automatically. Specify static columns in the `static_features` parameter.

First, convert string columns to categorical type so LightGBM can process them:

```python
# Convert string columns to categorical
cat_cols = ['family', 'city', 'state', 'type']
for col in cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

static_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']

# Select only static features (exclude dynamic columns for now)
train_static = train[['unique_id', 'ds', 'y'] + static_cols]
```

Now train the model with static features:

```python
fcst_static = MLForecast(
    models=lgb.LGBMRegressor(n_jobs=1, random_state=0, verbosity=-1),
    freq='D',
    lags=[1],
    num_threads=2,
)

fcst_static.fit(train_static, static_features=static_cols)
preds_static = fcst_static.predict(h=horizon)
```

Evaluate and compare to baseline:

```python
eval_static = test.merge(preds_static, on=['unique_id', 'ds'])
static_mae = mae(eval_static, models=['LGBMRegressor'])['LGBMRegressor'].mean()
improvement = (baseline_mae - static_mae) / baseline_mae * 100
print(f"Static features MAE: {static_mae:.2f} ({improvement:.1f}% improvement over baseline)")
```

```text
Static features MAE: 804.22 (-15.9% improvement over baseline)
```

Static features alone don't improve accuracy here because all stores in our subset are from Quito with similar characteristics. The model overfits to categorical noise rather than learning useful patterns.

Visualize how static features affect the predictions:

```python
plot_series(
    train,
    forecasts_df=preds_static,
    max_ids=4,
    plot_random=False,
    max_insample_length=50,
    engine='matplotlib'
)
```

```chart-multiple
{
  "id": "chart-2",
  "title": "Static Features Forecast",
  "dataSource": "chart-2.csv",
  "columns": 2,
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Sales" },
  "charts": [
    { "id": "chart-2-1", "title": "1_BEVERAGES", "series": [{ "column": "1_BEVERAGES_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_BEVERAGES_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-2-2", "title": "1_GROCERY I", "series": [{ "column": "1_GROCERY I_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_GROCERY I_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-2-3", "title": "1_PRODUCE", "series": [{ "column": "1_PRODUCE_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_PRODUCE_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-2-4", "title": "2_BEVERAGES", "series": [{ "column": "2_BEVERAGES_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "2_BEVERAGES_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] }
  ]
}
```

The flat predictions persist despite adding store metadata. With all stores from Quito sharing similar characteristics, the categorical features add noise rather than signal.

## Dynamic Exogenous Variables

Dynamic exogenous variables change over time but their future values are known in advance. Our dataset includes three dynamic features:

- `onpromotion`: Number of items on promotion (retailers plan promotions ahead)
- `oil_price`: Daily oil price (affects transportation costs and consumer spending in Ecuador)
- `is_holiday`: Whether the date is a holiday (known from the calendar)

The dataset already contains these features. We need to handle missing oil prices by forward-filling:

```python
# Select columns for dynamic features (excluding is_holiday from training for now)
dynamic_cols = ['onpromotion', 'oil_price', 'is_holiday']

# Prepare training data with static and dynamic features
train_dynamic = train[['unique_id', 'ds', 'y'] + static_cols + dynamic_cols].copy()
train_dynamic['oil_price'] = train_dynamic['oil_price'].ffill()

# Prepare future values for prediction
future_dynamic = test[['unique_id', 'ds'] + dynamic_cols].copy()
future_dynamic['oil_price'] = future_dynamic['oil_price'].ffill()
```

Train the model with both static and dynamic features. The `onpromotion` and `oil_price` columns are automatically treated as dynamic since they're not listed in `static_features`:

```python
fcst_dynamic = MLForecast(
    models=lgb.LGBMRegressor(n_jobs=1, random_state=0, verbosity=-1),
    freq='D',
    lags=[1],
    num_threads=2,
)

fcst_dynamic.fit(train_dynamic, static_features=static_cols)
```

For prediction, provide future values through the `X_df` parameter:

```python
preds_dynamic = fcst_dynamic.predict(h=horizon, X_df=future_dynamic)
```

Evaluate and compare:

```python
eval_dynamic = test.merge(preds_dynamic, on=['unique_id', 'ds'])
dynamic_mae = mae(eval_dynamic, models=['LGBMRegressor'])['LGBMRegressor'].mean()
improvement = (baseline_mae - dynamic_mae) / baseline_mae * 100
print(f"Dynamic features MAE: {dynamic_mae:.2f} ({improvement:.1f}% improvement over baseline)")
```

Dynamic features MAE: 537.92 (22.5% improvement over baseline)

Promotions and oil prices deliver significant gains. Products on promotion see predictable demand spikes, and oil prices affect transportation costs and consumer spending patterns in Ecuador's economy.

Visualize how dynamic features improve the forecasts:

```python
plot_series(
    train,
    forecasts_df=preds_dynamic,
    max_ids=4,
    plot_random=False,
    max_insample_length=50,
    engine='matplotlib'
)
```

```chart-multiple
{
  "id": "chart-3",
  "title": "Dynamic Features Forecast",
  "dataSource": "chart-3.csv",
  "columns": 2,
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Sales" },
  "charts": [
    { "id": "chart-3-1", "title": "1_BEVERAGES", "series": [{ "column": "1_BEVERAGES_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_BEVERAGES_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-3-2", "title": "1_GROCERY I", "series": [{ "column": "1_GROCERY I_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_GROCERY I_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-3-3", "title": "1_PRODUCE", "series": [{ "column": "1_PRODUCE_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_PRODUCE_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-3-4", "title": "2_BEVERAGES", "series": [{ "column": "2_BEVERAGES_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "2_BEVERAGES_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] }
  ]
}
```

The forecasts now show variation instead of flat lines. Promotion counts and oil prices give the model actionable signals about when demand will shift.

## Calendar Features

Calendar patterns like day-of-week and month effects are common in time series data. Retail sales spike on weekends, energy consumption varies by season, and traffic patterns follow weekly cycles.

MLForecast's [`date_features` parameter](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/custom_date_features.html) extracts these patterns automatically. You can pass pandas datetime attributes or custom functions:

```python
def is_weekend(dates):
    return dates.dayofweek >= 5

fcst_calendar = MLForecast(
    models=lgb.LGBMRegressor(n_jobs=1, random_state=0, verbosity=-1),
    freq='D',
    lags=[1],
    date_features=['dayofweek', 'month', is_weekend],
    num_threads=2,
)

fcst_calendar.fit(train_static, static_features=static_cols)
preds_calendar = fcst_calendar.predict(h=horizon)
```

Common `date_features` options:

- `dayofweek`: Day of week (0=Monday, 6=Sunday)
- `month`: Month of year (1-12)
- `dayofyear`: Day of year (1-366)
- `quarter`: Quarter of year (1-4)
- Custom functions like `is_weekend` above

Evaluate and compare:

```python
eval_calendar = test.merge(preds_calendar, on=['unique_id', 'ds'])
calendar_mae = mae(eval_calendar, models=['LGBMRegressor'])['LGBMRegressor'].mean()
improvement = (baseline_mae - calendar_mae) / baseline_mae * 100
print(f"Calendar features MAE: {calendar_mae:.2f} ({improvement:.1f}% improvement over baseline)")
```

```text
Calendar features MAE: 385.40 (44.5% improvement over baseline)
```

Calendar features deliver the largest improvement at 44.5%. Grocery shopping follows strong weekly rhythms: customers stock up before weekends, avoid certain weekdays, and shift behavior around month boundaries. These patterns are consistent and directly encoded in the timestamp.

Visualize how calendar features capture weekly patterns:

```python
plot_series(
    train,
    forecasts_df=preds_calendar,
    max_ids=4,
    plot_random=False,
    max_insample_length=50,
    engine='matplotlib'
)
```

```chart-multiple
{
  "id": "chart-4",
  "title": "Calendar Features Forecast",
  "dataSource": "chart-4.csv",
  "columns": 2,
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Sales" },
  "charts": [
    { "id": "chart-4-1", "title": "1_BEVERAGES", "series": [{ "column": "1_BEVERAGES_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_BEVERAGES_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-4-2", "title": "1_GROCERY I", "series": [{ "column": "1_GROCERY I_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_GROCERY I_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-4-3", "title": "1_PRODUCE", "series": [{ "column": "1_PRODUCE_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "1_PRODUCE_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] },
    { "id": "chart-4-4", "title": "2_BEVERAGES", "series": [{ "column": "2_BEVERAGES_y", "name": "Actual", "type": "line", "color": "blue-500" }, { "column": "2_BEVERAGES_LGBMRegressor", "name": "Forecast", "type": "line", "color": "orange-400", "strokeDashArray": "5 5" }] }
  ]
}
```

The forecasts now capture the weekly rhythm visible in the training data. Day-of-week features let the model distinguish high-traffic days from slower ones.

## Feature Importance with SHAP

Understanding which features drive your forecasts helps validate model behavior and guide feature engineering. [SHAP](https://shap.readthedocs.io/) (SHapley Additive exPlanations) values show how each feature contributes to predictions across your dataset.

First, extract the preprocessed features used during training:

```python
prep = fcst_calendar.preprocess(train_static)
X = prep.drop(columns=['unique_id', 'ds', 'y'])
X.head()
```

|     | store_nbr | family    | city  | state     | type | cluster | lag1   | dayofweek | month | is_weekend |
| --- | --------- | --------- | ----- | --------- | ---- | ------- | ------ | --------- | ----- | ---------- |
| 1   | 1         | BEVERAGES | Quito | Pichincha | D    | 13      | 0.0    | 5         | 1     | True       |
| 2   | 1         | BEVERAGES | Quito | Pichincha | D    | 13      | 1856.0 | 6         | 1     | True       |
| 3   | 1         | BEVERAGES | Quito | Pichincha | D    | 13      | 1048.0 | 0         | 1     | False      |
| 4   | 1         | BEVERAGES | Quito | Pichincha | D    | 13      | 3005.0 | 1         | 1     | False      |
| 5   | 1         | BEVERAGES | Quito | Pichincha | D    | 13      | 2374.0 | 2         | 1     | False      |

Compute SHAP values using TreeExplainer, which is optimized for tree-based models like LightGBM:

```python
import shap

explainer = shap.TreeExplainer(fcst_calendar.models_['LGBMRegressor'])
shap_values = explainer(X)
```

Visualize feature importance with a bar plot:

```python
shap.plots.bar(shap_values)
```

```chart
{
  "id": "chart-5",
  "title": "Feature Importance (SHAP)",
  "dataSource": "chart-5.csv",
  "xAxis": { "key": "mean_abs_shap" },
  "yAxis": { "label": "Features" },
  "series": [
    {
      "column": "mean_abs_shap",
      "type": "bar",
      "horizontal": true
    }
  ]
}
```

The results validate our earlier findings:

- `lag1` confirms yesterday's sales as the strongest predictor
- `dayofweek` provides the most value among calendar features, matching the 44.5% accuracy improvement
- Static features `city`, `state`, and `type` have zero impact, explaining why they didn't improve the baseline
- `is_weekend` adds nothing since the model already captures weekly patterns through `dayofweek`

## Conclusion

MLForecast provides a clean API for incorporating external factors into your forecasts:

- **Static features**: Specify in `static_features` parameter during `fit()`. Use for store metadata, product categories, and geographic attributes.
- **Dynamic exogenous variables**: All non-static columns are treated as dynamic. Provide future values via `X_df` in `predict()`. Use for prices, promotions, and scheduled events.
- **Calendar features**: Add via `date_features` parameter. Use built-in pandas attributes or custom functions for day-of-week effects, holidays, and seasonal patterns.

Start with calendar features and static metadata since they require no additional data preparation. Add dynamic exogenous variables when you have reliable forecasts of external factors like planned prices or scheduled promotions.
