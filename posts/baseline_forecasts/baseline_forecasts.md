---
title: "Effortless Accuracy Unlocking the Power of Baseline Forecasts"
description: "Understand what are baseline forecasts, why they are important and learn to create them easily with Nixtla's statsforecast package."
image: "/images/baseline_forecasts/forecast_featured.png"
categories: ["Time Series Forecasting"]
tags:
  - baseline forecasting
  - statsforecast
  - naive forecast
  - seasonal naive
author_name: Khuyen Tran
author_image: "/images/authors/khuyen.jpeg"
author_position: Developer Advocate - Nixtla
publication_date: 2025-08-26
---

So, you are working on a forecasting project. The data has been set up and the required analysis has been done. Now we jump straight into getting the best forecasts right?

Not so fast. You might be missing a crucial step. **Setting up Baseline Forecasts!**

A baseline forecast provides a critical point of comparison, serving as a reference for all other modeling techniques applied to a specific problem. It helps answer questions like:

- How much accuracy can be achieved with little effort? (or) How predictable is this data?
- How good or bad is the sophisticated model you are working on, compared to the baseline?
- Is the improvement in accuracy using a sophisticated model, compared to the baseline forecast, worth the effort?

So, what are Baseline forecasts? They are usually characterised by:

- Simplicity – Requires minimal training or specialized intelligence.
- Speed – Quick to implement and computationally trivial for prediction.

## Baseline Forecasting Methods

Commonly used trivial baseline forecasting methods involve,

- Mean Forecast
- Naive Forecast
- Seasonal Naive Forecast
- Rolling Averages

This is by no means an exhaustive list, but it is a fair start.

### Mean Forecast

This is simply the mean of all the past observations. This works for time series that are stationary and don't have trend or seasonality.

$$ ŷ\_{T+h \mid T} = \bar{y} = \frac{y_1 + \cdots + y_T}{T} $$

> $$ŷ_{T+h \mid T}$$ represents the forecast for time period $$T+h$$ when we have observations until time period $$T$$.
> $$\bar{y}$$ is the mean of past observations.
> $$y_1, ... , y_T$$ are the past observations.

**Example**:
Suppose we have monthly sales data for the last 5 months:
`[120, 130, 125, 135, 140]`

The mean of past observations is:

$$
\bar{y} = \frac{120 + 130 + 125 + 135 + 140}{5} = \frac{650}{5} = 130
$$

So the forecast for any future month (e.g., month 6, 7, 8, ...) will be:

$$ ŷ\_{T+h \mid T} = 130 $$

**Pros**: It is very simple to calculate and understand. It provides stable forecasts that do not fluctuate over the forecast horizon.

**Cons**: Recent information as well as past information is equally weighted. Hence more relevant recent information can be ignored. It doesn't work well for series with trends or seasonality.

### Naive Forecast

The Naive Forecast is the latest observed value in the current period. This kind of baseline is best suited for data that is close to a random walk pattern.

$$ ŷ\_{T+h \mid T} = y_T$$

**Example**:
Continuing with the 5-month sales data example (`[120, 130, 125, 135, 140]`), the forecast would be the latest observation. In this case the latest observation, $$y_T=140$$. So,
$$ ŷ\_{T+h \mid T} = 140$$

**Pros**: It is super simple to implement and requires minimal data (a single data point is enough!).

**Cons**: It cannot account for any trend or seasonality. Patterns beyond the latest data point is not captured.

### Seasonal Naive Forecast

What if your data set has strong seasonality? Seasonal Naive can serve as your baseline then. This is just like the naive forecast. But instead of the latest observation, we use the value observed in the same season in the previous cycle.

$$ ŷ*{T+h \mid T} = y*{T+h-m(k+1)}$$

> $$m$$ is the seasonal period. 12 for monthly data and 4 for quarterly data.
> $$k = \lfloor(h-1)/m\rfloor$$. This is the integer part (floor) of $$(h-1)/m$$. This ensures that even if we forecast for several seasons ahead, the season from the latest cycle is picked up.

**Example**:
Consider this quarterly sales data.
| **Quarter** | **Period** | **Sales** |
| ---------- | ----------- | --------- |
| 2024-Q1 | 1 | 80 |
| 2024-Q2 | 2 | 90 |
| 2024-Q3 | 3 | 110 |
| 2024-Q4 | 4 | 130 |

If $$T = 4$$, and we want to find the forecast for 2025-Q1, then $$h = 1$$ and $$T+h = 5$$.
We know that $$m = 4$$ for quarterly data.
And $$k = \lfloor(1-1)/4\rfloor = 0$$.
Now, $$T+h-m(k+1) = 4+1-4*(0+1) = 1$$
$$ŷ_{T+h \mid T} = y_{T+h-m(k+1)} = y_{1} = 80 $$

**Pros**: Seasonality is captured with minimal effort.
**Cons**: It does not account for recent observations, trend, or other effects. Multiple seasonalities are not captured. If seasonality is weak or other signals dominate, performance will degrade.

> If you are interested in learning more, read our tutorial on [multiple seasonalities](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/multipleseasonalities.html).

### Rolling Averages Forecast

Rolling Averages or Moving Averages or Window Averages is the mean of the last $$w$$ observations of a time series. The value of $$w$$ (referred to as window or span) has to be decided by the forecaster.
$$ ŷ*{T+h \mid T} = \frac{y_T + y*{T-1} + \cdots + y\_{T-w+1}}{w} $$

> $$w$$ is the window or span over which averages are to be calculated.

**Example**:
Recall the 5-month sales data example (`120, 130, 125, 135, 140`). If we set the window length as $$w=3$$ and we want to find the forecast at $$h=1$$, then $$T = 5$$ and $$T-w+1 = 5-3+1 = 3$$. Thus,
$$ ŷ\_{T+h \mid T} = \frac{y_5 + y_4 + y_3}{3} = \frac{125 + 135 + 140}{3} = 133.33$$

**Pros**: This is a simple technique. We can adjust the $$w$$ value to weight either recency (low $$w$$ value) or stability (high $$w$$ value).

**Cons**: Moving Average responds slower to new information. Seasonality cannot be captured.

### Which model should we use as our baseline?

Now that we've explored several baseline models, which one should you use? The table below summarizes when each model is appropriate:

| **Condition**                                                                     | **Recommended Baseline** |
| --------------------------------------------------------------------------------- | ------------------------ |
| Time series is stationary, no trend or seasonality, values centered around a mean | Mean Forecast            |
| Data resembles a random walk (next value = previous + random noise)               | Naive Forecast           |
| Clear and stable seasonal pattern, with seasonality as the dominant signal        | Seasonal Naive Forecast  |
| Recent values are strong predictors of future values                              | Rolling Average          |

## Easy Baseline Forecasts using the `statsforecast` package

This section guides you through how to implement baseline forecasts using the [StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/index.html) package, a fast and scalable library for statistical time series forecasting.

Install the `statsforecast` if you don't have it installed.

```bash
pip install statsforecast
```

Also, import the necessary packages:

```python
import pandas as pd
import numpy as np
import os
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, HistoricAverage, WindowAverage
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse
```

We will use a subset of the Tourism dataset (from the R `tsibble` package), limited to three regions.

Transform the `ds` column into a quarterly timestamp format to align it with the data's quarterly frequency:

```python
df = pd.read_csv('EffortlessAccuracyUnlockingThePowerOfBaselineForecasts_3Region_tourism.csv')
df['ds'] = pd.PeriodIndex(df['ds'], freq='Q').to_timestamp()
df
```

| unique_id | ds                  | y      |
| --------- | ------------------- | ------ |
| Adelaide  | 1998-01-01T00:00:00 | 658.55 |
| Adelaide  | 1998-04-01T00:00:00 | 449.85 |
| Adelaide  | 1998-07-01T00:00:00 | 592.90 |
| Adelaide  | 1998-10-01T00:00:00 | 524.24 |
| Adelaide  | 1999-01-01T00:00:00 | 548.39 |
| Adelaide  | 1999-04-01T00:00:00 | 568.69 |
| Adelaide  | 1999-07-01T00:00:00 | 538.05 |
| Adelaide  | 1999-10-01T00:00:00 | 562.42 |
| Adelaide  | 2000-01-01T00:00:00 | 646.35 |
| Adelaide  | 2000-04-01T00:00:00 | 562.75 |

The `df` dataframe contains three columns:

- `unique_id`: Identifies each individual time series. One model will be trained per `unique_id`.
- `ds`: The timestamp column. Ensure it is properly formatted with quarterly frequency.
- `y`: The target variable to forecast.

Split the data into training and testing sets:

```python
test_df = df.groupby("unique_id", group_keys=False).tail(4)
train_df = df[~df.index.isin(test_df.index)]
```

Define the baseline models to use for forecasting:

```python
models = [
    HistoricAverage(),
    Naive(),
    SeasonalNaive(season_length = 4), # Quarterly data seasonality = 4
    WindowAverage(window_size=4)
]
```

This list includes the four baseline models discussed earlier:

- `HistoricAverage()`: Mean Forecast
- `Naive()`: Naive Forecast
- `SeasonalNaive(season_length=4)`: Seasonal Naive Forecast. Here, a `season_length` of 4 is used to reflect quarterly seasonality.
- `WindowAverage(window_size=4)`: Rolling Averages. With `window_size=4`, the model averages the last 4 quarters to generate forecasts.

Initialize the `StatsForecast` object with models and frequency, then train it:

```python
sf = StatsForecast(
    models=models,
    freq='QS', # Quarterly frequency
)

# Train the data on all the four models.
sf.fit(train_df)
```

The code is quite brief. `statsforecast` has made training multiple models a walk in the park. Also, note that we didn't have to do anything about the 3 separate time series. This part is also nicely abstracted for us.

Set the number of periods to forecast, then generate predictions using the trained models:

```python
# Define forecast horizon
h = 4 # 4 quarters = 1 year
pred_df = sf.predict(h=h)
```

Let's take a look at our forecasts now using the `plot` method in the `StatsForecast` class:

```python
sf.plot(df, pred_df)
```

![](/images/baseline_forecasts/ts_plot.svg)

Now that we have predictions for the 4 models across the 3 different time series, let's evaluate the forecasts using the `evaluate` method with `rmse` as the error metric:

```python
accuracy_df =  pd.merge(test_df, pred_df, how = 'left', on = ['unique_id', 'ds'])
evaluate(accuracy_df, metrics=[rmse])
```

| unique id | metric | Historic Average | Naive | Seasonal Naive | Window Average |
| --------- | ------ | ---------------- | ----- | -------------- | -------------- |
| Adelaide  | rmse   | 129.97           | 64.81 | 49.45          | 43.65          |
| Ballarat  | rmse   | 61.25            | 46.27 | 51.04          | 49.31          |
| Barkly    | rmse   | 14.7             | 14.96 | 17.76          | 13.54          |

We see the window average forecast provides the best baseline for our data.

## Conclusion

We looked at what baseline forecasts are, why they are important, also how we can get some baseline forecasts up and running in no time using Nixtla's `statsforecast` package.

The right baseline depends on the data-generating process of the time series and the closeness of these processes to the assumptions behind the baseline forecasts.

It is often beneficial to compute several baseline forecasts. Especially when the effort required is so low. Taking a cue from the "No free lunch" theorem, no single baseline method is universally superior. The effectiveness of a baseline depends on how well its implicit assumptions match the data-generating process of the specific time series.

Once one or more satisfactory baselines are established, and their performance is quantified, the focus shifts to developing more [advanced forecasting models](https://www.nixtla.io/docs/tutorials-special_topics-tutorials-improve_forecast_accuracy_with_timegpt). The goal should then be to significantly outperform the best baseline.
