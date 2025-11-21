---
title: Long Term Mid Term and Short Term Forecasting with Polynomial Regression AutoARIMA and TimeGPT-1
description: Discover how to decompose your time series in multiple components with Fourier Transform and model each component with TimeGPT-1.
image: /images/MultiLevelForecasting/main_image.svg
categories: ["Time Series Forecasting"]
tags:
  - TimeGPT
  - AutoARIMA
  - multi-horizon forecasting
  - polynomial regression
author_name: Piero Paialunga
author_image: "/images/authors/piero.jpg"
author_position: Data Scientist
publication_date: 2025-08-26
---

With forecasting, you're trying to predict the future values of a time series based on its historical patterns, but **is your model aligned with how far ahead you're looking?** The time horizon, whether you're forecasting hours, weeks, or years ahead, drastically changes the type of model you should use. Ignoring your time horizon in favor of the trending "state of the art" forecasting model could lead to poor accuracy and wasted compute.

In general, we can define three levels of forecasting:

- **Long-term forecasting**, e.g. predicting the average New York temperatures for 2026–2028 using temperatures from 1925–2025.

- **Short-term forecasting**, e.g. predicting the next hour stock price using minute-by-minute updates.

- **Mid-term forecasting**, e.g. estimating electricity demand for the coming weeks using daily data from recent months.

In this blog post, we'll walk through a single forecasting problem and tackle it using three different approaches: one for long term forecasting, one for mid term forecasting, and one for short term forecasting.

## Multi-Horizon Forecasting Strategy

To handle long-term, mid-term, and short-term forecasts effectively, we can follow this structured approach:

- Start with the raw time series.
- Apply a rolling average of the data based on the desired time horizon (year, month, day). This is called **aggregation**.
- Apply a forecasting model suited to that level of aggregation to generate future estimates.

![image](/images/MultiLevelForecasting/diagram.svg)

## Data and Source

All the code used in this article is available in the GitHub repository [PieroPaialungaAI/MultiLevelForecasting](https://github.com/PieroPaialungaAI/MultiLevelForecasting/tree/main). The repository includes helper modules such as `utils.py`, `plotter.py`, `constants.py` and `models.py`, which handle the backend functionality so we can concentrate on the core logic of the algorithm.

The dataset used in this article is publicly available on [this Kaggle page](https://www.kaggle.com/datasets/lsind18/euro-exchange-daily-rates-19992020). It contains daily exchange rates of various currencies against the euro from 1999 to 2020. For convenience, the dataset is also included in the repository under the data folder ([PieroPaialungaAI/MultiLevelForecasting/data](https://github.com/PieroPaialungaAI/MultiLevelForecasting/tree/main/data)), so you don't need to download it separately.

The following block of code provides a quick overview of the dataset. The plot displays the time series of the US Dollar value against 1 euro from 1999 to 2025, shown in cyan.

```pyton
import numpy as np
import pandas as pd
from plotter import *
from utils import *

data = pd.read_csv('data/euro_conversion_data.csv')
plot_timeseries(data)
```

![image](/images/MultiLevelForecasting/raw_timeseries_plot.svg)

```chart
{
  "id": "chart-1",
  "title": "Raw Time Series",
  "dataSource": "chart-1.csv",
  "xAxis": {
    "key": "Date"
  },
  "yAxis": {
    "label": "US dollar - Euro Conversion"
  },
  "series": [
    {
      "column": "US dollar",
      "name": "Data",
      "type": "line"
    }
  ]
}
```

As we can see from the plot above, the timeseries shows a rich and complex structure. There are long-term trends, such as the rise and fall between 2002 and 2015, mid-term cycles, and short-term fluctuations that can be very interesting if we are monitoring the rapid evolution of the market.

This is perfect for our study case: we can average the raw time series into yearly, monthly, or daily aggregates and build a tailored forecasting algorithms based on our aggregated time series. Let's work on that.

## Aggregation Code

The **aggregation of the data** based on month/year/day can be done very easily using the `aggregate_data` function:

```python
def aggregate_data(data, key='year', column = Y_COLUMN):
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'])  # ensure proper dtype
    if key == 'year':
        data['time_group'] = data['Date'].dt.to_period('Y').astype(str)
    elif key == 'month':
        data['time_group'] = data['Date'].dt.to_period('M').astype(str)
    elif key == 'day':
        data['time_group'] = data['Date'].dt.to_period('D').astype(str)
    else:
        raise ValueError("Unsupported key. Choose from: 'year', 'month', 'day'")
    return data.groupby('time_group').mean(numeric_only=True).reset_index()[['time_group',column]]
```

Now that we have the aggregated function we can build this aggregated time series dictionary. For each one of them we will perform a forecasting task.

```python
keys = ['year','month','day']
key_dict = {key: aggregate_data(data, key = key) for key in keys}
```

## Long Term Forecasting

Let's give a look at the year aggregated time series.

```python
from models import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from constants import *

# Declare Dataset
y = np.array(key_dict['year']['US dollar'])
x_full = np.arange(0, len(y)).reshape(-1, 1)
print(x_full.shape)

(27, 1)
```

And it looks like this:

![image](/images/MultiLevelForecasting/year_averaged_data.svg)

```chart
{
  "id": "chart-2",
  "title": "Yearly Averaged Data",
  "dataSource": "chart-2.csv",
  "xAxis": {
    "key": "time_group"
  },
  "yAxis": {
    "label": "US dollar - Euro Conversion"
  },
  "series": [
    {
      "column": "US dollar",
      "name": "Data",
      "type": "line"
    }
  ]
}
```

In this case, we are interested in forecasting the next 2-3 averaged years of our time series.

### Model

In our yearly averaged time series we have 27 values: not a whole lot. This could be considered discouraging, as forecasting algorithms (and machine learning models in general) perform well with a large training dataset.

However, yearly aggregation has its own advantages. By averaging out short-term fluctuations, the signal becomes reasonably smooth and more focused on long-term trends. In this context, even simple models can capture meaningful patterns effectively, making them surprisingly powerful despite the limited data.

To model this kind of smooth, long-term behavior, we can use **polynomial regression**. This method fits a curved line (a polynomial) to the data using the least squares approach to find the best-fitting parameters. It allows us to capture more complex, nonlinear trends that a straight line wouldn’t be able to model.

A common approach to selecting the best **polynomial degree** and avoid overfitting is to use a train/validation/test split. The model is:

- **Trained** on the first part of the time series
- **Validated** on the next consecutive segment
- **Tested** on the final portion of the time series

### Code

The code to do such is this:

```python
train, val, test = split_timeseries(y)
model_tuple, best_degree = polynomial_fit_and_select(train, val)
poly = PolynomialFeatures(degree = best_degree).fit(x_full)
X_test_poly = poly.transform(x_full)
y_full_pred = model_tuple[0].predict(X_test_poly)
train_index = len(train)
val_index = len(val)
test_index = len(test)
```

And we can plot it using the following block of code. The plot represents:

- In **lime** the train part of the time series
- In **blue** the validation part of the time series
- In **white** the test part of the timeseries
- In **cyan** the long term forecasting algorithm result

```python
x_full = x_full + 1999
plt.figure(figsize = IMAGE_FIGSIZE)
plt.plot(x_full[0:train_index], y[:train_index], label='Train', color = 'lime', marker ='x')
plt.plot(x_full[train_index-1:train_index+val_index], y[train_index-1:train_index+val_index], label='Validation', color = 'blue', marker = 'x')
plt.plot(x_full[train_index+val_index-1:], y[train_index+val_index-1:], label='Test', color = 'white', marker = 'x')
plt.plot(x_full, y_full_pred, color = 'cyan', label = 'Model Prediction')
```

![image](/images/MultiLevelForecasting/year_prediction.svg)

```chart
{
  "id": "chart-3",
  "title": "Year Prediction with Polynomial Regression",
  "dataSource": "chart-3.csv",
  "xAxis": {
    "key": "year"
  },
  "yAxis": {
    "label": "US dollar - Euro Conversion"
  },
  "series": [
    {
      "column": "actual",
      "name": "Data",
      "type": "line"
    },
    {
      "column": "prediction",
      "name": "Data 2",
      "type": "line"
    },
    {
      "column": "split",
      "name": "Data 3",
      "type": "line"
    }
  ]
}
```

As we can see, the model closely follows the shape of the time series and manages to forecast the next three years (2023, 2024, and 2025) with surprising accuracy, despite its simplicity.

## Mid Term Forecasting

The mid term dataset is roughly 12 times bigger than the long term one, as we can see from the following code:

```python
df = key_dict['month']
x_full = np.arange(0, len(df)).reshape(-1, 1)
print(x_full.shape)
(316, 1)
```

And it looks like this:

![image](/images/MultiLevelForecasting/month_averaged_data.svg)

```chart
{
  "id": "chart-4",
  "title": "Monthly Averaged Data",
  "dataSource": "chart-4.csv",
  "xAxis": {
    "key": "time_group"
  },
  "yAxis": {
    "label": "US dollar - Euro Conversion"
  },
  "series": [
    {
      "column": "US dollar",
      "name": "Data",
      "type": "line"
    }
  ]
}
```

In this case, we are interested in understanding what will happen, let's say, in the next 3 or 4 months of our time series.

### Model

For this dataset, we can try to apply a more complex model. An approach that allows us to control the complexity of the model is called **ARIMA**.

[ARIMA (AutoRegressive Integrated Moving Average)](https://nixtlaverse.nixtla.io/statsforecast/docs/models/arima.html) is a statistical model that combines three components: autoregression (past values), differencing (to make the series stationary), and moving average (past forecast errors) to predict future values. It’s well-suited for univariate time series that show patterns over time, and it can adapt to trends and seasonality depending on how it's configured.

In particular, the ARIMA model is defined by three key hyperparameters: **p (_number of autoregressive terms_), d (_number of differences needed to make the series stationary_), and q (_number of moving average terms_)**, which together control how the model learns from past values and errors. Choosing these numbers is non trivial and requires some time series exploration; however the [**AutoARIMA**](https://nixtlaverse.nixtla.io/statsforecast/docs/models/autoarima.html) tool within [`statsforecast`](https://nixtlaverse.nixtla.io/statsforecast/index.html) does the exploration for us and helps us select the optimal p, d and q values for our time series.

### Code

With the following block of code, we run AutoARIMA on the most recent portion of the dataset, trimming the data to start from 2024.

```python
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
df = key_dict['month']
x_full = np.arange(0, len(y)).reshape(-1, 1)
df["unique_id"]="1"
df.columns=["ds", "y", "unique_id"]
Y_train_df = df[(df.ds>='2023-12-01') & (df.ds<='2024-12-01')]
Y_test_df = df[df.ds>='2024-12-01']
season_length = 4 # Monthly data
horizon = len(Y_test_df)+1 # number of predictions
models = [AutoARIMA(season_length=season_length)]
sf = StatsForecast(models=models, freq='MS')
sf.fit(df=Y_train_df)
Y_hat_df = sf.forecast(df=Y_train_df, h=horizon, fitted=True)
Y_hat_df.head()
values=sf.forecast_fitted_values()
values
```

|     | unique_id | ds         |       y | AutoARIMA |
| --: | --------: | :--------- | ------: | --------: |
|   0 |         1 | 2024-01-01 | 1.09051 |   1.08271 |
|   1 |         1 | 2024-02-01 | 1.07947 |   1.08518 |
|   2 |         1 | 2024-03-01 | 1.08722 |   1.07857 |
|   3 |         1 | 2024-04-01 | 1.07278 |   1.08633 |
|   4 |         1 | 2024-05-01 | 1.08122 |   1.07399 |
|   5 |         1 | 2024-06-01 |  1.0759 |   1.08562 |
|   6 |         1 | 2024-07-01 | 1.08441 |   1.07611 |
|   7 |         1 | 2024-08-01 | 1.10122 |   1.08623 |
|   8 |         1 | 2024-09-01 |  1.1106 |   1.08999 |
|   9 |         1 | 2024-10-01 | 1.09043 |   1.09315 |
|  10 |         1 | 2024-11-01 | 1.06301 |   1.08004 |
|  11 |         1 | 2024-12-01 | 1.04787 |     1.072 |

As now we have a fitted ARIMA Model, we can use it to predict the first months of 2025.

```python
Y_test_pred = sf.forecast(df=Y_train_df, h=4, level=[95])
```

The following block of code displays the forecast of the ARIMA model:

- In **lime**, we show the historical monthly US Dollar to Euro conversion rates starting from 2019, trimmed to focus on the most recent years.
- In **cyan**, we display the mid-term forecast generated by the AutoARIMA model.
- The **blue shaded area** represents the confidence interval around the forecast, capturing the model’s uncertainty over the next few months.

```python
x_full = np.array(pd.to_datetime(df['ds']))
y_full = np.array(df['y'])
x_test = np.array(pd.to_datetime(Y_test_df['ds']))
y_test = np.array(Y_test_df['y'])
y_pred_test = np.array(Y_test_pred['AutoARIMA'])
lower_bound = np.array(Y_test_pred['AutoARIMA-lo-95'])
upper_bound = np.array(Y_test_pred['AutoARIMA-hi-95'])
x_pred_test =  np.array(Y_test_pred['ds'])
plt.plot(x_full[240:], y_full[240:], color = 'lime', label = '')
plt.plot(x_pred_test, y_pred_test, color ='cyan', label = 'AutoARIMA prediction')
plt.fill_between(x_pred_test, lower_bound, upper_bound, color = 'cyan', alpha = 0.3, label = 'AutoARIMA 95p boundaries')
plt.plot(x_test, y_test, color = 'lime', label = 'Test Set Data')
plt.xlabel('Date (Year, sampled in Month)', fontsize = 12)
plt.ylabel('US Dollar-Euro Conversion',fontsize = 12 )
plt.legend()
plt.savefig('/images/MultiLevelForecasting/month_prediction.svg')
```

![image](/images/MultiLevelForecasting/month_prediction.svg)

```chart
{
  "id": "chart-5",
  "title": "Month Prediction with AutoARIMA",
  "dataSource": "chart-5.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "US dollar - Euro Conversion"
  },
  "series": [
    {
      "type": "area",
      "columns": {
        "high": "upper_bound",
        "low": "lower_bound"
      },
      "name": "AutoARIMA 95p Interval"
    },
    {
      "column": "actual",
      "type": "line",
      "name": "Test Set Data"
    },
    {
      "column": "prediction",
      "type": "line",
      "name": "Auto ARIMA Prediction"
    }
  ]
}
```

As we can see, the AutoARIMA model does a solid job here. The forecast (in cyan) tracks the actual test data (in lime) pretty closely, and the 95% confidence interval (in dark teal) comfortably captures the uncertainty without being overly wide. It’s a good example of how a well-tuned statistical model can perform strongly for mid-term forecasting.

## Short Term Forecasting

For short-term forecasting, we focus on the daily movements of the US Dollar–Euro conversion rate. Our dataset is already sampled at the daily level and this is how it looks like:

```python
df = key_dict['day']
x_full = np.arange(0, len(df)).reshape(-1, 1)
print(x_full.shape)
(6723, 1)
```

![image](/images/MultiLevelForecasting/day_averaged_data.svg)

```chart
{
  "id": "chart-6",
  "title": "Day Averaged Data",
  "dataSource": "chart-6.csv",
  "xAxis": {
    "key": "time_group"
  },
  "yAxis": {
    "label": "US dollar - Euro Conversion"
  },
  "series": [
    {
      "column": "US dollar",
      "name": "Data",
      "type": "line"
    }
  ]
}
```

### Model

For short-term forecasting, we shift our focus to daily data, where small fluctuations matter and recent changes carry the most weight. In this setting, traditional statistical models often fall short, we need something that can handle fine-grained patterns and fast dynamics. That’s where deep learning models, like [**TimeGPT-1**](https://www.nixtla.io/docs), come into play.

TimeGPT-1 is a transformer-based model specifically trained for time series forecasting across a wide range of domains. Unlike traditional models, it doesn't require manual feature engineering or extensive tuning, it learns patterns directly from raw temporal data. Its ability to generalize and handle complex, non-linear trends makes it especially powerful for short-term, high-frequency forecasting tasks.

### Code

The code to forecast the first 10 days of 2025 can be seen in the following block of code.

```python
from nixtla import NixtlaClient
keys = ['year','month','day']
key_dict = {key: aggregate_data(data, key = key) for key in keys}
df = key_dict['day']
df = df.rename(columns = {'time_group': 'ds'})
df = df[df.ds >= '2024-01-01']
df = df.reset_index().drop('index', axis  = 1)
x_full = np.arange(0, len(y)).reshape(-1, 1)
df["unique_id"]="1"
timestamps = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
df['ds'] = timestamps
df.columns=["ds", "y", "unique_id"]
Y_train_df = df[(df.ds <= '2024-11-09') & (df.ds >= '2024-01-01')]
Y_test_df = df[df.ds >= '2024-11-09']
test = Y_test_df
input_seq = Y_train_df
api_key = "your_api_key"
nixtla_client = NixtlaClient(api_key=api_key)
fcst_df = nixtla_client.forecast(
    df=input_seq,
    h=10,
    level=[75],
    finetune_steps=3,
    finetune_loss='mae',
    model='timegpt-1',
    time_col='ds',
    target_col='y'
)
```

And this is how it looks like:

- **In blue**, we show the training data, which is the historical daily US Dollar to Euro conversion leading up to the forecast window.

- **In lime**, we display the actual test data for the short-term future.

- **In cyan**, we plot the TimeGPT-1 forecast for the next few days.

- The **cyan shaded area** represents the 75% confidence interval produced by TimeGPT-1, capturing the model's uncertainty in its predictions.

![image](/images/MultiLevelForecasting/day_prediction.svg)

```chart
{
  "id": "chart-7",
  "title": "Day Prediction with TimeGPT",
  "dataSource": "chart-7.csv",
  "xAxis": {
    "key": "ds"
  },
  "yAxis": {
    "label": "US dollar - Euro Conversion"
  },
  "series": [
    {
      "type": "area",
      "columns": {
        "high": "upper_bound",
        "low": "lower_bound"
      },
      "name": "TimeGPT 95p Interval"
    },
    {
      "column": "actual",
      "type": "line",
      "name": "Training Data"
    },
    {
      "column": "prediction",
      "type": "line",
      "name": "TimeGPT Forecast"
    }
  ]
}
```

TimeGPT-1 is doing a great job here. The forecast (in cyan) aligns well with the actual test data (in lime), and the p75 confidence boundaries (in cyan) wrap the predictions nicely without being too loose or overly optimistic. It shows how transformer-based models can effectively capture short-term fluctuations, even in noisy daily data.

## Conclusions

In this article, we explored how to approach **forecasting across multiple time horizons, long term, mid term, and short term**, using data aggregation and tailored models.

In particular, we did the following:

- We introduced a strategy to handle different forecasting time scales by aggregating raw time series data into yearly, monthly, and daily frequencies.

- We applied **Polynomial Regression for long-term forecasting**, showing that even simple models can perform well on smoothed, low-frequency data.

- We used **AutoARIMA for mid-term forecasting**, allowing automatic selection of model parameters and achieving strong results on monthly data.

- We used **TimeGPT-1, a transformer-based model for short-term forecasting**, demonstrating its ability to capture fine-grained dynamics in high-frequency daily data.
