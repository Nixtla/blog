---
title: "Understanding Intermittent Demand"
description: Learn how to forecast intermittent demand using Python and Nixtla's TimeGPT. This step-by-step guide covers handling sparse time series, fine-tuning, and using exogenous variables to improve accuracy.
image: "/images/intermittent_demand/intermittent_demand.svg"
categories: ["Time Series Forecasting"]
tags:
  - TimeGPT
  - intermittent demand
  - sparse data
  - fine-tuning
author_name: Khuyen Tran
author_image: "/images/authors/khuyen.jpeg"
author_position: Developer Advocate - Nixtla
publication_date: 2025-08-26
---

Intermittent demand forecasting poses significant challenges for data scientists, especially when dealing with time series that contain numerous zero values.

![](/images/intermittent_demand/intermittent_demand_spikes.svg)

Traditional forecasting models often struggle with such sparse data, leading to inaccurate predictions. Nixtla's [TimeGPT](https://www.nixtla.io/docs/intro) offers a robust solution to this problem by effectively handling intermittent demand scenarios.

The source code of this article can be found [here](https://nixtla.github.io/nixtla_blog_examples/notebooks/intermittent_forecasting.html).

## Understanding Intermittent Demand

Intermittent demand refers to demand patterns characterized by irregular and unpredictable occurrences, often with many periods of zero demand. This is common in retail scenarios where certain products sell infrequently. Accurately forecasting such demand is crucial for inventory management and reducing holding costs.

## Leveraging TimeGPT for Forecasting

TimeGPT is a time series [forecasting](https://www.nixtla.io/docs/capabilities-forecast-forecast) model developed by [Nixtla](https://www.nixtla.io/), designed to handle various forecasting challenges, including intermittent demand. It utilizes advanced machine learning techniques to provide accurate predictions even in sparse data scenarios.

## Setting Up the Environment

To begin forecasting with TimeGPT, we first need to set up our environment by importing necessary libraries and initializing the [Nixtla client](https://dashboard.nixtla.io/sign_in).

````python
import time
import pandas as pd
import numpy as np

from nixtla import NixtlaClient
from utilsforecast.losses import mae
from utilsforecast.evaluation import evaluate
import os

NIXTLA_API_KEY = os.environ["NIXTLA_API_KEY"]
client = NixtlaClient(api_key=NIXTLA_API_KEY)
...

## Loading and Exploring the Dataset

We use a subset of the M5 dataset, which includes sales data for food items in a Californian store. This dataset is ideal for demonstrating intermittent demand forecasting due to its sparse nature.

```python
sales_data = pd.read_csv(
    "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/m5_sales_exog_small.csv"
)
sales_data["ds"] = pd.to_datetime(sales_data["ds"])
sales_data.head()
````

| unique_id   | ds                  | y   | sell_price | event_type_Cultural | event_type_National | event_type_Religious | event_type_Sporting |
| ----------- | ------------------- | --- | ---------- | ------------------- | ------------------- | -------------------- | ------------------- |
| FOODS_1_001 | 2011-01-29 00:00:00 | 3   | 2.0        | 0                   | 0                   | 0                    | 0                   |
| FOODS_1_001 | 2011-01-30 00:00:00 | 0   | 2.0        | 0                   | 0                   | 0                    | 0                   |
| FOODS_1_001 | 2011-01-31 00:00:00 | 0   | 2.0        | 0                   | 0                   | 0                    | 0                   |
| FOODS_1_001 | 2011-02-01 00:00:00 | 1   | 2.0        | 0                   | 0                   | 0                    | 0                   |
| FOODS_1_001 | 2011-02-02 00:00:00 | 4   | 2.0        | 0                   | 0                   | 0                    | 0                   |

The dataset contains the following columns:

- `unique_id`: Identifier for each product.
- `ds`: Date of the observation.
- `y`: Sales quantity.
- `sell_price`: Price of the product.
- Event types: Indicators for cultural, national, religious, and sporting events.

## Visualizing the Data

To understand the demand patterns, we plot the time series data.

```python
 client.plot(
     sales_data,
      max_insample_length=365,
  )
```

![](/images/intermittent_demand/sales_plot.svg)

The plot reveals the intermittent nature of the demand, with many periods showing zero sales.

## Transforming the Data

We apply a log transformation to the data to stabilize variance, reduce the influence of sharp peaks, and bring the distribution closer to normal—conditions that benefit many forecasting models. Because the log of zero is undefined, we add one to each sales value before applying the transformation.

```python
log_transformed_data = sales_data.copy()
log_transformed_data["y"] = np.log(log_transformed_data["y"] + 1)
log_transformed_data.head()
```

Let's compare the original data with the transformed data by plotting:

```python
import matplotlib.pyplot as plt

# Create a figure and axis for Matplotlib
_, ax = plt.subplots(figsize=(10, 5))

# Plot the original data
client.plot(
    sales_data,
    max_insample_length=30,
    unique_ids=["FOODS_1_001"],
    engine="matplotlib",
    ax=ax,
)

# Plot the transformed data on the same axes
client.plot(
    log_transformed_data,
    max_insample_length=30,
    unique_ids=["FOODS_1_001"],
    engine="matplotlib",
    ax=ax,
)

# Manually change the color of the second line plot
lines = ax.get_lines()
if len(lines) > 1:
    lines[1].set_color("#006400")  # New color for transformed data
    lines[1].set_linestyle("--")

# Add legend with custom labels
handles, labels = ax.get_legend_handles_labels()
labels = ["Original Sales", "Transformed Sales"]
ax.legend(handles, labels)

ax
```

![](/images/intermittent_demand/compared_plot.svg)

The plot shows a clear distinction between the original and log-transformed sales series for FOODS_1_001. The original data has sharp peaks and frequent zero values, indicating highly volatile demand.

After log transformation, the extreme spikes are significantly reduced, and the series becomes smoother and more stable.

## Splitting the Data into Train and Test Sets

To evaluate model performance more realistically, we split the data into training and test sets. We hold out the last 28 observations for each time series as test data:

```python
# Select the last 28 observations for each unique_id — used as test data
test_data = log_transformed_data.groupby("unique_id").tail(28)

# Drop the test set indices from the original dataset to form the training set
train_data = log_transformed_data.drop(test_data.index).reset_index(drop=True)
```

## Forecasting with TimeGPT

With the data prepared, we proceed to forecast future demand using [TimeGPT](https://dashboard.nixtla.io/sign_in).

```python
 log_forecast = client.forecast(
      df=train_data,
      h=28,
      level=[80],
      model="timegpt-1-long-horizon",
      time_col="ds",
      target_col="y",
      id_col="unique_id",
  )
```

In this code:

- `h=28` specifies the forecast horizon of 28 days.
- `level=[80, 95]` indicates the prediction intervals at 80% and 95% confidence levels.
- `model="timegpt-1-long-horizon"` sets the specific TimeGPT variant optimized for long-range forecasting.
- `time_col="ds"`, `target_col="y"`, and `id_col="unique_id"` explicitly identify the time, target, and ID columns in the input DataFrame.

## Evaluating the Forecast

Before evaluating forecast accuracy, we reverse the log transformation to bring predictions back to the original scale:

```python
def reverse_log_transform(df):
    df = df.copy()
    value_cols = [col for col in df if col not in ["ds", "unique_id"]]
    df[value_cols] = np.exp(df[value_cols]) - 1
    return df

base_forecast = reverse_log_transform(log_forecast)
base_forecast.head()
```

| unique_id   | ds                  | TimeGPT  | TimeGPT-hi-80 | TimeGPT-lo-80 |
| ----------- | ------------------- | -------- | ------------- | ------------- |
| FOODS_1_001 | 2016-05-23 00:00:00 | 0.374683 | 1.165591      | -0.127373     |
| FOODS_1_001 | 2016-05-24 00:00:00 | 0.410514 | 0.901326      | 0.046401      |
| FOODS_1_001 | 2016-05-25 00:00:00 | 0.383807 | 1.420369      | -0.208830     |
| FOODS_1_001 | 2016-05-26 00:00:00 | 0.393173 | 1.350957      | -0.174408     |
| FOODS_1_001 | 2016-05-27 00:00:00 | 0.416641 | 1.084891      | -0.037422     |

To assess the accuracy of our forecasts, we define utility functions to merge the forecast with real observations and compute the mean MAE:

```python
def merge_forecast(real_data, forecast):
    merged_results = pd.merge(
        real_data, forecast, "left", ["unique_id", "ds"]
    )
    return merged_results

def get_mean_mae(real_data, forecast):
    merged_results = merge_forecast(real_data, forecast)
    model_evaluation = evaluate(
        merged_results,
        metrics=[mae],
        models=["TimeGPT"],
        target_col="y",
        id_col="unique_id",
    )
    return model_evaluation.groupby("metric")["TimeGPT"].mean()["mae"]

base_mae = get_mean_mae(test_data, base_forecast)
print(base_mae)
```

Output:

```
0.5140717205912171
```

## Forecasting with Fine-Tuned TimeGPT

We can further improve results by [fine-tuning](https://www.nixtla.io/docs/capabilities-forecast-fine_tuning) TimeGPT with the train dataset.

Fine-tuning adapts the pre-trained TimeGPT model to a specific dataset or task by performing additional training. While TimeGPT can already forecast in a zero-shot fashion (no extra data required), fine-tuning with your custom dataset often boosts accuracy.

```python
 log_finetuned_forecast = client.forecast(
      df=train_data,
      h=28,
      level=[80],
      finetune_steps=10,
      finetune_loss="mae",
      model="timegpt-1-long-horizon",
      time_col="ds",
      target_col="y",
      id_col="unique_id",
  )
```

- `finetune_steps=10`: Instructs the model to perform 10 gradient update steps on the training data, enabling it to better capture dataset-specific patterns.
- `finetune_loss="mae"`: Sets the optimization objective to Mean Absolute Error, which is well-suited for datasets with many zeros and sharp demand changes, such as intermittent series.

## Evaluating the Fine-Tuned Forecast

We evaluate the fine-tuned forecasts using the same MAE metric.

```python
finetuned_forecast = reverse_log_transform(log_finetuned_forecast)
finedtune_mae = get_mean_mae(test_data, finetuned_forecast)
print(finedtune_mae)
```

Output:

```
0.492569421499707
```

The fine-tuned model achieves a lower MAE than the base model, indicating improved forecasting accuracy after adapting to the training data.

## Incorporating Exogenous Variables

To enhance the forecast, we include future [exogenous variables](https://www.nixtla.io/docs/capabilities-forecast-add_exogenous_variables) like calendar events that affect demand but aren't part of the target series. These features add helpful context—such as holidays or promotions—so the model can make more informed predictions.

We pull them from test_data to ensure they align with the forecast period and product IDs.

```python
non_exogenous_variables = ["y", "sell_price"]
futr_exog_data = test_data.drop(non_exogenous_variables, axis=1)
futr_exog_data.head()
```

|      | unique_id   | ds                  | event_type_Cultural | event_type_National | event_type_Religious | event_type_Sporting |
| ---- | ----------- | ------------------- | ------------------- | ------------------- | -------------------- | ------------------- |
| 1941 | FOODS_1_001 | 2016-05-23 00:00:00 | 0                   | 0                   | 0                    | 0                   |
| 1942 | FOODS_1_001 | 2016-05-24 00:00:00 | 0                   | 0                   | 0                    | 0                   |
| 1943 | FOODS_1_001 | 2016-05-25 00:00:00 | 0                   | 0                   | 0                    | 0                   |
| 1944 | FOODS_1_001 | 2016-05-26 00:00:00 | 0                   | 0                   | 0                    | 0                   |
| 1945 | FOODS_1_001 | 2016-05-27 00:00:00 | 0                   | 0                   | 0                    | 0                   |

We can incorporate these exogenous variables into the forecast by passing them as `X_df` to the `forecast()` method:

```python
log_exogenous_forecast = client.forecast(
    df=train_data,
    X_df=futr_exog_data,
    h=28,
    level=[80],
    finetune_steps=10,
    finetune_loss="mae",
    model="timegpt-1-long-horizon",
    time_col="ds",
    target_col="y",
    id_col="unique_id",
)
```

## Evaluating Forecasts with Exogenous Variables

We again evaluate the forecasts using the MAE to determine whether the inclusion of exogenous features has improved performance.

```python
exogenous_forecast = reverse_log_transform(log_exogenous_forecast)
exogenous_mae = get_mean_mae(test_data, exogenous_forecast)
print(exogenous_mae)
```

Output:

```
0.48701362689497757
```

Including event-type features reduced the MAE to 0.487—slightly better than the fine-tuned forecast (0.492) and clearly better than the base model (0.514), confirming their value in improving forecast precision.

## MAE Comparison Table

Below is a comparison of the mean MAE across the base, fine-tuned, and exogenous-enhanced TimeGPT variants:

```python
# Define the mean absolute error (MAE) values for different TimeGPT variants
mae_values = {
    "Model Variant": ["Base TimeGPT", "Fine-Tuned TimeGPT", "TimeGPT with Exogenous"],
    "MAE": [base_mae, finedtune_mae, exogenous_mae]
}

mae_table = pd.DataFrame(mae_values)
mae_table
```

| Model Variant          | MAE          |
| ---------------------- | ------------ |
| Base TimeGPT           | 0.5140717206 |
| Fine-Tuned TimeGPT     | 0.4925758924 |
| TimeGPT with Exogenous | 0.4870164269 |

## Conclusion

Forecasting intermittent demand is challenging due to the sporadic nature of sales data. However, TimeGPT demonstrates superior performance in handling such scenarios, providing more accurate and reliable forecasts.

By incorporating fine-tuning and accommodating exogenous variables, TimeGPT proves to be a valuable tool for data scientists dealing with intermittent demand forecasting.

For more detailed information and additional use cases, refer to [the official Nixtla documentation](https://www.nixtla.io/docs/intro).
