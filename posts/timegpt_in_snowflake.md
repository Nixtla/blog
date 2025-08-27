---
title: "TimeGPT vs Snowflake - 50x Faster Forecasting with Better Accuracy"
description: Discover SQL-native time series forecasting for Snowflake that's 10x faster than native tools. Nixtla provides state-of-the-art accuracy without Python, ML infrastructure, or complex setup.
image: /images/timegpt_in_snowflake/performance_accuracy_quadrant.svg
categories: ["Time Series Forecasting"]
tags:
  - TimeGPT
  - Snowflake
  - SQL forecasting
  - performance comparison
author_name: Khuyen Tran
author_image: "/images/authors/khuyen.jpeg"
author_position: Developer Advocate - Nixtla
publication_date: 2025-08-26
---

Forecasting directly inside Snowflake is highly desirable: it keeps data in place, simplifies governance, and allows teams to use familiar SQL workflows. Business teams already use Snowflake for analytics, so adding forecasting into the same environment reduces friction and accelerates deployment.

However, native Snowflake forecasting tools fall short. They are difficult to configure, slow to run, and frequently produce inaccurate results—especially at scale.

Nixtla offers a better approach. It brings state-of-the-art forecasting models to Snowflake, fully accessible through SQL with no infrastructure setup or external orchestration required.

## Why Native Snowflake Forecasting Falls Short

**Snowflake's native forecasting capabilities** are hindered by the limits of SQL itself. SQL isn't designed for advanced machine learning workflows, and switching between SQL and Python is cumbersome for most teams.

Even when implemented, Snowflake's default forecasting tends to be slow, the syntax is unintuitive, and the results are often imprecise.

## Introducing Nixtla on Snowflake

**[Nixtla](https://nixtla.io) brings production-grade forecasting tools directly into Snowflake**, combining the ease of SQL with the power of state-of-the-art models.

Key features:

- **Fully SQL-based**: no Python, notebooks, or ML infrastructure required.
- **API-based or self-hosted**: run Nixtla from outside or _inside_ your Snowflake instance.
- **Fast, accurate forecasts**: purpose-built models for time series data.

## The Nixtla Forecasting Suite

Nixtla offers a full suite of time series tools, all accessible via SQL:

### 1. Zero-Shot Forecast

`NIXTLA_FORECAST` is the primary function for generating forecasts in Snowflake using only SQL. It produces accurate, out-of-the-box predictions without any model training or configuration:

```sql
SELECT *
FROM TABLE(
    NIXTLA_FORECAST(
        TABLE(table_name), {'h': 28}
    )
)
```

The `h` parameter controls the forecast horizon (e.g., 28 days). You can also specify the model explicitly, for example:

```sql
SELECT * FROM TABLE(
    NIXTLA_FORECAST(
        TABLE(table_name),
        {'h': 28, 'model': 'timegpt-1.5'}
    )
);

```

The input table used in `NIXTLA_FORECAST` must contain the following columns:

- `unique_id`: the identifier for each time series (VARCHAR)
- `ds`: the timestamp column (TIMESTAMP)
- `y`: the target variable to forecast (FLOAT)

If your dataset uses different column names, you should alias them appropriately before passing to the function.

```sql
SELECT * FROM TABLE(
    NIXTLA_FORECAST(
        TABLE(SELECT store_id AS unique_id, date AS ds, sales AS y FROM table_name),
        {'h': 28, 'model': 'timegpt-1.5'}
    )
);

```

### 2. Finetune

`NIXTLA_FINETUNE` allows you to adapt a pre-trained forecasting model to your specific dataset using a small number of training steps. This fine-tuning process helps improve forecast accuracy for time series with unique patterns or local behaviors:

```sql
CALL NIXTLA_FINETUNE(
    TABLE(table_name), {'finetune_steps': 10}
)
```

The input table should follow the same format required by `NIXTLA_FORECAST`, including the columns `unique_id`, `ds`, and `y`.

### 3. Evaluate

`NIXTLA_EVALUATE` is used to measure the performance of one or more forecast columns against actual values using metrics like MAPE or RMSE:

```sql
SELECT *
FROM TABLE(
    NIXTLA_EVALUATE(
        TABLE(table_name), ['mape', 'rmse']
    )
)
```

The input table for `NIXTLA_EVALUATE` should contain:

- `unique_id`: time series identifier
- `ds`: timestamp column
- `y`: actual observed values
- One or more forecast columns (e.g., `zeroshot`, `finetuned`, `sf_best`)

Each forecast column will be compared against `y` to compute the specified metrics.

## Syntax Comparison: Snowflake Forecasting Vs Nixtla Forecasting

Let's compare Nixtla and Snowflake forecasting in terms of syntax.

**Snowflake Syntax:**

Create a forecasting object in Snowflake using the 'best' method (auto-selects optimal model):

```sql
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sf_best(
    INPUT_DATA => TABLE(example_train),
    TIMESTAMP_COLNAME => 'ds',
    TARGET_COLNAME => 'y',
    SERIES_COLNAME => 'unique_id',
    CONFIG_OBJECT => {'method':'best'}
);
```

Generate a 28-step forecast using the 'sf_best' model and store the result:

```sql
CREATE OR REPLACE TEMP TABLE sf_best_forecast AS
SELECT
    TRIM(SERIES, '"') AS unique_id,
    ts AS ds,
    forecast
FROM TABLE(sf_best!FORECAST(FORECASTING_PERIODS => 28));
```

Snowflake's forecasting syntax is verbose and rigid, making it harder to adopt and maintain. Specifically:

- Forecasting requires several manual steps just to configure and retrieve predictions
- Named objects like forecasting models must be created and referenced explicitly
- Uses non-standard SQL constructs like `!FORECAST`, which break consistency and are difficult to remember
- Outputs need post-processing, such as trimming string-encoded fields like `SERIES`

**Nixtla Syntax:**

Run a zero-shot forecast using default settings:

```sql
CREATE OR REPLACE TEMP TABLE nixtla_0shot AS
SELECT * FROM TABLE(
    NIXTLA_FORECAST(TABLE(example_train), {'h': 28})
);
```

Apply column aliasing and type casting when your table doesn't match the required `unique_id`, `ds`, and `y` format:

```sql
CREATE OR REPLACE TEMP TABLE nixtla_0shot AS
SELECT * FROM TABLE(
    NIXTLA_FORECAST(
        TABLE(
            SELECT store_id AS unique_id, date AS ds, sales AS y
            FROM example_train
        ),
        {'h': 28}
    )
);
```

Nixtla's forecasting syntax:

- Follows conventional SQL structure for easier readability and learning
- Eliminates the need for creating persistent named objects
- Avoids proprietary or unfamiliar constructs, enabling faster adoption

## Performance Comparison: Snowflake Forecasting Vs Nixtla Forecasting

Let's compare Nixtla and Snowflake forecasting in terms of performance.

### Data Preparation

We will use daily CTA ridership data from the Chicago Transit Authority, available at [CTA Ridership - Daily Boarding Totals](https://catalog.data.gov/dataset/cta-ridership-daily-boarding-totals), to demonstrate public transit forecasting. This dataset contains 120,000+ records across 84 time series, making it ideal for testing forecasting performance at scale.

We have processed the original CTA ridership data and split it into training and test sets:

- **Training set**: 114,996 records at [train_ridership.parquet](https://github.com/Nixtla/nixtla_blog_examples/blob/main/data/train_ridership.parquet)
- **Test set**: 5,288 records at [test_ridership.parquet](https://github.com/Nixtla/nixtla_blog_examples/blob/main/data/test_ridership.parquet)

To upload these datasets to Snowflake, first create the tables with the required schema:

```sql
CREATE OR REPLACE TABLE EXAMPLE_TRAIN (
    UNIQUE_ID VARCHAR(16777216),
    DS TIMESTAMP_NTZ(9),
    Y FLOAT
);

CREATE OR REPLACE TABLE EXAMPLE_TEST (
    UNIQUE_ID VARCHAR(16777216),
    DS TIMESTAMP_NTZ(9),
    Y FLOAT
);
```

Then upload the parquet files to these tables using [Snowflake's web interface](https://docs.snowflake.com/en/user-guide/data-load-web-ui) or `COPY INTO` commands.

### Forecasting with Snowflake

Create a forecasting object in Snowflake using the `best` method:

```sql
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sf_best(
    INPUT_DATA => TABLE(example_train),
    TIMESTAMP_COLNAME => 'ds',
    TARGET_COLNAME => 'y',
    SERIES_COLNAME => 'unique_id',
    CONFIG_OBJECT => {'method':'best'}
);
```

Generate a 28-step forecast using the `sf_best` model:

```sql
CREATE OR REPLACE TEMP TABLE sf_best_forecast AS
SELECT
    TRIM(SERIES, '"') AS unique_id,
    ts AS ds,
    forecast
FROM TABLE(sf_best!FORECAST(FORECASTING_PERIODS => 28));
```

Create another forecasting object using the `fast` method:

```sql
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST sf_fast(
    INPUT_DATA => TABLE(example_train),
    TIMESTAMP_COLNAME => 'ds',
    TARGET_COLNAME => 'y',
    SERIES_COLNAME => 'unique_id',
    CONFIG_OBJECT => {'method':'fast'}
);
```

Generate a 28-step forecast using the 'sf_fast' model:

```sql
CREATE OR REPLACE TEMP TABLE sf_fast_forecast AS
SELECT
    TRIM(SERIES, '"') AS unique_id,
    ts AS ds,
    forecast
FROM TABLE(sf_fast!FORECAST(FORECASTING_PERIODS => 28));
```

### Forecasting with Nixtla

Run a forecast using [TimeGPT Long Horizon model](https://docs.nixtla.io/docs/tutorials-long_horizon_forecasting) explicitly, which is optimized for extended forecasting periods:

```sql
CREATE OR REPLACE TEMP TABLE nixtla_15 AS
SELECT * FROM TABLE(
    NIXTLA_FORECAST(
        TABLE(example_train),
        {'h':28, 'model': 'timegpt-1-long-horizon'}
    )
);
```

Fine-tune a model on the training dataset with a small number of steps:

```sql
CALL NIXTLA_FINETUNE(
    TABLE(example_train),
    {'finetune_steps':10}
);
```

This returns a model ID: `ce772812-7447-40d1-adfb-df50ff8b3fbe`

Run a forecast using the fine-tuned model, which is stored in the `finetuned_model_id` parameter:

```sql
CREATE OR REPLACE TEMP TABLE nixtla_finetuned AS
SELECT * FROM TABLE(
    NIXTLA_FORECAST(
        TABLE(example_train),
        {
            'h': 28,
            'model': 'timegpt-1-long-horizon',
            'finetuned_model_id': 'ce772812-7447-40d1-adfb-df50ff8b3fbe'
        }
    )
);
```

### Compare Forecasted Results

Combine the actual data with predictions from both Snowflake forecasting methods and all three Nixtla model variants into a single evaluation table:

```sql
CREATE OR REPLACE TEMP TABLE eval AS
SELECT
    example_all_data.*,
    nixtla_0shot.forecast AS zeroshot,
    nixtla_finetuned.forecast AS finetuned,
    nixtla_15.forecast AS onefive,
    sf_best_forecast.forecast AS sf_best,
    sf_fast_forecast.forecast AS sf_fast
FROM example_all_data
INNER JOIN nixtla_0shot USING (unique_id, ds)
INNER JOIN nixtla_finetuned USING (unique_id, ds)
INNER JOIN nixtla_15 USING (unique_id, ds)
INNER JOIN sf_best_forecast USING (unique_id, ds)
INNER JOIN sf_fast_forecast USING (unique_id, ds);
```

Evaluate MAPE (Mean Absolute Percentage Error) for all forecast methods and aggregate the results:

```sql
SELECT
    forecaster,
    metric,
    AVG(value) AS value
FROM TABLE(NIXTLA_EVALUATE(TABLE(eval), ['mape']))
GROUP BY 1, 2
ORDER BY 2, 3;
```

**Forecast Accuracy**

The table below shows the MAPE for all forecast methods. Lower MAPE values indicate better forecasting performance.

| Model                 | MAPE  |
| --------------------- | ----- |
| TimeGPT LH: Finetuned | 0.078 |
| TimeGPT LH: 0 Shot    | 0.089 |
| Snowflake Best        | 0.091 |
| Snowflake Fast        | 0.099 |

![forecast_accuracy_mape_by_model](/images/timegpt_in_snowflake/forecast_accuracy_mape_by_model.svg)

We can see that:

- TimeGPT Long Horizon with fine-tuning achieves the best accuracy at 7.8% MAPE
- Even zero-shot TimeGPT outperforms both Snowflake forecasting methods
- Fine-tuning provides a 12% accuracy improvement over the zero-shot approach
- Snowflake's "fast" method delivers the worst performance despite targeting speed
- All TimeGPT variants demonstrate superior forecasting capabilities compared to native Snowflake tools

**Runtime Performance**

We also measured how long each method took to train and forecast across 10 time series.

| Model                 | Time (sec) |
| --------------------- | ---------- |
| TimeGPT LH: Finetuned | 26         |
| TimeGPT LH: 0 Shot    | 19         |
| Snowflake Best        | 2499       |
| Snowflake Fast        | 1340       |

![runtime_performance_by_model](/images/timegpt_in_snowflake/runtime_performance_by_model.svg)

The table shows that:

- TimeGPT dramatically outperforms Snowflake in execution speed by 50-130x
- Zero-shot TimeGPT completes in 19 seconds compared to Snowflake's 22-42 minutes
- Fine-tuned TimeGPT takes 26 seconds, still dramatically faster than Snowflake
- Snowflake's "fast" method takes over 22 minutes, contradicting its speed claims
- TimeGPT's performance advantage becomes more pronounced at scale

As data volume increases, these performance differences compound, making Nixtla much better suited for production-scale forecasting.

**Quadrant Chart**

The following quadrant chart maps each model's position across the two critical dimensions: execution time and forecast accuracy. This visualization clearly demonstrates TimeGPT's superiority in both metrics.

![Performance vs Accuracy Quadrant](/images/timegpt_in_snowflake/performance_accuracy_quadrant.svg)

## Final Thoughts

TimeGPT transforms forecasting from a time-consuming bottleneck into a competitive advantage within Snowflake. The 50x speed improvement means analysts can iterate on forecasts in real-time rather than waiting hours for results, while the superior accuracy directly translates to better business decisions and reduced forecast error costs.

This performance leap enables forecasting to become part of daily analytics workflows rather than a separate, resource-intensive process.
