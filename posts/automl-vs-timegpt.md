---
title: "Still using Prophet or AutoARIMA for forecasting?"
seo_title: TimeGPT vs Databricks AutoML Forecasting Benchmark — Energy, Weather, Retail
description: "We benchmarked TimeGPT-2.1 against Databricks AutoML on electricity, weather, and retail demand datasets, comparing accuracy, runtime, and per-series win rates. TimeGPT-2.1 improved MAE, RMSE, MAPE, and WAPE across all three benchmarks."
image: "/images/automl-benchmark/results_overview.png"
categories: ["Time Series Forecasting"]
tags:
  - TimeGPT
  - Databricks
  - AutoML
  - benchmark
  - electricity forecasting
  - weather forecasting
  - retail forecasting
author_name: Nixtla Team
author_position: Nixtla
publication_date: 2026-05-28
---

Prophet and AutoARIMA are workhorses. They've powered production forecasting systems for decades, and tools like Databricks AutoML have made them even more accessible: automatic model selection, tuned configurations, reproducible notebooks.

However, in recent years, foundation models have changed the baseline. TimeGPT-2.1 requires no training, no feature engineering, and no hyperparameter search, yet consistently outperforms a tuned AutoML pipeline across multiple domains. We ran the numbers to see by exactly how much.

## What We Tested

[Databricks AutoML](https://docs.databricks.com/en/machine-learning/automl/index.html) searches over AutoARIMA and Prophet configurations, selects the best trial, and generates a reproducible training notebook. In these runs, the selected AutoML trial was Prophet.

We benchmarked it against TimeGPT-2.1 on three public datasets spanning different forecasting domains, evaluating both methods on identical rolling forecast windows with the same held-out data.

| Dataset | Domain | Frequency | Series | Horizon | Windows |
|---|---|---|---|---|---|
| PJM electricity load | Energy | Hourly | 30 | 24h | 7 |
| NOAA ISD weather | Weather | Hourly | 491 | 24h | 7 |
| M5 Walmart sales | Retail | Daily | 500 | 28 days | 1 |

## Energy: PJM Electricity Load

Electricity demand is a textbook use case for classical forecasting: strong seasonality, relatively stable patterns, limited noise. 

**Accuracy:**

| Model | MAE | RMSE | MAPE | WAPE | Bias |
|---|---|---|---|---|---|
| AutoML (Prophet) | 576.5 | 1,708.9 | 14.39% | 10.65% | −462.0 |
| TimeGPT-2.1 | **122.0** | **313.5** | **4.24%** | **2.25%** | 36.3 |

TimeGPT reduces MAPE from 14.39% to 4.24% — a 70% reduction in percentage error. The bias figure for AutoML (−462) also reveals a consistent systematic under-forecast across the series.

**Per-series win rates:**

| Metric | AutoML Win Rate | TimeGPT Win Rate |
|---|---|---|
| MAE | 3.3% | **96.7%** |
| RMSE | 3.3% | **96.7%** |
| MAPE | 3.3% | **96.7%** |
| WAPE | 3.3% | **96.7%** |
| AbsBias | 13.3% | **86.7%** |

TimeGPT wins on 29 of 30 series for MAE, RMSE, MAPE, and WAPE, and on 26 of 30 series for absolute bias.

**Runtime:** AutoML took 13 minutes to train, plus 12 seconds for inference. TimeGPT took **20 seconds total**.

## Weather: NOAA ISD Temperature

The [NOAA Integrated Surface Database](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) contains hourly temperature readings from weather stations worldwide. With 491 independent series spanning different climates and observation patterns, this dataset tests how well each method generalizes at scale.

**Accuracy:**

| Model | MAE | RMSE | MAPE | WAPE | Bias |
|---|---|---|---|---|---|
| AutoML (Prophet) | 2.721 | 3.600 | 19.15% | 15.00% | 1.419 |
| TimeGPT-2.1 | **1.333** | **1.887** | **8.69%** | **7.35%** | **0.334** |

TimeGPT cuts MAE and RMSE roughly in half, and nearly halves MAPE as well. Across 491 series, the aggregate differences are large.

**Per-series win rates:**

| Metric | AutoML Win Rate | TimeGPT Win Rate |
|---|---|---|
| MAE | 4.9% | **95.1%** |
| RMSE | 7.9% | **92.1%** |
| MAPE | 3.9% | **96.1%** |
| WAPE | 4.9% | **95.1%** |
| AbsBias | 11.0% | **89.0%** |

**Runtime:** AutoML required 2 hours 46 minutes to train across 491 series, plus 3 minutes of inference — nearly 3 hours end-to-end. TimeGPT required **37 seconds**.

## Retail: M5 Walmart Sales

The [M5 Forecasting competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset is the hardest test in this benchmark. Daily retail demand is sparse, intermittent, and volatile — the kind of signal that resists clean seasonal decomposition. We selected 500 item-store series from the CA3 store group, forecasting 28 days ahead.

**Accuracy:**

| Model | MAE | RMSE | MAPE | WAPE | Bias |
|---|---|---|---|---|---|
| AutoML (Prophet) | 3.470 | 7.125 | 69.28% | 54.83% | −0.292 |
| TimeGPT-2.1 | **3.179** | **6.621** | **58.30%** | **50.23%** | −0.995 |

The margin is smaller here — retail demand is genuinely difficult for any model. TimeGPT still wins on the main aggregate error metrics — MAE, RMSE, MAPE, and WAPE — though AutoML has lower aggregate bias magnitude.

**Per-series win rates:**

| Metric | AutoML Win Rate | TimeGPT Win Rate |
|---|---|---|
| MAE | 32.0% | **68.0%** |
| RMSE | 40.6% | **59.4%** |
| MAPE | 25.4% | **74.6%** |
| WAPE | 32.0% | **68.0%** |
| AbsBias | 44.2% | **55.8%** |

Even on the most challenging dataset, TimeGPT wins on more than two-thirds of series by MAE and WAPE.

**Runtime:** AutoML took over 5 hours to train, plus 50 seconds for inference. TimeGPT took **37 seconds**. At that scale, AutoML's training time isn't a cost, it's a blocker. A 5-hour training run rules out same-day iteration, live reforecasting, and rapid experimentation entirely.

## The Full Picture

| Dataset | AutoML Training | AutoML Inference | TimeGPT Inference | TimeGPT WAPE Improvement |
|---|---|---|---|---|
| PJM electricity (30 series) | 13 min | 12s | **20s** | 10.65% → 2.25% |
| NOAA weather (491 series) | 2h 46min | 3min | **37s** | 15.00% → 7.35% |
| M5 retail (500 series) | **5+ hours ** | 50s | **37s** | 54.83% → 50.23% |

TimeGPT has no training column — there is no training step.

Across all three domains, the pattern is the same: TimeGPT improves the main aggregate error metrics — MAE, RMSE, MAPE, and WAPE — and is much faster to deploy. The accuracy gap is largest where data is cleanest and most structured (energy, weather) and narrower on noisy intermittent retail data, with potential improvement available by adding finetune steps.

## Why This Happens

AutoML's ceiling is defined by its candidate models. When the search space is AutoARIMA and Prophet, the best possible outcome is a well-tuned classical model. Foundation models like TimeGPT-2.1 have been pretrained on a large, diverse corpus of real-world time series — they bring that learned prior to every new dataset, zero-shot.

The result is a model that already understands seasonality, trend, and temporal dynamics before it sees a single row of your data.

If you're still choosing between Prophet and AutoARIMA as your starting point, there's now a faster and more accurate enterprise solution available.

[Get started with TimeGPT →](https://nixtla.io/free-trial?utm_source=nixtla.io&utm_campaign=/blog/still-using-prophet)
