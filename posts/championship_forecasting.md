---
title: "Forecasting Championship Results using Time Series and Nixtla"
description: "Learn how to forecast championship standings using Nixtla's StatsForecast library."
image: "/images/title_image.svg"
categories: ["TimeGPT Forecasting"]
tags:
  - StatsForecast
  - forecasting
  - championship
  - sports analytics
  - time series
  - AutoARIMA
author_name: Piero Paialunga
author_image: "/images/authors/piero.jpg"
author_position: Data Scientist
publication_date: 2025-01-12
---

## Introduction
  
In the real world, we often face forecasting problems in environments where several teams, departments, or companies are competing and building up their performance over time. Think about how sales branches stack up revenue throughout the year, how factories compare production outputs, or how bids play out during a procurement cycle. In all these settings, we’re not just interested in the numbers themselves, but in how each competitor measures up against the rest as the results accumulate.

**Championship tournaments** provide an excellent case study for this pattern. Unlike simple time series where we forecast a single variable in isolation, championship-style data creates unique challenges:

- **Panel structure**: Multiple entities (teams, plants, branches) tracked simultaneously
- **Cumulative metrics**: Performance compounds over time (points, sales, production)
- **Fixed horizon**: A predetermined endpoint where final rankings matter
- **Historical patterns**: Entities establish consistent performance trajectories

These characteristics appear across industries. In manufacturing, production lines accumulate defect rates or output volumes over quarters. In finance, regional offices accumulate sales targets. In logistics, distribution centers accumulate delivery performance metrics. Understanding how to forecast these **cumulative, competitive time series** has broad applications beyond sports.

In this blog post, we'll use a championship tournament as our example system to demonstrate how **Nixtla's forecasting algorithms** can predict final outcomes by analyzing **cumulative performance time series**. The same methodology applies whenever you need to forecast how multiple entities will perform relative to each other over a defined period.

To accomplish this, we'll follow a systematic approach:

1. **Prepare the Data**: Generate a simulated championship with cumulative points time series for each team
2. **Hold Out Last N Matches**: Keep final matches for evaluation
3. **Train Forecast Model**: Fit the model on matches 1 to T−N using **StatsForecast** and **AutoARIMA**
4. **Predict Last N Outcomes**: Generate forecasts for the remaining matches
5. **Evaluate and Visualize Results**: Compare predictions with actual outcomes and assess forecast accuracy

The setup is summarized in the following chart:

![](/images/championship_forecasting/workflow.svg)

It seems like we have a lot to cover. Let's get to it!

### 1. Setup Championship Teams and Matches

To generate realistic championship data, we need to model teams with different strengths and simulate match outcomes. The key concepts are:

1. **Team strength parameters**: Each team gets a strength value that influences their scoring ability
2. **Poisson match model**: Goals are generated using a Poisson distribution based on team strengths
3. **Home advantage**: Home teams get a slight boost in expected goals

The core logic uses a Poisson process where expected goals depend on:
- Team strength differential
- Home advantage (typically ~0.3 goals)
- Base scoring rate (~1.35 goals per team)

Match outcomes translate to points: **Win = 3 points**, **Draw = 1 point**, **Loss = 0 points**.

### 2. Generate Championship Schedule

For a valid championship, each team must play every other team exactly twice (once home, once away). We use the **circle method** algorithm:

1. First half of season: N-1 rounds with rotating pairings
2. Second half: Mirror of first half (swap home/away)
3. Validation: Each team plays N-1 home games and N-1 away games

For 20 teams, this creates **38 matchdays** with **380 total matches**.

**Sample Output:**

```
Rounds: 38; Matches total: 380 (should be 38 & 380)

Matchday 1
Team12 vs Team08
Team19 vs Team06
Team17 vs Team03
Team02 vs Team13
Team11 vs Team07
...
```

### 3. Simulate Results and Build Time Series

Now we put everything together: simulate matches, track cumulative statistics, and transform the data into a **panel time series** ready for forecasting.

The key transformation is converting match-by-match results into a **cumulative points time series** for each team:

- **Panel structure**: `unique_id` (team), `ds` (matchday), `y` (cumulative points)
- **Cumulative metrics**: Points, goals for/against, wins/draws/losses accumulate over time
- **Train/test split**: Hold out final matchdays for evaluation

This structure is exactly what Nixtla's forecasting libraries expect and is analogous to tracking cumulative sales across branches, production output across facilities, or any competitive metric across entities.

> **Full implementation**: For the complete code covering team setup, calendar generation, match simulation, and data transformation, see the [championship_forecasting.ipynb notebook](../examples/notebooks/championship_forecasting.ipynb).

**Running the simulation:** 


```python
teams = [f"Team{i:02d}" for i in range(1, 21)]
season = generate_calendar(teams, seed=2025, shuffle_rounds=True)
strengths = make_tiered_strengths(teams)

# 1) Full season → dataframes for plots + forecasting
full_season_results = prepare_forecasting_data(teams, season, strengths, seed=777)
matches_df = full_season_results["matches_df"]
full_season_ts = full_season_results["ts_df"]  # (unique_id, ds, y) ready for StatsForecast/TimeGPT
standings_df = full_season_results["standings_df"]

# 2) Train on first 20 matchdays, forecast remaining 18
train_data = prepare_forecasting_data(teams, season, strengths, seed=777, cutoff_matchday=35)
train_ts = train_data["ts_df"]  # ds ∈ [1..20]
forecast_horizon = train_data["h"]  # 18 matchdays remaining
``` 

The following assumptions are made:
1. We are considering 20 teams (so 38 matchdays per team, 380 matches total).
2. We are training on the first 35 matchdays and predicting the last 3.
3. Thanks to the structure of the output, we can train on part of the championship and predict the final championship results and standings.



|     | unique_id   |   ds |   y |   pts | opponent   | ha   |   goals_for |   goals_against | result   |   cum_gf |   cum_ga |   cum_gd |   cum_w |   cum_d |   cum_l |
|----:|:------------|-----:|----:|------:|:-----------|:-----|------------:|----------------:|:---------|---------:|---------:|---------:|--------:|--------:|--------:|
|   0 | Team01      |    1 |   3 |     3 | Team20     | H    |           6 |               0 | W        |        6 |        0 |        6 |       1 |       0 |       0 |
|  20 | Team01      |    2 |   6 |     3 | Team09     | H    |           4 |               1 | W        |       10 |        1 |        9 |       2 |       0 |       0 |
|  40 | Team01      |    3 |   9 |     3 | Team11     | H    |           5 |               1 | W        |       15 |        2 |       13 |       3 |       0 |       0 |
|  60 | Team01      |    4 |  10 |     1 | Team07     | H    |           0 |               0 | D        |       15 |        2 |       13 |       3 |       1 |       0 |
|  80 | Team01      |    5 |  13 |     3 | Team12     | H    |           5 |               3 | W        |       20 |        5 |       15 |       4 |       1 |       0 |
| 117 | Team01      |    6 |  16 |     3 | Team19     | A    |           7 |               1 | W        |       27 |        6 |       21 |       5 |       1 |       0 |
| 120 | Team01      |    7 |  16 |     0 | Team10     | H    |           1 |               2 | L        |       28 |        8 |       20 |       5 |       1 |       1 |
| 140 | Team01      |    8 |  19 |     3 | Team14     | H    |           3 |               1 | W        |       31 |        9 |       22 |       6 |       1 |       1 |
| 160 | Team01      |    9 |  22 |     3 | Team04     | H    |           5 |               2 | W        |       36 |       11 |       25 |       7 |       1 |       1 |
| 180 | Team01      |   10 |  25 |     3 | Team05     | H    |           2 |               1 | W        |       38 |       12 |       26 |       8 |       1 |       1 |
| 200 | Team01      |   11 |  28 |     3 | Team13     | H    |           4 |               1 | W        |       42 |       13 |       29 |       9 |       1 |       1 |
| 220 | Team01      |   12 |  31 |     3 | Team06     | H    |           6 |               0 | W        |       48 |       13 |       35 |      10 |       1 |       1 |
| 240 | Team01      |   13 |  34 |     3 | Team17     | H    |           6 |               0 | W        |       54 |       13 |       41 |      11 |       1 |       1 |
| 273 | Team01      |   14 |  37 |     3 | Team16     | A    |           5 |               0 | W        |       59 |       13 |       46 |      12 |       1 |       1 |
| 280 | Team01      |   15 |  40 |     3 | Team03     | H    |           3 |               1 | W        |       62 |       14 |       48 |      13 |       1 |       1 |
| 315 | Team01      |   16 |  43 |     3 | Team18     | A    |           4 |               1 | W        |       66 |       15 |       51 |      14 |       1 |       1 |
| 320 | Team01      |   17 |  46 |     3 | Team02     | H    |           2 |               0 | W        |       68 |       15 |       53 |      15 |       1 |       1 |
| 347 | Team01      |   18 |  47 |     1 | Team08     | A    |           0 |               0 | D        |       68 |       15 |       53 |      15 |       2 |       1 |
| 373 | Team01      |   19 |  50 |     3 | Team15     | A    |           3 |               0 | W        |       71 |       15 |       56 |      16 |       2 |       1 |
| 380 | Team01      |   20 |  53 |     3 | Team08     | H    |           2 |               1 | W        |       73 |       16 |       57 |      17 |       2 |       1 |
| 415 | Team01      |   21 |  56 |     3 | Team12     | A    |           1 |               0 | W        |       74 |       16 |       58 |      18 |       2 |       1 |
| 439 | Team01      |   22 |  59 |     3 | Team20     | A    |           5 |               1 | W        |       79 |       17 |       62 |      19 |       2 |       1 |
| 447 | Team01      |   23 |  62 |     3 | Team06     | A    |           2 |               1 | W        |       81 |       18 |       63 |      20 |       2 |       1 |
...
| 637 | Team20      |   32 |  18 |     0 | Team14     | A    |           1 |               4 | L        |       26 |       91 |      -65 |       4 |       6 |      22 |
| 657 | Team20      |   33 |  18 |     0 | Team13     | A    |           2 |               4 | L        |       28 |       95 |      -67 |       4 |       6 |      23 |
| 678 | Team20      |   34 |  19 |     1 | Team15     | H    |           0 |               0 | D        |       28 |       95 |      -67 |       4 |       7 |      23 |
| 683 | Team20      |   35 |  19 |     0 | Team04     | A    |           1 |               5 | L        |       29 |      100 |      -71 |       4 |       7 |      24 |



### 4. Predict and Forecast with StatsForecast

Now that we have all the data, we can let StatsForecast do the magic. In particular, we will use the AutoARIMA feature to train and forecast the last three matches for the entire championship. 

The whole thing can be done in literally three lines of code:

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

sf = StatsForecast(models=[AutoARIMA()], freq=1)
sf.fit(train_ts)

forecast_raw = sf.predict(h=forecast_horizon, level=[95])
```

### 5. Evaluate the Results


The championship forecast outputs are stored in `forecast_raw`. Let's use the following helpers to display the quality of our predictions.


```python
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Sequence

def _conform_forecast_df(
    fcst: pd.DataFrame,
    team: str,
    ds_col: str = "ds",
    mean_col: Optional[str] = None,
    lo_col: Optional[str] = None,
    hi_col: Optional[str] = None,
    model_name: Optional[str] = None,
    level: Optional[int] = None,
) -> pd.DataFrame:
    """
    Normalize a forecast frame to columns: ds, yhat, yhat_lo, yhat_hi for a single team.

    Works with:
      - Generic frames already named: 'yhat', 'yhat_lo', 'yhat_hi'
      - StatsForecast output (wide): columns like ['AutoARIMA', 'AutoARIMA-lo-80', 'AutoARIMA-hi-80']
      - Any custom naming if you pass mean_col/lo_col/hi_col explicitly.
    """
    g = fcst[fcst["unique_id"] == team].copy()

    # If user specified columns, use them.
    if mean_col:
        g = g.rename(columns={mean_col: "yhat"})
        if lo_col: g = g.rename(columns={lo_col: "yhat_lo"})
        if hi_col: g = g.rename(columns={hi_col: "yhat_hi"})
    else:
        # Try common names
        if "yhat" in g.columns:
            pass
        else:
            # StatsForecast wide format
            # Guess model name if not provided: take the first non-id/ds column
            if model_name is None:
                candidate_cols = [c for c in g.columns if c not in {"unique_id", ds_col}]
                model_name = candidate_cols[0] if candidate_cols else None
            # Guess level if not provided: prefer 95, fall back to 80
            if level is None:
                level = 95 if f"{model_name}-lo-95" in g.columns else (80 if f"{model_name}-lo-80" in g.columns else None)

            mapping = {}
            if model_name and model_name in g.columns:
                mapping[model_name] = "yhat"
            if model_name and level is not None:
                lo_name = f"{model_name}-lo-{level}"
                hi_name = f"{model_name}-hi-{level}"
                if lo_name in g.columns: mapping[lo_name] = "yhat_lo"
                if hi_name in g.columns: mapping[hi_name] = "yhat_hi"
            g = g.rename(columns=mapping)

    keep = ["unique_id", ds_col, "yhat"] + [c for c in ["yhat_lo", "yhat_hi"] if c in g.columns]
    g = g[keep].rename(columns={ds_col: "ds"})
    return g

def plot_team_cumpoints_with_forecast(
    ts_df: pd.DataFrame,
    team: str,
    fcst_df: Optional[pd.DataFrame] = None,
    *,
    ds_col: str = "ds",
    y_col: str = "y",
    mean_col: Optional[str] = None,
    lo_col: Optional[str] = None,
    hi_col: Optional[str] = None,
    model_name: Optional[str] = None,
    level: Optional[int] = None,
    title: Optional[str] = None,
    show: bool = True,
):
    # Actuals (all available ds for the team)
    act = ts_df.loc[ts_df["unique_id"] == team, [ds_col, y_col]].sort_values(ds_col).rename(
        columns={ds_col: "ds", y_col: "y"}
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(act["ds"].values, act["y"].values, marker="o", linewidth=1.5, label="Actual cum. points")

    # Optional forecast
    if fcst_df is not None and len(fcst_df):
        g = _conform_forecast_df(
            fcst_df, team,
            ds_col=ds_col, mean_col=mean_col, lo_col=lo_col, hi_col=hi_col,
            model_name=model_name, level=level
        )
        # Shade interval if present
        if {"yhat_lo", "yhat_hi"}.issubset(g.columns):
            ax.fill_between(g["ds"].values, g["yhat_lo"].values, g["yhat_hi"].values, alpha=0.2, label="Prediction interval", color = 'lime')

        ax.plot(g["ds"].values, g["yhat"].values, linestyle="--", linewidth=1.8, label="Forecast mean", color = 'lime')

        # Draw a vertical line at the last observed ds (split point)
        if len(act):
            split = 36
            ax.axvline(split, linestyle=":", alpha=0.6)
            ax.text(split, ax.get_ylim()[1], " Train/Forecast Split", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Match Day Number", fontsize = 15)
    ax.set_ylabel("Championship Points", fontsize = 15)
    ax.set_title(title or f"{team}: Championship Points & Forecast", fontsize = 15)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize = 15)
    if show:
        plt.savefig('/Users/pieropaialunga/Desktop/blog/images/championship_forecasting/title-image.svg')
        plt.show()
    return fig, ax


def round_forecast_to_valid_points(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Round forecast values to integers since football points must be whole numbers.
    
    In football, you can only earn 0, 1 (draw), or 3 (win) points per match.
    Therefore, cumulative points must always be integers.
    
    This function rounds all forecast columns (mean, lower/upper bounds) to the nearest integer.
    """
    df = forecast_df.copy()
    for col in df.columns:
        if col not in ['unique_id', 'ds']:
            df[col] = df[col].round().astype(int)
    return df
```

And display the results using the following block of code:

```python
# Round to valid integer points (football only allows 0, 1, or 3 points per match)
forecast = round_forecast_to_valid_points(forecast_raw)

# Add actual values to compare with predictions
full_season_results = prepare_forecasting_data(teams, season, strengths, seed=777)
full_season_ts = full_season_results["ts_df"][["unique_id", "ds", "y"]]

# Merge actual values into the forecast dataframe
forecast = forecast.merge(
    full_season_ts.rename(columns={"y": "actual"}),
    on=["unique_id", "ds"],
    how="left"
)
plot_team_cumpoints_with_forecast(
    ts_df=full_season_results["ts_df"],  # full actuals for context
    team="Team01",
    fcst_df=forecast,
    model_name="AutoARIMA",  # tell the helper how to read the wide columns
    level=95
)
```

This is the output for `Team01`:

![Cumulative Points Forecast Example](/images/championship_forecasting/title-image.svg)

And this is how the predictions look (`forecast` for the full championship):

|    | unique_id   |   ds |   AutoARIMA |   AutoARIMA-lo-95 |   AutoARIMA-hi-95 |   actual |   error |   abs_error |   squared_error |
|---:|:------------|-----:|------------:|------------------:|------------------:|---------:|--------:|------------:|----------------:|
|  0 | Team01      |   36 |          89 |                87 |                92 |       90 |      -1 |           1 |               1 |
|  1 | Team01      |   37 |          92 |                89 |                95 |       93 |      -1 |           1 |               1 |
|  2 | Team01      |   38 |          94 |                91 |                98 |       94 |       0 |           0 |               0 |
|  3 | Team02      |   36 |          83 |                81 |                85 |       84 |      -1 |           1 |               1 |
|  4 | Team02      |   37 |          86 |                83 |                89 |       87 |      -1 |           1 |               1 |
|  5 | Team02      |   38 |          88 |                85 |                92 |       87 |       1 |           1 |               1 |
|  6 | Team03      |   36 |          78 |                76 |                81 |       79 |      -1 |           1 |               1 |
|  7 | Team03      |   37 |          81 |                77 |                84 |       82 |      -1 |           1 |               1 |
|  8 | Team03      |   38 |          83 |                78 |                88 |       85 |      -2 |           2 |               4 |
|  9 | Team04      |   36 |          85 |                83 |                88 |       86 |      -1 |           1 |               1 |
| 10 | Team04      |   37 |          88 |                85 |                91 |       89 |      -1 |           1 |               1 |
| 11 | Team04      |   38 |          90 |                86 |                94 |       90 |       0 |           0 |               0 |
| 12 | Team05      |   36 |          72 |                69 |                75 |       70 |       2 |           2 |               4 |
| 13 | Team05      |   37 |          75 |                70 |                79 |       73 |       2 |           2 |               4 |
| 14 | Team05      |   38 |          78 |                72 |                84 |       76 |       2 |           2 |               4 |
| 15 | Team06      |   36 |          67 |                64 |                70 |       68 |      -1 |           1 |               1 |
| 16 | Team06      |   37 |          69 |                65 |                73 |       71 |      -2 |           2 |               4 |
| 17 | Team06      |   38 |          71 |                66 |                75 |       74 |      -3 |           3 |               9 |
| 18 | Team07      |   36 |          73 |                70 |                76 |       71 |       2 |           2 |               4 |
| 19 | Team07      |   37 |          75 |                71 |                79 |       74 |       1 |           1 |               1 |
| 20 | Team07      |   38 |          77 |                72 |                82 |       77 |       0 |           0 |               0 |
| 21 | Team08      |   36 |          71 |                68 |                74 |       70 |       1 |           1 |               1 |
| 22 | Team08      |   37 |          73 |                69 |                77 |       73 |       0 |           0 |               0 |
...
| 56 | Team19      |   38 |          23 |                19 |                27 |       21 |       2 |           2 |               4 |
| 57 | Team20      |   36 |          20 |                18 |                21 |       19 |       1 |           1 |               1 |
| 58 | Team20      |   37 |          20 |                17 |                23 |       19 |       1 |           1 |               1 |
| 59 | Team20      |   38 |          21 |                17 |                24 |       19 |       2 |           2 |               4 |

Thanks to the power of StatsForecast and AutoARIMA, we are able to predict the full championship in a few seconds, together with the prediction intervals and the average prediction for each team in the championship. 

### Conclusions

Let's recap what we covered in this post:

- **We built a synthetic championship simulation** that mimics real-world tournament dynamics: teams with different strength levels, a complete double round-robin schedule, and match results generated using a Poisson model that accounts for team strengths and home advantage.

- **We transformed match results into cumulative points time series** for each team. By tracking cumulative points across matchdays, we created a panel dataset where each team's performance evolves over time, perfect for time series forecasting.

- **We trained a forecasting model on historical performance** by holding out the final 3 matchdays from our 38-match season. This allowed us to use matches 1-35 for training and evaluate our predictions on the actual final results.

- **We leveraged StatsForecast and AutoARIMA** to automatically select and fit the best ARIMA model for each team's cumulative points trajectory. The model generated point forecasts along with 95% prediction intervals, giving us both expected outcomes and uncertainty ranges.

- **We validated the approach** by comparing predictions with actual results. The forecasts captured team performance patterns well, with most predictions within 1-2 points of the actual final standings, demonstrating that cumulative points follow predictable time series patterns throughout a championship.

Overall, this workflow shows how synthetic championship data, combined with Nixtla's forecasting models, can provide accurate predictions for the final matches of a tournament. By analyzing the cumulative points time series, we can forecast not just individual match outcomes, but the entire final standings, helping teams and fans understand where their championship race is heading before the final whistle.