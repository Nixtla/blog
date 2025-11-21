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
  
If you follow any championship-based sport (like football, soccer, or basketball), you know that the end of the season is an extremely intense moment. Teams spend their entire season preparing for the final push, and it often happens that an entirely different tournament emerges in the last few matches. Some teams maintain their winning streak, others unexpectedly surge to the top, while some teams collapse and either give away championships or face relegation.

However, the end of the championship is rarely completely unexpected if you look at the bigger picture of the entire tournament. Typically, teams establish their **own patterns** throughout the season: some **consistently strong**, others **consistently struggling**. These patterns can be observed and analyzed from the **cumulative data** of the championship.

In this blog post, we'll explore how to use **Nixtla's forecasting algorithms** to predict the final matches of a championship by analyzing the **cumulative points time series** for each team. This approach allows us to leverage the **entire season's performance data** to forecast how teams will perform in their remaining matches.

To accomplish this, we'll follow a systematic approach:

1. **Setup Championship**: Define teams and match structure
2. **Generate Championship Schedule**: Create a complete tournament calendar
3. **Simulate Results**: Generate match outcomes with team strengths
4. **Track Results over Championship**: Monitor cumulative points across all matches
5. **Hold Out Last N Matches**: Keep final matches for evaluation
6. **Train Forecast Model**: Fit the model on matches 1 to T−N using **StatsForecast** and **AutoARIMA**
7. **Predict Last N Outcomes**: Generate forecasts for the remaining matches
8. **Evaluate and Visualize Results**: Compare predictions with actual outcomes and assess forecast accuracy

The setup is summarized in the following chart:

![](/images/championship_forecasting/workflow.svg)

It seems like we have a lot to cover. Let's get to it!

### 1. Setup Championship Teams and Matches

The first thing we need to do is define a "Team" in our Python environment. Here's what we will do:

1. We will build the `TeamRow` class that will keep track of the team statistics (goals scored, goals given, points, etc.).
2. Each team will have a strength parameter. The number of goals that a team will score in a match is related to that parameter.
3. We are giving a slight advantage to the home team versus the away team (usually, that is the case for real-life championships). 

This is the code implementation of the Teams and Matches:

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random
import numpy as np

@dataclass
class TeamRow:
    pts: int = 0
    gf: int = 0
    ga: int = 0
    gd: int = 0
    w: int = 0
    d: int = 0
    l: int = 0


# ----------------------------
# Strengths helpers
# ----------------------------
def make_tiered_strengths(teams: List[str]) -> Dict[str, float]:
    """
    Simple, reproducible strengths:
    - Strong top 6, solid next 6, average next 4, weaker bottom 4.
    Values are on a free scale; 0 = league-average.
    """
    # Customize to taste
    tiers = (
        +0.55, +0.45, +0.35, +0.30, +0.25, +0.20,   # top 6
        +0.10, +0.08, +0.06, +0.04, +0.02,  0.00,   # next 6
        -0.02, -0.04, -0.06,                       # next 4
        -0.20, -0.30, -0.45, -0.55                 # bottom 4
    )
    tiers = tiers if len(tiers) >= len(teams) else np.linspace(0.6, -0.6, len(teams))
    strengths = {t: float(tiers[i]) for i, t in enumerate(teams)}
    return strengths

# ----------------------------
# Poisson match model
# ----------------------------
def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's algorithm (fine for this use)
    L = np.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1

def simulate_match(home: str, away: str, strengths: Dict[str, float],
                   base_rate: float = 1.35, home_adv: float = 0.30,
                   rng: random.Random = None) -> Tuple[int, int, int, int]:
    """
    Returns (home_pts, away_pts, gh, ga)
    """
    if rng is None:
        rng = random.Random()
    s_h = strengths[home]
    s_a = strengths[away]
    xg_h = base_rate * np.exp(home_adv + (s_h - s_a))
    xg_a = base_rate * np.exp(0.0      + (s_a - s_h))
    gh = _poisson(rng, xg_h)
    ga = _poisson(rng, xg_a)

    if gh > ga:
        return 3, 0, gh, ga
    elif ga > gh:
        return 0, 3, gh, ga
    else:
        return 1, 1, gh, ga
```

### 2. Generate Championship Schedule

Now that we have the matches, we need to build the calendar. This is more straightforward:
1. We are generating random matches between teams.
2. We are making sure that if you play away on match number X, you are going to play home in match number X+1 (and vice versa).
3. Each team needs to play against all the other teams twice.

```python
from dataclasses import dataclass
from typing import List, Dict
import random
from collections import Counter

@dataclass(frozen=True)
class Match:
    home: str
    away: str

def generate_calendar(teams: List[str], seed: int = 42, shuffle_rounds: bool = True) -> List[List[Match]]:
    """
    Circle method (even N):
    - First half: N-1 rounds; in each round i, pair arr[j] vs arr[-1-j].
    - Rotate all except the first team: arr = [arr[0], arr[-1], arr[1], ..., arr[-2]]
    - Second half: mirror (swap home/away) of first half.
    Guarantees: one match/team/round; 19 home + 19 away per team.
    """
    assert len(teams) % 2 == 0, "Number of teams must be even."
    rng = random.Random(seed)
    arr = teams[:]
    rng.shuffle(arr)
    n = len(arr)
    half = n // 2

    rounds_first_half: List[List[Match]] = []
    for r in range(n - 1):
        # Pair fronts with backs
        round_pairs = []
        for j in range(half):
            a = arr[j]
            b = arr[-1 - j]
            # Alternate home/away by round and by pair index to help balance
            if (r + j) % 2 == 0:
                round_pairs.append(Match(home=a, away=b))
            else:
                round_pairs.append(Match(home=b, away=a))
        rounds_first_half.append(round_pairs)

        # Rotate all but the first item: [A, B, C, ..., Y, Z] -> [A, Z, B, C, ..., Y]
        if n > 2:
            arr = [arr[0]] + [arr[-1]] + arr[1:-1]

    # Mirror for second half (swap home/away)
    rounds_second_half = [[Match(home=m.away, away=m.home) for m in rnd] for rnd in rounds_first_half]

    # Optionally shuffle within halves to randomize matchday order (keeps validity)
    if shuffle_rounds:
        rng.shuffle(rounds_first_half)
        rng.shuffle(rounds_second_half)

    season = rounds_first_half + rounds_second_half
    _validate_calendar(season, teams)
    return season

def _validate_calendar(season: List[List[Match]], teams: List[str]) -> None:
    n = len(teams)
    assert len(season) == 2*(n-1), f"Expected {2*(n-1)} rounds, got {len(season)}."
    # Each round: every team appears once
    teamset = set(teams)
    for i, rnd in enumerate(season, 1):
        seen = set()
        for m in rnd:
            assert m.home in teamset and m.away in teamset and m.home != m.away
            assert m.home not in seen and m.away not in seen, f"Team plays twice in round {i}"
            seen.add(m.home); seen.add(m.away)
        assert len(seen) == n, f"Missing teams in round {i}"

    # Home/away exactly n-1 each; each ordered pair exactly once
    home_counts = Counter()
    away_counts = Counter()
    pair_counts = Counter()
    for rnd in season:
        for m in rnd:
            home_counts[m.home] += 1
            away_counts[m.away] += 1
            pair_counts[(m.home, m.away)] += 1

    for t in teams:
        assert home_counts[t] == (n-1), f"{t} home games: {home_counts[t]} != {n-1}"
        assert away_counts[t] == (n-1), f"{t} away games: {away_counts[t]} != {n-1}"

    for a in teams:
        for b in teams:
            if a == b: continue
            assert pair_counts[(a,b)] == 1, f"Pair {a} vs {b} appears {pair_counts[(a,b)]} times"
```

Let's give it a quick test:

```python
teams = [f"Team{i:02d}" for i in range(1, 21)]
season = generate_calendar(teams, seed = 2,shuffle_rounds=True)
print(f"Rounds: {len(season)}; Matches total: {sum(len(r) for r in season)} (should be 38 & 380)")
for md in range(2):
    print(f"\nMatchday {md+1}")
    for m in season[md]:
        print(f"{m.home} vs {m.away}")
```

```
Rounds: 38; Matches total: 380 (should be 38 & 380)

Matchday 1
Team12 vs Team08
Team19 vs Team06
Team17 vs Team03
Team02 vs Team13
Team11 vs Team07
Team18 vs Team05
Team15 vs Team09
Team20 vs Team10
Team04 vs Team16
Team14 vs Team01

Matchday 2
Team08 vs Team19
Team12 vs Team03
Team02 vs Team06
Team17 vs Team07
Team18 vs Team13
Team11 vs Team09
Team20 vs Team05
Team15 vs Team16
Team14 vs Team10
Team04 vs Team01
```

### 3. Simulate Results

To simulate the results, we will use the following helper functions. 

> *Note!* These functions implement the simulation logic we've already described. They don't introduce any new concepts, just put together the pieces we've already built (teams, matches, strengths, and the Poisson model).

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

# ---------- Run a season AND keep detailed logs ----------
def simulate_season_with_logs(
    teams: List[str],
    season: List[List[Match]],
    strengths: Dict[str, float],
    base_rate: float = 1.35,
    home_adv: float = 0.30,
    seed: int = 7
) -> Tuple[Dict[str, TeamRow], List[dict]]:
    """
    Returns:
      - table: final TeamRow per team
      - logs: list of dicts, one per match, including matchday & per-team stats
    """
    rng = random.Random(seed)
    table: Dict[str, TeamRow] = {t: TeamRow() for t in teams}
    logs: List[dict] = []

    for md, rnd in enumerate(season, 1):
        for m in rnd:
            p_h, p_a, gh, ga = simulate_match(m.home, m.away, strengths, base_rate, home_adv, rng)

            # update table
            table[m.home].pts += p_h
            table[m.away].pts += p_a
            table[m.home].gf  += gh; table[m.home].ga += ga
            table[m.away].gf  += ga; table[m.away].ga += gh
            if p_h == 3:
                table[m.home].w += 1; table[m.away].l += 1
            elif p_a == 3:
                table[m.away].w += 1; table[m.home].l += 1
            else:
                table[m.home].d += 1; table[m.away].d += 1

            # match-level logs (both perspectives)
            logs.append({
                "matchday": md, "home": m.home, "away": m.away,
                "gh": gh, "ga": ga,
                "pts_home": p_h, "pts_away": p_a
            })

    for t in teams:
        table[t].gd = table[t].gf - table[t].ga

    return table, logs

# ---------- Build tidy dataframes ----------
def build_match_df(logs: List[dict]) -> pd.DataFrame:
    """
    One row per match with basic info and result labels.
    """
    df = pd.DataFrame(logs).sort_values(["matchday", "home"])
    def result(gh, ga):
        if gh > ga: return "H"
        if ga > gh: return "A"
        return "D"
    df["result"] = np.where(df["gh"] > df["ga"], "H", np.where(df["ga"] > df["gh"], "A", "D"))
    return df

def build_team_timeseries_df(teams: List[str], match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Panel time series for forecasting.
    Returns columns:
      unique_id (team), ds (matchday 1..38), y (cumulative points),
      pts (points gained that round), gf, ga, gd (cumulative),
      w,d,l (cumulative), opponent, ha ('H'/'A'), goals_for, goals_against, result
    """
    rows = []
    # Expand to team perspective
    for _, r in match_df.iterrows():
        # Home row
        rows.append({
            "matchday": r.matchday, "team": r.home, "opponent": r.away, "ha": "H",
            "goals_for": r.gh, "goals_against": r.ga,
            "pts": 3 if r.gh > r.ga else (1 if r.gh == r.ga else 0),
            "result": "W" if r.gh > r.ga else ("D" if r.gh == r.ga else "L")
        })
        # Away row
        rows.append({
            "matchday": r.matchday, "team": r.away, "opponent": r.home, "ha": "A",
            "goals_for": r.ga, "goals_against": r.gh,
            "pts": 3 if r.ga > r.gh else (1 if r.ga == r.gh else 0),
            "result": "W" if r.ga > r.gh else ("D" if r.ga == r.gh else "L")
        })

    td = pd.DataFrame(rows).sort_values(["team", "matchday"])
    # Cumulative aggregates per team
    td["cum_pts"] = td.groupby("team")["pts"].cumsum()
    td["cum_gf"]  = td.groupby("team")["goals_for"].cumsum()
    td["cum_ga"]  = td.groupby("team")["goals_against"].cumsum()
    td["cum_gd"]  = td["cum_gf"] - td["cum_ga"]

    # Cumulative W/D/L (nice features if needed)
    td["w1"] = (td["result"] == "W").astype(int)
    td["d1"] = (td["result"] == "D").astype(int)
    td["l1"] = (td["result"] == "L").astype(int)
    td["cum_w"] = td.groupby("team")["w1"].cumsum()
    td["cum_d"] = td.groupby("team")["d1"].cumsum()
    td["cum_l"] = td.groupby("team")["l1"].cumsum()
    td.drop(columns=["w1","d1","l1"], inplace=True)

    # StatsForecast/TimeGPT-ready view
    ts = td.rename(columns={
        "team": "unique_id",
        "matchday": "ds",
        "cum_pts": "y"
    })[[
        "unique_id", "ds", "y",                 # <-- required for many Nixtla pipelines
        "pts", "opponent", "ha",
        "goals_for", "goals_against", "result",
        "cum_gf", "cum_ga", "cum_gd", "cum_w", "cum_d", "cum_l"
    ]]
    ts["ds"] = ts["ds"].astype(int)
    return ts, td

def build_standings_by_round(td: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a tidy standings table for every matchday:
    columns: matchday, pos, team, pts, gd, gf
    """
    # pick cumulative metrics at each round
    snap = td[["matchday", "team", "cum_pts", "cum_gd", "cum_gf"]].copy()
    snap = snap.rename(columns={"cum_pts":"pts", "cum_gd":"gd", "cum_gf":"gf"})
    # rank within each round
    snap["pos"] = snap.groupby("matchday") \
                      .apply(lambda g: g.sort_values(["pts","gd","gf"], ascending=False)
                                      .assign(pos=lambda x: np.arange(1, len(x)+1))) \
                      .reset_index(level=0, drop=True)["pos"]
    return snap.sort_values(["matchday", "pos"]).reset_index(drop=True)

# ---------- One-shot convenience wrapper ----------
def prepare_forecasting_data(
    teams: List[str],
    season: List[List[Match]],
    strengths: Dict[str, float],
    base_rate: float = 1.35,
    home_adv: float = 0.30,
    seed: int = 7,
    cutoff_matchday: int = None
):
    """
    Simulate a season and return:
      - matches_df: one row per match with scores and results
      - ts_df: timeseries dataframe ready for forecasting (columns: unique_id, ds, y, ...)
      - standings_df: positions for every team at every matchday
    If cutoff_matchday is provided (e.g., 20), trims the data to 1..cutoff_matchday (for training)
    and returns 'h' = remaining matchdays (38 - cutoff_matchday) as the forecast horizon.
    """
    final_table, logs = simulate_season_with_logs(
        teams, season, strengths, base_rate=base_rate, home_adv=home_adv, seed=seed
    )
    matches_df = build_match_df(logs)
    ts_df, team_detail_df = build_team_timeseries_df(teams, matches_df)
    standings_df = build_standings_by_round(team_detail_df)

    h = None
    if cutoff_matchday is not None:
        ts_df = ts_df[ts_df["ds"] <= cutoff_matchday].copy()
        standings_df = standings_df[standings_df["matchday"] <= cutoff_matchday].copy()
        h = 38 - int(cutoff_matchday)

    return {
        "matches_df": matches_df,      # match results (one row per match)
        "ts_df": ts_df,                # timeseries for forecasting: unique_id, ds, y (cumulative points)
        "standings_df": standings_df,  # team rankings at each matchday
        "h": h                         # forecast horizon (remaining matchdays) if cutoff was used
    }
```
  


Now that we have all the ingredients, we can easily simulate the whole championship in a few lines. This is how we run the full simulation: 


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