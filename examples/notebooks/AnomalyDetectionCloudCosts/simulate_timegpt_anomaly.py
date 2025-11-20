from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Literal
from nixtla import NixtlaClient
import numpy as np

MIN_HISTORY = 35  # TimeGPT needs ~35 points


def plot_cloud_cost_timeseries(
    df: pd.DataFrame, save_path: str = "Images/CloudCostTimeSeries.svg"
):
    """
    Plot the cloud cost time series and export data to CSV.

    Parameters
    ----------
    df : DataFrame with columns ['timestamp', 'value', 'promo_flag']
    save_path : Path to save the SVG plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot the time series
    ax.plot(df["timestamp"], df["value"], linewidth=2, alpha=0.85, label="Cloud Cost")

    # Highlight promotional days
    promo_dates = df[df["promo_flag"] == 1]["timestamp"]
    if len(promo_dates) > 0:
        promo_values = df[df["promo_flag"] == 1]["value"]
        ax.scatter(
            promo_dates,
            promo_values,
            color="red",
            s=100,
            marker="*",
            label="Promotional Event",
            zorder=5,
        )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cloud Cost (USD)", fontsize=12)
    ax.set_title("Cloud Cost Time Series", fontsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()

    # Export Chart 1: Cloud Cost Time Series (CloudCostTimeSeries.svg)
    chart_1_data = pd.DataFrame(
        {"ds": df["timestamp"], "value": df["value"], "promo_flag": df["promo_flag"]}
    )
    chart_1_data.to_csv(
        "../../../posts/images/anomalyinproduction/chart-1.csv", index=False
    )


@dataclass
class AnomalyResult:
    unique_id: str
    ds: pd.Timestamp
    actual: float
    predicted: float
    lower: float
    upper: float
    anomaly: bool


def simulate_next_day_anomaly(
    df: pd.DataFrame,
    api_key: str | None = None,
    freq: str = "D",
    level: int = 99,
    model: str = "timegpt-1",
    k: int = 5,  # how many last days to simulate
) -> list[AnomalyResult]:
    """
    Same as simulate_next_day_anomaly, but only simulate the last `k` days.
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    client = NixtlaClient(api_key=api_key)
    results: list[AnomalyResult] = []

    # Require at least MIN_HISTORY + k observations
    if len(df) < MIN_HISTORY + k:
        return results

    # Indices to evaluate: last k days (excluding the very last because we need next-day)
    start_idx = len(df) - k - 1
    for t in range(start_idx, len(df) - 1):
        hist = df.loc[:t, ["unique_id", "ds", "y"]]
        next_row = df.loc[t + 1]

        fcst = client.forecast(
            df=hist,
            h=1,
            freq=freq,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            level=[level],
            model=model,
        )

        row = fcst.iloc[-1]
        pred = float(row["TimeGPT"])
        lo = float(row.get(f"TimeGPT-lo-{level}", float("nan")))
        hi = float(row.get(f"TimeGPT-hi-{level}", float("nan")))
        actual = float(next_row["y"])
        is_anom = not (lo <= actual <= hi)

        results.append(
            AnomalyResult(
                unique_id=next_row["unique_id"],
                ds=pd.to_datetime(next_row["ds"]),
                actual=actual,
                predicted=pred,
                lower=lo,
                upper=hi,
                anomaly=is_anom,
            )
        )
    return results


from nixtla import NixtlaClient

MIN_HISTORY = 35  # TimeGPT needs ~35 points per series

import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Literal
from nixtla import NixtlaClient

MIN_HISTORY = 35  # TimeGPT needs ~35 points per series


def plot_last_k_days_next_h_forecasts(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    *,
    unique_id: str = "cloud_cost_usd",
    time_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
    level: int = 99,
    k: int = 5,
    h: int = 7,
    model: Literal["timegpt-1", "timegpt-1-long-horizon", "azureai"] = "timegpt-1",
    title: Optional[str] = None,
):
    """
    Plot K rolling, next-h forecasts from TimeGPT using the last K anchor days.

    Parameters
    ----------
    df : DataFrame with columns [unique_id, time_col, target_col]
    api_key : Nixtla API key (or set env var NIXTLA_API_KEY)
    unique_id : series id (assumes a single series)
    time_col : timestamp column name (e.g., 'ds')
    target_col : value column name (e.g., 'y')
    freq : pandas frequency string (e.g., 'D')
    level : confidence level for intervals (e.g., 99)
    k : number of anchor days from the end to simulate
    h : forecast horizon (e.g., 7 for one week)
    model : TimeGPT model string
    title : optional plot title
    """
    # 0) Basic checks & prep
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # If user didn't include unique_id col for a single series, add it.
    if "unique_id" not in df.columns:
        df["unique_id"] = unique_id

    # Filter to single series (extend to multi later if needed)
    g = df[df["unique_id"] == unique_id].copy()
    if g.empty:
        raise ValueError(f"No rows for unique_id='{unique_id}'")

    # Regularize at freq (important for models and plotting)
    full_idx = pd.date_range(g[time_col].min(), g[time_col].max(), freq=freq)
    g = (
        g.set_index(time_col)
        .reindex(full_idx)
        .rename_axis(time_col)
        .reset_index()
        .rename(columns={"index": time_col})
    )
    # Forward/backward fill unique_id and interpolate y
    g["unique_id"] = g["unique_id"].ffill().bfill()
    g[target_col] = g[target_col].interpolate(limit_direction="both")

    if len(g) < MIN_HISTORY + 1:
        raise ValueError(
            f"Series too short: need at least {MIN_HISTORY + 1} points, got {len(g)}."
        )

    # 1) Determine anchor indices: last k days (but respect MIN_HISTORY)
    end_idx = len(g) - 1
    start_anchor = max(MIN_HISTORY, end_idx - k)  # ensure enough history
    anchors = list(range(start_anchor, end_idx))  # predict t+1..t+h from each anchor

    client = NixtlaClient(api_key=api_key)

    # --- FIGURE: two rows -> (top) full, (bottom) zoom on right/prediction
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=False)

    # 2) Plot observed history (TOP)
    (line_obs_top,) = ax.plot(
        g[time_col].to_numpy(),
        g[target_col].to_numpy(),
        label="Observed",
        linewidth=2,
        alpha=0.85,
    )

    # Keep forecasts to build the zoom view later
    fcsts = []
    interval_patches = []
    mean_lines = []

    # 3) For each anchor, forecast next h and plot mean + interval
    n_anchors = len(anchors)
    for i, t in enumerate(anchors):
        hist = g.loc[:t, ["unique_id", time_col, target_col]].copy()
        fcst = client.forecast(
            df=hist,
            h=h,
            freq=freq,
            id_col="unique_id",
            time_col=time_col,
            target_col=target_col,
            level=[level],
            model=model,
        )
        fcsts.append(fcst)

        mean_col = "TimeGPT"
        lo_col = f"TimeGPT-lo-{level}"
        hi_col = f"TimeGPT-hi-{level}"

        alpha = 0.15 + 0.6 * (i + 1) / n_anchors

        patch = ax.fill_between(
            fcst[time_col].to_numpy(),
            fcst[lo_col].to_numpy(),
            fcst[hi_col].to_numpy(),
            alpha=alpha * 0.4,
            label=None if i < n_anchors - 1 else f"Forecast interval (±{level}%)",
        )
        interval_patches.append(patch)

        (line_mean,) = ax.plot(
            fcst[time_col].to_numpy(),
            fcst[mean_col].to_numpy(),
            linewidth=2,
            alpha=alpha,
            label=None if i < n_anchors - 1 else "Forecast mean",
        )
        mean_lines.append(line_mean)

    # Axis titles (TOP)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cloud Cost")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)

    # Build a compact legend
    # Only label once for intervals/means; also include "Observed" and a generic "Anchor"
    handles = [line_obs_top]
    if mean_lines:
        handles.append(mean_lines[-1])
    if interval_patches:
        handles.append(interval_patches[-1])
    # Create a dummy line for anchors legend entry
    if anchors:
        (anchor_dummy,) = ax.plot(
            [], [], linestyle="--", color="gray", alpha=0.6, label="Anchor day"
        )
        handles.append(anchor_dummy)

    ax.legend(handles=handles, loc="best")

    # --------- BOTTOM (ZOOM) SUBPLOT ---------
    # Choose zoom window: from a bit before the last anchor to the end of forecasts
    if fcsts:
        last_fcst_end = max(fcst[time_col].max() for fcst in fcsts)
    else:
        last_fcst_end = g[time_col].iloc[-1]

    # Start zoom some days before the last anchor (e.g., 2*h days)
    pad_days = max(7, h * 2)
    zoom_start = g[time_col].iloc[max(0, end_idx - pad_days)]
    zoom_end = last_fcst_end

    # Observed (tail) in zoom
    mask_zoom_obs = (g[time_col] >= zoom_start) & (g[time_col] <= zoom_end)
    ax2.plot(
        g.loc[mask_zoom_obs, time_col].to_numpy(),
        g.loc[mask_zoom_obs, target_col].to_numpy(),
        label="Observed (zoom)",
        linewidth=2,
        alpha=0.9,
    )

    # Forecasts in zoom (same styling approach)
    for i, fcst in enumerate(fcsts):
        mean_col = "TimeGPT"
        lo_col = f"TimeGPT-lo-{level}"
        hi_col = f"TimeGPT-hi-{level}"
        alpha = 0.15 + 0.6 * (i + 1) / max(1, n_anchors)

        ax2.fill_between(
            fcst[time_col].to_numpy(),
            fcst[lo_col].to_numpy(),
            fcst[hi_col].to_numpy(),
            alpha=alpha * 0.4,
        )
        ax2.plot(
            fcst[time_col].to_numpy(),
            fcst[mean_col].to_numpy(),
            linewidth=2,
            alpha=alpha,
        )

    # Axis titles (BOTTOM)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cloud Cost (zoom)")
    ax2.set_xlim([zoom_start, zoom_end])
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()

    # Export Chart 2: Forecasting on Cloud (ForecastingOnCloud.svg)
    # Combine observed data with all forecasts
    chart_2_rows = []

    # Add observed history
    for idx, row in g.iterrows():
        chart_2_rows.append(
            {
                "ds": row[time_col],
                "actual": row[target_col],
                "forecast_mean": np.nan,
                "forecast_lo": np.nan,
                "forecast_hi": np.nan,
                "anchor_index": np.nan,
                "type": "observed",
            }
        )

    # Add forecasts for each anchor
    for i, (t, fcst) in enumerate(zip(anchors, fcsts)):
        mean_col = "TimeGPT"
        lo_col = f"TimeGPT-lo-{level}"
        hi_col = f"TimeGPT-hi-{level}"

        for idx, row in fcst.iterrows():
            chart_2_rows.append(
                {
                    "ds": row[time_col],
                    "actual": np.nan,
                    "forecast_mean": row[mean_col],
                    "forecast_lo": row[lo_col],
                    "forecast_hi": row[hi_col],
                    "anchor_index": i,
                    "type": "forecast",
                }
            )

    chart_2_data = pd.DataFrame(chart_2_rows)
    chart_2_data.to_csv(
        "../../../posts/images/anomalyinproduction/chart-2.csv", index=False
    )


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nixtla import NixtlaClient


def simulate_and_plot_last_k_next_day_anomalies(
    df: pd.DataFrame,
    api_key: str | None = None,
    *,
    freq: str = "D",
    level: int = 99,
    model: str = "timegpt-1",
    k: int = 14,  # simulate last k days
    title: str | None = None,
    min_abs_delta: float = 0.0,  # e.g., 5.0 means require >= $5 difference
    min_rel_delta: float = 0.01,  # e.g., 0.01 means require >= 1% difference
) -> list[AnomalyResult]:
    """
    Simulate next-day anomaly detection only for the last `k` days and plot:
      - observed tail
      - predicted mean for next-day points
      - prediction interval (±level %)
      - anomalies highlighted

    An observation is flagged as anomaly only if:
      1) it's outside the [lo, hi] interval, AND
      2) |actual - pred| >= max(min_abs_delta, min_rel_delta * max(1, |pred|))

    Returns a list[AnomalyResult].
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    if len(df) < MIN_HISTORY + k:
        raise ValueError(
            f"Not enough history: need at least {MIN_HISTORY + k}, got {len(df)}."
        )

    client = NixtlaClient(api_key=api_key)
    results: list[AnomalyResult] = []

    # Indices to evaluate: last k days (exclude last because we need next-day)
    start_idx = max(len(df) - k - 1, MIN_HISTORY)
    end_idx = len(df) - 1

    # For plotting
    ds_pred, pred_mean, pred_lo, pred_hi, actuals, is_anom = [], [], [], [], [], []

    for t in range(start_idx, end_idx):
        hist = df.loc[:t, ["unique_id", "ds", "y"]]
        next_row = df.loc[t + 1]

        fcst = client.forecast(
            df=hist,
            h=1,
            freq=freq,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            level=[level],
            model=model,
        )

        row = fcst.iloc[-1]
        pred = float(row["TimeGPT"])
        lo = float(row.get(f"TimeGPT-lo-{level}", np.nan))
        hi = float(row.get(f"TimeGPT-hi-{level}", np.nan))
        actual = float(next_row["y"])

        abs_err = abs(actual - pred)
        rel_err = abs_err / max(1.0, abs(pred))
        big_enough = abs_err >= max(min_abs_delta, min_rel_delta * max(1.0, abs(pred)))
        outside = (
            not (lo <= actual <= hi) if np.isfinite(lo) and np.isfinite(hi) else False
        )

        anom = bool(outside and big_enough)

        results.append(
            AnomalyResult(
                unique_id=next_row["unique_id"],
                ds=pd.to_datetime(next_row["ds"]),
                actual=actual,
                predicted=pred,
                lower=lo,
                upper=hi,
                anomaly=anom,
            )
        )

        ds_pred.append(pd.to_datetime(next_row["ds"]))
        pred_mean.append(pred)
        pred_lo.append(lo)
        pred_hi.append(hi)
        actuals.append(actual)
        is_anom.append(anom)

    # ------------ Plot ------------
    ds_pred = np.array(ds_pred)
    pred_mean = np.array(pred_mean, dtype=float)
    pred_lo = np.array(pred_lo, dtype=float)
    pred_hi = np.array(pred_hi, dtype=float)
    actuals = np.array(actuals, dtype=float)
    is_anom = np.array(is_anom, dtype=bool)

    tail_start = max(0, len(df) - 2 * k)
    obs_tail = df.iloc[tail_start:].copy()

    fig, (ax_main, ax_zoom) = plt.subplots(2, 1, figsize=(12, 9))

    # Main: observed tail
    ax_main.plot(
        obs_tail["ds"].to_numpy(),
        obs_tail["y"].to_numpy(),
        label="Observed",
        linewidth=2,
        alpha=0.85,
    )

    if len(ds_pred) > 0:
        ax_main.fill_between(
            ds_pred, pred_lo, pred_hi, alpha=0.25, label=f"±{level}% interval"
        )
        ax_main.plot(ds_pred, pred_mean, linewidth=2, alpha=0.9, label="Predicted mean")
        ax_main.scatter(ds_pred, actuals, s=40, label="Actual (next-day)", zorder=3)
        if is_anom.any():
            ax_main.scatter(
                ds_pred[is_anom],
                actuals[is_anom],
                s=60,
                marker="x",
                color="red",
                label="Anomaly",
                zorder=4,
            )

    ax_main.set_title(
        title or f"Next-day anomalies (last {len(ds_pred)} days, level={level}%)"
    )
    ax_main.set_xlabel("Date")
    ax_main.set_ylabel("y")
    ax_main.grid(True, alpha=0.25)
    ax_main.legend(loc="best")

    # Zoom on predictions
    if len(ds_pred) > 0:
        ax_zoom.plot(
            obs_tail["ds"].to_numpy(),
            obs_tail["y"].to_numpy(),
            label="Observed (tail)",
            linewidth=2,
            alpha=0.4,
        )
        ax_zoom.fill_between(ds_pred, pred_lo, pred_hi, alpha=0.25)
        ax_zoom.plot(ds_pred, pred_mean, linewidth=2, alpha=0.9, label="Predicted mean")
        ax_zoom.scatter(ds_pred, actuals, s=40, label="Actual (next-day)", zorder=3)
        if is_anom.any():
            ax_zoom.scatter(
                ds_pred[is_anom],
                actuals[is_anom],
                s=60,
                marker="x",
                color="red",
                label="Anomaly",
                zorder=4,
            )

        pad = max(1, len(ds_pred) // 8)
        x0 = ds_pred[max(0, len(ds_pred) - (k + pad))]
        x1 = ds_pred[-1]
        ax_zoom.set_xlim([x0, x1])

    ax_zoom.set_xlabel("Date")
    ax_zoom.set_ylabel("y (zoom)")
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.legend(loc="best")

    plt.tight_layout()
    plt.show()

    # Export Chart 3: Monitoring Algorithm (MonitoringAlgorithm.svg)
    chart_3_rows = []

    # Add observed tail data
    for idx, row in obs_tail.iterrows():
        chart_3_rows.append(
            {
                "ds": row["ds"],
                "actual": row["y"],
                "forecast_mean": np.nan,
                "forecast_lo": np.nan,
                "forecast_hi": np.nan,
                "is_anomaly": False,
                "type": "observed",
            }
        )

    # Add prediction data
    for i in range(len(ds_pred)):
        chart_3_rows.append(
            {
                "ds": ds_pred[i],
                "actual": actuals[i],
                "forecast_mean": pred_mean[i],
                "forecast_lo": pred_lo[i],
                "forecast_hi": pred_hi[i],
                "is_anomaly": is_anom[i],
                "type": "prediction",
            }
        )

    chart_3_data = pd.DataFrame(chart_3_rows)
    chart_3_data.to_csv(
        "../../../posts/images/anomalyinproduction/chart-3.csv", index=False
    )

    return results


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("NIXTLA_API_KEY")

    if not api_key:
        raise ValueError(
            "Please set NIXTLA_API_KEY in your .env file or environment variables"
        )

    print("Generating synthetic cloud cost data...")

    # Helper functions from the blog post
    def _rng(seed):
        return np.random.default_rng(None if seed is None else seed)

    def _as_df_long(metric_id, ts, values, **extras):
        df = pd.DataFrame({"metric_id": metric_id, "timestamp": ts, "value": values})
        for k, v in extras.items():
            df[k] = v
        return df

    def generate_traffic_pattern(
        idx, weekday_weekend_ratio, trend_growth, random_walk_std, base_traffic, rng
    ):
        """Generate base traffic pattern with weekday factor, trend, and random walk."""
        n = len(idx)
        weekday = idx.weekday
        weekday_factor = np.where(weekday < 5, 1.0, weekday_weekend_ratio)
        trend = np.linspace(1.0, 1.0 + trend_growth, n)
        random_walk = np.cumsum(rng.normal(0, random_walk_std, n))
        traffic = (base_traffic * weekday_factor * trend * (1 + random_walk)).clip(
            min=base_traffic * 0.4
        )
        return traffic

    def apply_promotions(traffic, idx, promo_days, promo_lift):
        """Apply promotional lift to traffic and return promo flags."""
        if promo_days is None:
            promo_days = []
        promo_days = (
            pd.to_datetime(list(promo_days)) if promo_days else pd.to_datetime([])
        )
        promo_flag = np.isin(idx, promo_days).astype(int)
        traffic = traffic * (1 + promo_lift * promo_flag)
        return traffic, promo_flag

    def calculate_cost_from_traffic(
        traffic, baseline_infra_usd, cost_per_request, noise_usd, n, rng
    ):
        """Calculate final cost from traffic with baseline infrastructure cost and noise."""
        noise = rng.normal(0, noise_usd, n)
        cost = baseline_infra_usd + traffic * cost_per_request + noise
        return cost

    def make_cloud_cost_daily(
        start,
        end,
        baseline_infra_usd,
        cost_per_request,
        base_traffic,
        weekday_weekend_ratio=0.92,
        trend_growth=0.55,
        noise_usd=2.0,
        random_walk_std=0.002,
        promo_days=None,
        promo_lift=0.25,
        seed=42,
    ):
        """Generate synthetic cloud cost dataset."""
        rng = _rng(seed)
        idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
        n = len(idx)

        traffic = generate_traffic_pattern(
            idx, weekday_weekend_ratio, trend_growth, random_walk_std, base_traffic, rng
        )
        traffic, promo_flag = apply_promotions(traffic, idx, promo_days, promo_lift)
        cost = calculate_cost_from_traffic(
            traffic, baseline_infra_usd, cost_per_request, noise_usd, n, rng
        )

        return _as_df_long(
            "cloud_cost_usd",
            idx,
            np.round(cost, 2),
            traffic=traffic.astype(int),
            promo_flag=promo_flag,
        )

    # Generate cloud cost dataset
    cloud_cost_df = make_cloud_cost_daily(
        start="2025-01-01",
        end="2025-08-31",
        baseline_infra_usd=2000.0,
        cost_per_request=8e-4,
        base_traffic=1_000_000,
        promo_days=("2025-03-15", "2025-05-10", "2025-07-04"),
    )

    print(f"Generated {len(cloud_cost_df)} rows of cloud cost data")
    print("\nFirst few rows:")
    print(cloud_cost_df.head())

    # Chart 1: Plot cloud cost time series
    print("\n" + "=" * 60)
    print("Generating Chart 1: Cloud Cost Time Series...")
    print("=" * 60)
    plot_cloud_cost_timeseries(cloud_cost_df)
    print("✓ Chart 1 displayed")
    print("✓ Data exported: posts/images/anomalyinproduction/chart-1.csv")

    # Prepare data for TimeGPT (rename columns)
    cloud_cost_df_renamed = cloud_cost_df.rename(
        columns={"metric_id": "unique_id", "timestamp": "ds", "value": "y"}
    )

    # Chart 2: Forecasting on Cloud
    print("\n" + "=" * 60)
    print("Generating Chart 2: Forecasting on Cloud...")
    print("=" * 60)
    plot_last_k_days_next_h_forecasts(
        df=cloud_cost_df_renamed,
        api_key=api_key,
        freq="D",
        level=99,
        k=5,
        h=7,
        model="timegpt-1",
        title="Cloud Cost, 1-week forecasts",
    )
    print("✓ Chart 2 displayed")
    print("✓ Data exported: posts/images/anomalyinproduction/chart-2.csv")

    # Chart 3: Monitoring Algorithm
    print("\n" + "=" * 60)
    print("Generating Chart 3: Monitoring Algorithm...")
    print("=" * 60)
    results = simulate_and_plot_last_k_next_day_anomalies(
        df=cloud_cost_df_renamed[:-40],
        api_key=api_key,
        k=30,
        level=99,
        model="timegpt-1",
        title="Cloud Cost — next-day anomalies (last 30 days)",
    )
    print("✓ Chart 3 displayed")
    print("✓ Data exported: posts/images/anomalyinproduction/chart-3.csv")

    print("\n" + "=" * 60)
    print("All charts and CSV files generated successfully!")
    print("=" * 60)
    print(
        f"\nDetected {sum(1 for r in results if r.anomaly)} anomalies out of {len(results)} days monitored"
    )
    print("\nAnomaly details:")
    for r in results:
        if r.anomaly:
            print(
                f"  - {r.ds.date()}: Actual=${r.actual:.2f}, Predicted=${r.predicted:.2f}, "
                f"Interval=[${r.lower:.2f}, ${r.upper:.2f}]"
            )
