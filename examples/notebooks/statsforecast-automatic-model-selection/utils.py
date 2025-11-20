import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_metric_bar_multi(dfs, metric="mae", colors=None):
    if colors is None:
        colors = [
            "#5D4E9D",  # soft indigo
            "#597DBF",  # medium blue
            "#5AAFA0",  # teal-green
            "#9ABD70",  # olive-green
            "#E1B570",  # warm sand/gold
        ]
    all_metrics = []

    for i, df in enumerate(dfs):
        # Flatten each df
        if "metric" not in df.columns:
            df = df.copy().reset_index()
            if df.columns[0] != "metric":
                df = df.rename(columns={df.columns[0]: "metric"})

        m = df.melt(id_vars="metric", var_name="model", value_name="value")
        m = m[m["metric"] == metric].copy()
        m["value"] = pd.to_numeric(m["value"], errors="coerce")
        m["group"] = f"Group {i + 1}"
        m["color"] = colors[i % len(colors)]
        all_metrics.append(m[["model", "value", "color"]])

    # Concatenate all models
    combined = pd.concat(all_metrics, ignore_index=True)
    combined = combined.sort_values("value", ascending=False)

    # Plot
    fig = plt.figure(figsize=(8, 5))
    plt.bar(combined["model"], combined["value"], color=combined["color"])
    plt.title(f"{metric.upper()} Comparison Across Model Groups")
    plt.ylabel(metric.upper())
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=35)

    # Add labels
    for i, (_, row) in enumerate(combined.iterrows()):
        plt.text(
            i,
            row["value"] + 0.5,
            f"{row['value']:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    return fig


def evaluate_cv(df, metric):
    models = df.columns.drop(["unique_id", "ds", "y", "cutoff"]).tolist()
    evals = metric(df, models=models)
    evals["best_statsforecast_model"] = evals[models].idxmin(axis=1)
    return evals


def get_best_model_forecast(forecasts_df, evaluation_df):
    with_best = forecasts_df.merge(
        evaluation_df[["unique_id", "best_statsforecast_model"]]
    )
    res = with_best[["unique_id", "ds"]].copy()
    for suffix in ("", "-lo-90", "-hi-90"):
        res[f"best_statsforecast_model{suffix}"] = with_best.apply(
            lambda row: row[row["best_statsforecast_model"] + suffix], axis=1
        )
    return res
