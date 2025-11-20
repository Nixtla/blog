import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_metric_bar_multi(dfs, metric='mae', colors=None):

    if colors is None:
        colors = [
            '#5D4E9D',  # soft indigo
            '#597DBF',  # medium blue
            '#5AAFA0',  # teal-green
            '#9ABD70',  # olive-green
            '#E1B570'   # warm sand/gold
        ]
    all_metrics = []
    
    for i, df in enumerate(dfs):
        # Flatten each df
        if 'metric' not in df.columns:
            df = df.copy().reset_index()
            if df.columns[0] != 'metric':
                df = df.rename(columns={df.columns[0]: 'metric'})
        
        m = df.melt(id_vars='metric', var_name='model', value_name='value')
        m = m[m['metric'] == metric].copy()
        m['value'] = pd.to_numeric(m['value'], errors='coerce')
        m['group'] = f'Group {i+1}'
        m['color'] = colors[i % len(colors)]
        all_metrics.append(m[['model', 'value', 'color']])
    
    # Concatenate all models
    combined = pd.concat(all_metrics, ignore_index=True)
    combined = combined.sort_values('value', ascending=False)

    # Plot
    fig = plt.figure(figsize=(8,5))
    plt.bar(combined['model'], combined['value'],
            color=combined['color'])
    plt.title(f'{metric.upper()} Comparison Across Model Groups')
    plt.ylabel(metric.upper())
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=35)

    # Add labels
    for i, (_, row) in enumerate(combined.iterrows()):
        plt.text(i, row['value'] + 0.5, f"{row['value']:.2f}",
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig


def evaluate_cv(df, metric):
    models = df.columns.drop(['unique_id', 'ds', 'y', 'cutoff']).tolist()
    evals = metric(df, models=models)
    evals['best_statsforecast_model'] = evals[models].idxmin(axis=1)
    return evals

def get_best_model_forecast(forecasts_df, evaluation_df):
    with_best = forecasts_df.merge(evaluation_df[['unique_id', 'best_statsforecast_model']])
    res = with_best[['unique_id', 'ds']].copy()
    for suffix in ('', '-lo-90', '-hi-90'):
        res[f'best_statsforecast_model{suffix}'] = with_best.apply(lambda row: row[row['best_statsforecast_model'] + suffix], axis=1)
    return res


def plot_nixtla_forecast(train_df, forecast_df, model_col='best_statsforecast_model',
                         level=90, max_insample_length=120, title='Automatic Model Selection with StatsForecast'):
    """
    Create a Nixtla-branded forecast visualization for featured images.

    Parameters:
    - train_df: Training data with columns ['unique_id', 'ds', 'y']
    - forecast_df: Forecast data with prediction and intervals
    - model_col: Name of the column containing forecasts
    - level: Prediction interval level (e.g., 90 for 90%)
    - max_insample_length: Maximum number of historical points to show
    - title: Plot title
    """
    # Nixtla brand colors
    NIXTLA_TEAL = '#72FCDB'
    NIXTLA_PINK = '#E583B6'
    NIXTLA_BLUE = '#72BEFA'
    BLACK_BG = '#000000'
    WHITE = '#FFFFFF'

    # Use dark background
    plt.style.use('dark_background')

    # Create figure with 16:9 aspect ratio for featured image
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor(BLACK_BG)
    ax.set_facecolor(BLACK_BG)

    # Get the unique_id (assuming single series)
    unique_id = train_df['unique_id'].iloc[0]

    # Filter data
    train_plot = train_df.tail(max_insample_length).copy()
    forecast_plot = forecast_df.copy()

    # Plot historical data
    ax.plot(train_plot['ds'], train_plot['y'],
            color=WHITE, linewidth=2, label='Historical', alpha=0.9)

    # Plot forecast
    ax.plot(forecast_plot['ds'], forecast_plot[model_col],
            color=NIXTLA_TEAL, linewidth=2.5, label='Forecast', marker='o', markersize=4)

    # Plot prediction interval
    lo_col = f'{model_col}-lo-{level}'
    hi_col = f'{model_col}-hi-{level}'

    if lo_col in forecast_plot.columns and hi_col in forecast_plot.columns:
        ax.fill_between(forecast_plot['ds'],
                        forecast_plot[lo_col],
                        forecast_plot[hi_col],
                        color=NIXTLA_TEAL, alpha=0.2, label=f'{level}% Prediction Interval')

    # Styling
    ax.set_title(title, fontsize=20, fontweight='bold', color=WHITE, pad=20)
    ax.set_xlabel('Date', fontsize=16, color=WHITE)
    ax.set_ylabel('Value', fontsize=16, color=WHITE)

    # Customize tick labels
    ax.tick_params(axis='both', labelsize=14, colors=WHITE)

    # Legend
    ax.legend(fontsize=14, frameon=False, labelcolor=WHITE, loc='upper left')

    # Remove grid and customize spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(NIXTLA_TEAL)
    ax.spines['left'].set_color(NIXTLA_TEAL)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    plt.tight_layout()
    return fig