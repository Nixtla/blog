# Nixtla Blog Code

This repository contains the code and examples for blog articles published on [nixtla.io](https://nixtla.io).

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python package management. To set up the environment:

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
uv sync
pre-commit install
```

## Project Structure

- `notebooks/`: Contains Jupyter notebooks for each blog post
- `README.md`: This file

Each blog post's code is organized in its own notebook, named according to the blog post title.

## Available Notebooks

- **Anomaly Detection** (`anomaly_detection.ipynb`) - Learn how to detect anomalies in time series data using TimeGPT
- **Baseline Forecasts** (`baseline_forecasts.ipynb`) - Explore baseline forecasting methods and their effectiveness  
- **Intermittent Forecasting** (`intermittent_forecasting.ipynb`) - Explore demand forecasting techniques for intermittent time series

## Running Notebooks Locally

The notebooks are standard Jupyter notebooks that can be run in any Jupyter environment.

To run the notebooks locally, you can use:

```bash
# Start Jupyter Lab
uv run jupyter lab

# Or start Jupyter Notebook  
uv run jupyter notebook
```

This will start a local server where you can interact with the notebooks in your browser.

## Contributing

We welcome contributions to improve the code examples and documentation. Please see [CONTRIBUTION.md](CONTRIBUTION.md) for detailed guidelines on:

- Style and structure of blog posts
- Development workflow