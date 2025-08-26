# Contribution Guidelines

## Table of Contents

- [Writing Checklist](#writing-checklist)
- [Write Article Draft](#write-article-draft)
- [Write Code](#write-code)
- [Pull Request Process](#pull-request-process)

## Writing Checklist

### Writing Style Checklist

- [ ] Use action verbs instead of passive voice
- [ ] Structure content for quick scanning with clear headings and bullet points
- [ ] Keep paragraphs short (2–4 sentences maximum)
- [ ] Frame the code with context: explain what it does before presenting it, and clarify key parts immediately after
- [ ] Avoid filler words (just, really, actually, basically, in order to)
- [ ] Avoid phrases that hide the subject (there is, there are)
- [ ] Avoid ambiguous pronouns (it, this, that)
- [ ] Avoid -ing verb forms when possible (using, having, going)
- [ ] Avoid culture-specific references or idioms (kill two birds with one stone, silver bullet)


### Audience Assumptions Checklist

- [ ] Write for data scientists who are familiar with basic time series concepts
- [ ] Explain Nixtla tools as if readers are new to them
- [ ] Include enough examples for quick understanding of concepts

### Content Checklist

- [ ] Begin with a real-world time series problem or use case
- [ ] Present a solution that addresses the problem, making it the central focus of your article
- [ ] Include clear explanations of time series concepts and terminology
- [ ] When mentioning install commands or configuration flags, keep them minimal and link out to official docs for details

## Writing Style Examples

### Action Verbs vs Passive Voice

❌ *The model was trained on the dataset.*

✅ *Train the model on the dataset.*

### Code Explanation

Start by defining the baseline models for comparison:

```python
models = [
    HistoricAverage(),
    Naive(),
    SeasonalNaive(season_length = 4),
    WindowAverage(window_size=4)
]
```

This list includes four common baseline models:

- `HistoricAverage()`: Mean Forecast
- `Naive()`: Naive Forecast
- `SeasonalNaive(season_length=4)`: Seasonal Naive Forecast for quarterly seasonality
- `WindowAverage(window_size=4)`: Averages the last 4 quarters to forecast future values

### Filler Words

❌ *You just need to install the library to get started.*

✅ *Install the library to get started.*

### Subject Clarity

❌ *There are several reasons to use Nixtla tools.*

✅ *Nixtla tools offer several advantages.*

### Pronoun Clarity

❌ *This improves accuracy.*

✅ *The seasonal adjustment improves accuracy.*

### Verb Forms

❌ *Using this model helps reduce error.*

✅ *This model helps reduce error.*

### Cultural References

❌ *There's no silver bullet for intermittent demand.*

✅ *No single model solves all intermittent demand problems.*

## Write Article Draft

1. Create your blog post in the [nixtla/web](https://github.com/nixtla/web) repository
2. Follow [these instructions](https://github.com/nixtla/web?tab=readme-ov-file#blog) to create a new blog post

## Write Code

### Environment Setup

#### Install uv

[uv](https://github.com/astral.sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

#### Install Dependencies

```bash
# Install dependencies from pyproject.toml
uv sync
```

#### Install Pre-commit Hooks

We use pre-commit to ensure code quality and consistency.

```bash
# Install pre-commit hooks
uv run pre-commit install
```

### Working with Jupyter Notebooks

#### Creating a New Notebook

Create a new notebook in the `notebooks` directory using Jupyter:

```bash
uv run jupyter lab
```

Then create a new notebook file in the `notebooks/` directory.

#### Notebook Creation Guidelines

- [ ] Use snake_case for notebook names (e.g., `anomaly_detection.ipynb`, `intermittent_forecasting.ipynb`)  
- [ ] Keep notebook names short but descriptive
- [ ] Create headings using markdown cells
- [ ] Structure notebooks with clear sections and explanations

#### Running Notebooks

To run the notebooks locally:

```bash
# Start Jupyter Lab
uv run jupyter lab

# Or start Jupyter Notebook  
uv run jupyter notebook
```

This will start a local server where you can interact with the notebooks in your browser.

### Pull Request Process

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request with a clear description of changes