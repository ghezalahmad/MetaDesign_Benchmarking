# MetaDesign Benchmarking

A Flask web application for benchmarking **MetaDesign Active Learning (AL)** against random search on materials science datasets. Upload any tabular dataset, configure the optimization problem, and measure how efficiently different surrogate models and acquisition strategies find target materials — with built-in support for LLM-driven Bayesian Optimization.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Surrogate Portfolio](#surrogate-portfolio)
3. [Acquisition Strategies](#acquisition-strategies)
4. [LLM Modes](#llm-modes)
5. [Installation](#installation)
6. [Starting the App](#starting-the-app)
7. [Usage Guide](#usage-guide)
8. [Results & Exports](#results--exports)
9. [Data Format](#data-format)
10. [References](#references)

---

## What It Does

MetaDesign Active Learning benchmarks the following question:

> *How many experiments does an AI-guided search need to find the top-performing materials, compared to random selection?*

For each configuration the app runs *N* randomized active learning simulations on your dataset. In each simulation the surrogate model iteratively selects the next experiment to measure. The number of development cycles required to reach the target is recorded and compared against a random-process baseline using the Wilcoxon signed-rank test.

**Key capabilities:**

- 12 surrogate models — 8 traditional ML + 4 LLM/Hybrid (2026 SOTA)
- 5 acquisition strategies covering pure exploitation through full exploration
- Multi-objective optimization with weighted target combination
- A-priori information (zero-uncertainty targets, e.g. cost, CO₂)
- Per-target thresholds, directions (maximize / minimize) and weights
- Latin Hypercube initial sampling for space-filling coverage
- Live progress plots, final result plots, Pareto front, feature importance, parity plot
- Model comparison — runs all models in parallel and renders a ranked histogram
- CSV and PDF export

---

## Surrogate Portfolio

### Traditional ML

| Model | Uncertainty method | Notes |
|---|---|---|
| **Gaussian Process Regression** | Analytical GP posterior σ | Default. Robust for small datasets |
| **Lolo Random Forest** | Jackknife-in-bags | Requires Java SE ≥ 17 |
| **Gauss with PCA** | GP posterior | Dimensionality reduction before GP |
| **RF with PCA** | Lolo jackknife | Dimensionality reduction before RF |
| **Tuned Gauss** | GP posterior | Sequential forward feature selection + grid search |
| **Tuned RF** | Lolo jackknife | Forward feature selection + grid search |
| **Decision Trees** | Jackknife | Fast; high variance |
| **Random Forest (scikit)** | Jackknife | scikit-learn RF, 10 estimators |

> Tuned variants are computationally expensive. With 25 runs and many iterations, expect runtimes of 30 minutes to a few hours depending on dataset size.

### LLM / Hybrid (2026 SOTA)

| Model | Paradigm | LLM calls per iteration |
|---|---|---|
| **LLM-Only (Claude)** | Anthropic Claude scores every candidate via in-context learning | `ceil(N_candidates / batch)` |
| **LLM-Only (GPT-4)** | OpenAI GPT-4 scores every candidate via in-context learning | `ceil(N_candidates / batch)` |
| **Hybrid: GP + LLM** | GP scores all candidates; LLM rescores top-K by GP-UCB; outputs blended | 1 |
| **Hybrid: RF + LLM** | RF (jackknife) + LLM, same top-K blending scheme | 1 |

See [LLM Modes](#llm-modes) for configuration details.

---

## Acquisition Strategies

| Strategy | Formula | Character |
|---|---|---|
| **MEI (exploit)** | `argmax(μ + μ_fixed)` | Pure exploitation — follows the predicted best |
| **MLI (explore & exploit)** | `argmax(μ + μ_fixed + σ·σ_factor)` | Balances mean and uncertainty |
| **UCB (Upper Confidence Bound)** | `argmax(μ + μ_fixed + κ·σ)` | Classic BO exploration bound |
| **EI (Expected Improvement)** | `argmax(ΔΦ(z) + σφ(z))` | Improvement over current best |
| **Thompson Sampling** | `argmax(N(μ, \|σ\|))` | Stochastic; naturally explores |

The **σ / κ factor** slider scales the uncertainty term for MLI, UCB, and Thompson Sampling. A higher value favours exploration; a lower value favours exploitation.

---

## LLM Modes

LLM-based surrogates use **in-context learning**: the model receives a table of all observed (features → score) pairs as few-shot examples and returns predicted scores with uncertainty for unseen candidates.

### LLM-Only
The LLM is the sole surrogate. No ML model is used. Every candidate is scored by the LLM in batches. This mode is slower and incurs API costs, but requires no assumptions about the functional form of the property landscape.

### Hybrid
A traditional ML surrogate (GP or RF) scores **all** candidates first. The LLM then re-evaluates only the **top-K candidates** selected by GP-UCB — limiting each iteration to a single LLM call. Predictions are re-scaled to the ML numerical range and blended using the **LLM Blend Weight** slider:

```
mean_hybrid = (1 - α) · mean_ML  +  α · mean_LLM
std_hybrid  = (1 - α) · std_ML   +  α · std_LLM
```

`α = 0` → pure ML fallback. `α = 1` → LLM dominates. `α = 0.5` → equal weight.

### Requirements

| Provider | Install | Default model |
|---|---|---|
| Anthropic | `pip install anthropic` | `claude-sonnet-4-6` |
| OpenAI | `pip install openai` | `gpt-4o` |

API keys are sent directly to your chosen provider at request time and are never stored server-side.

Use the **Validate Connection** button in the LLM Configuration panel to verify your key before starting a run.

---

## Installation

**Python 3.10 or higher required.**

### 1. Clone the repository

```sh
git clone <repo-url>
cd MetaDesign_Benchmarking
```

### 2. Create and activate a virtual environment

```sh
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install core dependencies

```sh
pip install -r flask_requirements.txt
```

### 4. Optional — Lolo Random Forest (requires Java SE ≥ 17)

Download Java SE from <https://www.oracle.com/java/technologies/downloads/#java17>, then:

```sh
pip install lolopy
```

### 5. Optional — LLM surrogate modes

Install at least one provider SDK to enable LLM-Only and Hybrid models:

```sh
pip install anthropic   # for LLM-Only (Claude) and Hybrid: GP/RF + LLM
pip install openai      # for LLM-Only (GPT-4)
```

---

## Starting the App

```sh
python app.py
```

The server starts on **http://localhost:5050**. Open that URL in any modern browser.

To change the port, edit the last line of `app.py`:

```python
app.run(debug=True, port=5050, threaded=True)
```

Set the `SECRET_KEY` environment variable in production:

```sh
SECRET_KEY=your-secret-here python app.py
```

---

## Usage Guide

The interface is divided into four tabs.

### 1. Upload

Import your materials dataset as **CSV or Excel** (`.csv`, `.xlsx`, `.xls`).

Configuration options:

| Option | Purpose |
|---|---|
| Separator | Column delimiter: comma, semicolon, or space |
| Decimal | Decimal point character: `.` or `,` |
| Eraser | Strip tabs, quotes, or `%` signs from raw text |
| Skip rows | Skip header rows before the column-name row |

Click **Preview** to inspect the first 10 rows before committing. Click **Upload** to load the data into the session.

> All columns must be numeric. Non-numeric columns are coerced; rows with NaN in any selected column are dropped silently.

---

### 2. Data Info

Inspect the uploaded dataset in three views:

- **Preview** — first 10 rows
- **Stats** — descriptive statistics (mean, std, min, max, quartiles)
- **Info** — column dtypes, null counts, memory usage

---

### 3. Design Space Explorer

Visualize the dataset before configuring the optimization:

| Chart | Use |
|---|---|
| **Scatter** | Bivariate relationship with optional hue and size encoding |
| **Scatter Matrix** | All pairwise distributions for a selected subset of columns |
| **Correlation Heatmap** | Pearson correlation matrix with annotated values |

Use this tab for feature selection (check for collinearities) and to identify trade-offs between target properties.

---

### 4. Benchmarking

The core active learning benchmarking workflow, organized into four sections.

#### Configure Optimization

Select columns for three roles:

| Role | Description |
|---|---|
| **Materials Data (Input Features)** | Composition or process variables the model uses for prediction |
| **Target Properties** | Measured outputs to optimize (can select multiple for multi-objective) |
| **A-priori Information** | Known-without-experiment properties (e.g. cost, CO₂ footprint) included in the objective but not predicted |

Per-target controls appear for each selected column:

- **Direction** — maximize or minimize
- **Weight** — relative importance (default 1.0)
- **Threshold** — optional hard constraint (absolute value)

**Target Quantile** — defines what "success" means. The top *Q*% of rows (by weighted combined score, after applying thresholds) become the target set. Lower values mean a larger, easier target; higher values (e.g. 95%) mean a small, challenging target.

Click **Show Target Data** to preview which rows would be considered targets under the current settings.

---

#### MetaDesign Active Learning

Configure the benchmark parameters:

| Parameter | Default | Description |
|---|---|---|
| # of AL Runs | 25 | Number of independent randomized simulations |
| Initial Sample Size | 4 | Starting training set (drawn by Latin Hypercube Sampling) |
| Batch Size | 1 | Candidates selected per development cycle |
| Random Seed | — | Fix for reproducibility; leave blank for random |
| Model | Gaussian Process | Surrogate model (see [Surrogate Portfolio](#surrogate-portfolio)) |
| Strategy | MEI (exploit) | Acquisition function (see [Acquisition Strategies](#acquisition-strategies)) |
| σ / κ factor | 2.0 | Uncertainty weight for MLI, UCB, Thompson Sampling |

Click **Run Benchmarking** to start. A live progress bar and streaming log appear immediately. After the first complete simulation, **live plots** update after each run:

- **Input-space progress** — minimum distance from sampled points to any target row
- **Output-space progress** — maximum combined property score achieved so far
- **Performance histogram** — distribution of required experiments: MetaDesign AL vs Random Process

Click **Compare All Models** to run all ML models in parallel and produce a ranked comparison histogram. LLM models are included only if an API key is configured.

---

#### LLM Configuration

Appears automatically when a LLM-Only or Hybrid model is selected.

| Field | Description |
|---|---|
| Provider | Anthropic (Claude) or OpenAI (GPT-4) |
| API Key | Your provider key — sent only to the provider, never stored |
| Model | Defaults to `claude-sonnet-4-6` or `gpt-4o`; override as needed |
| LLM Blend Weight | Hybrid only — fraction of LLM vs ML contribution (0–1) |
| Max context rows | Maximum observed experiments included in each prompt |
| Max candidates per call | Batch size for LLM queries (Hybrid mode: always 1 batch/iter) |

Click **Validate Connection** to confirm the key and model are reachable before committing to a full run.

---

#### Results

After a run completes, the following plots are generated:

| Plot | What it shows |
|---|---|
| **Performance histogram** | Experiments required: AL vs Random, across all simulations |
| **Simple regret curve** | How quickly the best-found score approaches the global optimum |
| **Target property progress** | Best sampled value per development cycle, per target |
| **A-priori information progress** | Same, for fixed-target columns |
| **Pareto front** | Non-dominated solutions in 2-objective space (requires ≥ 2 targets) |
| **Feature importance** | Absolute Pearson correlation of each feature with the combined objective |
| **Surrogate parity plot** | Predicted vs actual scores on held-out candidates |

Summary statistics appear as cards:

- **SL cycles (mean / std)** — average and spread of required experiments
- **Random cycles (mean)** — random baseline
- **R²** — surrogate model quality on held-out data
- **MAE (norm)** — mean absolute error in normalized target space
- **p (Wilcoxon)** — one-sided p-value testing AL < Random (p < 0.05 = significant)

---

## Results & Exports

Results from each benchmarking run accumulate in the **Performance Log** table within the session.

| Export | How | Contents |
|---|---|---|
| **CSV** | Download CSV Log button | All result rows — algorithm, strategy, metrics, parameters |
| **PDF** | Export PDF Report button | Summary page + all result plots from the last completed run |

---

## Data Format

The app accepts any tabular dataset where:

- **Rows** represent individual material candidates or experiments
- **Columns** represent composition variables, process parameters, or measured properties
- All values are **numeric** (strings are coerced; categorical columns are not supported)
- The dataset is **complete** — every row should have measured values for the chosen targets

Example column layout for a cementitious composites dataset:

| water | cement | fly_ash | ggbs | sp | CompressiveStrength_MPa | EmbodiedCO2 |
|---|---|---|---|---|---|---|
| 180 | 350 | 100 | 50 | 2.5 | 48.2 | 312 |
| … | … | … | … | … | … | … |

A sample dataset `BenchmarkingExampleData.csv` is included in the repository.

---

## Project Structure

```
MetaDesign_Benchmarking/
├── app.py                  # Flask application — routes and SSE streaming
├── sl_engine.py            # SequentialLearner engine — all active learning logic
├── llm_surrogate.py        # LLM surrogate — LLMConfig, LLMSurrogate
├── flask_requirements.txt  # Core Python dependencies
├── requirements.txt        # Legacy Jupyter/Voilà dependencies (not needed for Flask app)
├── BenchmarkingExampleData.csv
├── templates/
│   └── index.html          # Single-page Bootstrap UI
└── static/
    ├── css/style.css
    └── js/app.js           # Frontend logic and SSE consumers
```

---

## References

- Völker et al. 2022, "ACCELERATING THE SEARCH FOR ALKALI ACTIVATED CEMENTS WITH SEQUENTIAL LEARNING", <http://dx.doi.org/10.13140/RG.2.2.33502.92480/1>
- Völker et al. 2021, "Sequential learning to accelerate discovery of alkali-activated binders", <http://dx.doi.org/10.13140/RG.2.2.18388.94087/1>
- Liu et al. 2023, "LLAMBO: Large Language Models for Bayesian Optimization", in-context learning as surrogate
- Citrine Informatics, Lolo Random Forest with uncertainty: <https://github.com/CitrineInformatics/lolo>
