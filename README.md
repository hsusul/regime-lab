# RegimeLab

[![CI](https://github.com/hsusul/regime-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/hsusul/regime-lab/actions/workflows/ci.yml)

RegimeLab is a backend-first financial machine learning project for classifying broad market regimes from historical time-series data. It focuses on reproducible pipelines, clean APIs, testing, and documentation rather than direct stock-price prediction.

RegimeLab is educational and research software only. It is not financial advice, does not provide trading recommendations, does not place trades, and does not guarantee profitable predictions.

## Current MVP Status

- Data ingestion and local raw CSV caching.
- Leakage-aware feature engineering.
- Rule-based regime labeling.
- Baseline supervised model training.
- Model evaluation metrics.
- File-based experiment metadata.
- Optional SQLite experiment registry with active-model selection.
- Cached-artifact FastAPI endpoints.
- Docker packaging for serving the API.
- Synthetic end-to-end smoke coverage for the local pipeline.
- Optional Hidden Markov Model exploratory regime reports.
- Retrospective forward return and risk analysis reports.
- Optional walk-forward validation reports for temporal model stability.
- Lightweight model explanation reports for latest predictions.
- Model comparison reports for baseline classifier selection.

## Architecture

```text
Data ingestion -> feature engineering -> rule labeling -> training -> evaluation -> API serving
                                         \-> optional HMM analysis
                                         \-> forward return analysis
```

- Data ingestion downloads daily OHLCV data and caches normalized CSVs under `data/raw/`.
- Feature engineering computes return, volatility, moving-average, drawdown, and volume features under `data/processed/`.
- Rule labeling creates heuristic `rule_label` targets for four market regimes.
- Training fits baseline scikit-learn classifiers and saves joblib artifacts under `models/`.
- Evaluation writes metrics JSON and updates file-based experiment metadata under `reports/`.
- The optional SQLite registry mirrors experiment metadata in `reports/regimelab.db` and can mark one experiment active.
- FastAPI serves health, regime prediction, history, metrics, and experiment endpoints from cached local files.
- Optional HMM analysis fits latent states from return and volatility sequences and writes separate artifacts/reports.
- Forward return analysis measures retrospective outcomes after rule-based or predicted regimes.
- Walk-forward validation evaluates supervised model stability across multiple chronological folds.
- Explanation reports summarize latest predictions, key feature values, and global model importance when available.
- Model comparison reports train/evaluate selected baselines and rank them by macro F1.

## Results from Sample Run

Current cached real-data run:

- Tickers: `SPY`, `QQQ`, `AAPL`, `NVDA`
- Labeled feature date range: `2015-10-16` to `2026-05-14`
- Supervised model: `random_forest`
- Test accuracy: `0.9948`
- Test macro F1: `0.9918`
- Combined label distribution: `stable_growth` 32.6%, `volatile_recovery` 8.5%, `sideways_defensive` 41.8%, `stress_selloff` 17.1%
- Latest regimes on `2026-05-14`: `SPY`, `QQQ`, and `AAPL` were `stable_growth`; `NVDA` was `sideways_defensive`

These metrics measure agreement with heuristic labels, not real-world trading performance. Detailed sample results, walk-forward validation ranges, interpretation notes, and resume bullets are in [docs/RESULTS.md](docs/RESULTS.md).

## How to Interpret Results

- Regime labels are transparent heuristics, not objective market truth.
- High model accuracy means the model reproduces heuristic labels; it does not imply profitable prediction.
- Forward returns are retrospective analysis outputs, not trading signals.
- HMM state names are interpreted after training from state statistics.
- If an artifact compatibility warning appears, retrain the model in the current environment.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- FastAPI
- Pydantic
- pytest
- joblib
- Docker
- yfinance
- hmmlearn, optional for HMM analysis

## Setup

Clone the repo:

```bash
git clone <repo-url>
cd regime-lab
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies:

```bash
python -m pip install -e .
```

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Install optional HMM dependencies:

```bash
python -m pip install -e ".[hmm]"
```

## Local Usage Flow

Project defaults live in [configs/default.yaml](configs/default.yaml). CLI arguments keep priority, so passing `--tickers`, `--start-date`, `--model-type`, or path flags overrides config values.

1. Ingest historical OHLCV data:

```bash
python -m src.data --tickers SPY QQQ AAPL NVDA --start-date 2015-01-01 --end-date 2026-05-15
```

2. Build features:

```bash
python -m src.features --tickers SPY QQQ AAPL NVDA
```

3. Label regimes:

```bash
python -m src.labeling --tickers SPY QQQ AAPL NVDA
```

4. Train a model:

```bash
python -m src.train --model-type random_forest --tickers SPY QQQ AAPL NVDA
```

You can also supply config-backed defaults:

```bash
python -m src.data --config configs/default.yaml
python -m src.features --config configs/default.yaml
python -m src.labeling --config configs/default.yaml
python -m src.train --config configs/default.yaml
```

5. Evaluate the latest model:

```bash
python -m src.evaluate --experiment-id latest
```

6. Run a diagnostics pass:

```bash
python -m src.diagnostics --tickers SPY QQQ AAPL NVDA
```

7. Optionally inspect or activate experiments in SQLite:

```bash
python -m src.experiment_registry list
python -m src.experiment_registry show <experiment_id>
python -m src.experiment_registry activate <experiment_id>
```

The API prefers an active SQLite experiment with a valid artifact path. If none exists, it falls back to `reports/experiments.json`.

8. Optionally run HMM latent regime analysis:

```bash
python -m src.hmm --tickers SPY QQQ AAPL NVDA --n-states 4
```

9. Optionally run forward return analysis:

```bash
python -m src.forward_returns --tickers SPY QQQ AAPL NVDA --horizons 5 20 60
```

Use `--save-csv` to also write a flattened summary table under `reports/`.

10. Optionally run walk-forward validation:

```bash
python -m src.walk_forward --tickers SPY QQQ AAPL NVDA --model-type random_forest --start-year 2018 --test-window-years 1
```

Use `--save-csv` to also write a fold summary table under `reports/`.

11. Optionally run a lightweight explanation report:

```bash
python -m src.explain --tickers SPY QQQ AAPL NVDA
```

12. Optionally compare baseline models:

```bash
python -m src.compare_models --tickers SPY QQQ AAPL NVDA --models logistic_regression random_forest --save-csv
```

13. Run the API:

```bash
uvicorn app.main:app --reload
```

## Diagnostics

Summarize cached labels, predictions, latest regimes, metrics, and artifact compatibility:

```bash
python -m src.diagnostics --tickers SPY QQQ AAPL NVDA
```

Optional HMM analysis fits an unsupervised Gaussian Hidden Markov Model using `return_1d` and `volatility_20d`:

```bash
python -m src.hmm --tickers SPY QQQ AAPL NVDA --n-states 4
```

Optional forward return analysis measures what historically happened after each observed regime:

```bash
python -m src.forward_returns --tickers SPY QQQ AAPL NVDA --horizons 5 20 60
```

Optional walk-forward validation evaluates model stability across expanding chronological folds:

```bash
python -m src.walk_forward --tickers SPY QQQ AAPL NVDA --model-type random_forest --start-year 2018 --test-window-years 1
```

Optional explanation reports summarize latest predictions and global model importance:

```bash
python -m src.explain --tickers SPY QQQ AAPL NVDA
```

Optional model comparison trains/evaluates selected baselines and writes a comparison report:

```bash
python -m src.compare_models --tickers SPY QQQ AAPL NVDA --models logistic_regression random_forest --save-csv
```

These paths write local reports under `reports/` and stay separate from the supervised API. Walk-forward metrics, explanations, and model comparisons measure agreement with heuristic labels and model behavior, not trading profitability.

## API Examples

The API uses cached data, labeled feature files, metrics, experiment metadata, and model artifacts generated by the CLI commands. It does not download live market data inside request handlers.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/regime/SPY
curl http://localhost:8000/history/SPY
curl http://localhost:8000/metrics
curl http://localhost:8000/experiments
```

`GET /history/{ticker}` returns the latest 100 rows by default. Pass `limit` for a smaller or larger slice; requests above 1,000 rows are capped and include a warning.

## Docker

Build the image:

```bash
docker build -t regimelab .
```

Run the API:

```bash
docker run -p 8000:8000 regimelab
```

The image does not bake in local `data/`, `models/`, or `reports/` artifacts. To serve predictions from existing local artifacts, mount those directories:

```bash
docker run -p 8000:8000 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/reports:/app/reports" \
  regimelab
```

## Testing

```bash
pytest
python -m compileall app src
python -m ruff check .
```

`ruff` is part of the development dependencies. Install with `python -m pip install -e ".[dev]"` before running it.

## Project Structure

```text
regime-lab/
  app/
    main.py
    schemas.py
    routes/
      health.py
      regime.py
      metrics.py
      experiments.py
  src/
    config.py
    data.py
    features.py
    labeling.py
    train.py
    evaluate.py
    predict.py
    experiment_registry.py
    hmm.py
    forward_returns.py
    walk_forward.py
    explain.py
    compare_models.py
    experiments.py
  data/
    raw/
    processed/
  models/
  reports/
  notebooks/
  tests/
  docs/
  configs/
    default.yaml
  Dockerfile
  pyproject.toml
  PRODUCT_REQUIREMENTS.md
  README.md
```

## Regime Labels

- `stable_growth`: positive trend, support from moving averages, contained volatility, and limited drawdown.
- `volatile_recovery`: improving recent returns while drawdown or long-term trend damage remains.
- `sideways_defensive`: valid rows that do not meet stronger growth, recovery, or selloff rules.
- `stress_selloff`: negative 20-day return with meaningful drawdown, weak moving-average position, or elevated volatility.

These labels are transparent heuristics used as supervised training targets. They are not objective market truth.

## Leakage Prevention

- Rolling features use only current and prior observations.
- Forward returns are generated only inside `src.forward_returns` reports, never as model inputs.
- Walk-forward validation trains each fold only on dates before that fold's test window.
- Features are computed separately per ticker.
- Training uses a chronological split: earliest 80% of unique dates for training and latest 20% for testing.
- The train/test split is shared across tickers for combined-universe training.
- Scalers and estimators are fit only on the training split.

## Limitations

- Labels are rule-based heuristics, not ground truth.
- Results measure agreement with heuristic labels, not investment performance.
- HMM states are latent clusters interpreted after fitting; their names are analytical summaries, not known market truth.
- Forward return reports are retrospective summaries and can be affected by regime label quality, market period selection, and survivorship-style assumptions.
- Walk-forward validation tests temporal stability against heuristic labels, not profitability.
- Explanation reports are descriptive model diagnostics and do not include SHAP.
- Data comes from `yfinance`, which is useful for an educational project but not a production-grade market data source.
- The API reads local cached files and artifacts; it does not run live downloads or training inside request handlers.
- File-based experiment metadata is intentionally simple for the MVP.

## Future Roadmap

- HMM stability analysis and richer transition diagnostics.
- Richer forward risk analysis with drawdown paths and visualizations.
- Experiment tracking and model registry.
- More robust walk-forward validation variants.
- SHAP or other model interpretability.
- Streamlit dashboard.
