# RegimeLab Product Requirements

## 1. Project Overview

RegimeLab is a backend-first financial machine learning system for classifying broad market regimes from historical time-series data. Instead of trying to predict exact stock prices, the system learns and serves interpretable labels for market environments such as `stable_growth`, `volatile_recovery`, `sideways_defensive`, and `stress_selloff`.

The project is designed as a serious ML engineering and backend systems portfolio project. It should show that the builder can ingest real financial data, engineer time-series features, label and train models without temporal leakage, evaluate results honestly, expose predictions through a clean API, and document the system clearly.

Regime classification is more realistic than naive stock price prediction because exact price forecasting is noisy, unstable, and often misleading. A regime model asks a more practical question: "What kind of environment does this asset appear to be in?" This framing supports risk analysis, market research, and model interpretability without claiming to produce profitable trades.

Target users:

- Student builder: uses the project to practice ML pipelines, API design, testing, Docker, and reproducible engineering.
- Recruiters and interviewers: review the project as evidence of applied ML engineering, backend design, and disciplined project execution.
- Technical reviewers: inspect implementation quality, assumptions, data handling, leakage controls, metrics, and documentation.

## 2. Goals

- Build a reproducible financial ML pipeline from historical daily market data.
- Classify market regimes using transparent rule-based labels and baseline supervised classifiers.
- Expose predictions, history, metrics, and experiment metadata through a FastAPI backend.
- Keep training and evaluation CLI-first in the MVP, with API-triggered training optional for local demonstration only.
- Track model runs, metrics, artifacts, feature versions, and basic experiment metadata.
- Demonstrate clean software engineering practices: modular code, typed schemas, tests, documentation, Docker, and clear repository structure.
- Produce a polished GitHub project suitable for internship applications and technical interviews.
- Keep the MVP achievable while leaving room for advanced extensions such as HMM-based regime detection, forward return analysis, and richer experiment tracking.

## 3. Non-Goals

- RegimeLab is not a trading bot.
- RegimeLab is not financial advice.
- RegimeLab does not promise profitable predictions.
- RegimeLab does not perform real-time or high-frequency trading.
- RegimeLab does not place orders, connect to brokerages, or manage portfolios.
- RegimeLab does not attempt to predict exact future prices in the MVP.
- RegimeLab does not require a complex frontend for the MVP.
- RegimeLab does not optimize for production-scale distributed training in the initial version.

## 4. Core Concepts

- Asset/ticker: A market symbol such as `SPY`, `QQQ`, `AAPL`, `NVDA`, or optionally `BTC-USD`.
- Daily OHLCV data: Historical daily open, high, low, close, adjusted close, and volume data.
- Time-series feature window: A rolling lookback period used to compute features from past observations only.
- Market regime: A broad label describing the current market environment for an asset.
- Rule-based label: A regime label assigned by transparent deterministic rules using returns, volatility, moving averages, and drawdown.
- Supervised classifier: A model trained to predict rule-based regime labels from engineered features.
- Unsupervised regime model: A model, such as a Hidden Markov Model, that infers latent market states without explicit labels.
- Experiment run: One training/evaluation execution with a defined dataset, feature set, model type, parameters, metrics, and artifact path.
- Prediction response: The API output containing a predicted regime, confidence scores when available, key signals, data timestamp, model metadata, and warnings if needed.

## 5. MVP Scope

The MVP is CLI-first for data ingestion, feature generation, training, and evaluation. FastAPI is required for serving predictions and reading results, but command-line workflows should be the primary reproducible interface for building datasets, training models, and generating reports. `POST /train` is optional in the MVP and should be treated as a local demonstration endpoint, not the canonical training interface.

Required CLI commands:

```bash
python -m src.data --tickers SPY QQQ AAPL NVDA --start-date 2015-01-01 --end-date 2026-05-15
python -m src.features --tickers SPY QQQ AAPL NVDA
python -m src.train --model-type random_forest --tickers SPY QQQ AAPL NVDA
python -m src.evaluate --experiment-id 2026-05-18T120000_random_forest
uvicorn app.main:app --reload
```

CLI expectations:

- Commands should be documented in `README.md`.
- Commands should use sensible defaults but allow explicit ticker, date range, model type, and output path arguments.
- Commands should write predictable files under `data/`, `models/`, and `reports/`.
- Training and evaluation commands should be reproducible from cached input data.

### A. Data Ingestion

The MVP should provide a clean data loading layer that can use `yfinance` initially while keeping the implementation pluggable enough to support other data sources later.

Requirements:

- Support at least `SPY`, `QQQ`, `AAPL`, and `NVDA`.
- Treat `BTC-USD` as optional because crypto trades on a different calendar than equities and can complicate feature alignment.
- Download historical daily OHLCV data.
- Use adjusted prices where appropriate for return calculations.
- Save or cache raw data locally under `data/raw/`.
- Avoid repeated network downloads when cached data is fresh enough.
- Validate required columns before downstream processing.
- Handle missing data with explicit documented behavior.

Expected data fields:

- `date`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`
- `ticker`

### B. Feature Engineering

The MVP should compute deterministic, testable time-series features using only current and historical data.

Required features:

- 1-day return
- 5-day return
- 20-day return
- 20-day rolling volatility
- 50-day moving average distance
- 200-day moving average distance
- Drawdown from recent rolling high
- Volume change
- Momentum features

Implementation expectations:

- Feature functions should be pure or mostly pure where practical.
- Feature generation should drop or flag rows without enough lookback history.
- Features should avoid lookahead bias.
- Rolling features must use only current and prior rows.
- Forward returns must never be used as model inputs.
- Feature names should be stable and documented.
- Feature calculations should be covered by unit tests.

Example feature definitions:

- `return_1d`: percentage change from previous adjusted close.
- `return_5d`: percentage change over five trading days.
- `return_20d`: percentage change over twenty trading days.
- `volatility_20d`: rolling standard deviation of daily returns over twenty trading days.
- `ma_50_distance`: adjusted close divided by 50-day moving average minus 1.
- `ma_200_distance`: adjusted close divided by 200-day moving average minus 1.
- `drawdown_60d`: adjusted close divided by rolling 60-day high minus 1.
- `volume_change_20d`: current volume divided by 20-day average volume minus 1.

### C. Rule-Based Regime Labeling

The MVP should start with four transparent regimes:

- `stable_growth`
- `volatile_recovery`
- `sideways_defensive`
- `stress_selloff`

Labeling should be deterministic and easy to explain. Initial rules may be tuned, but the logic should be documented and tested.

Suggested rule logic:

- `stable_growth`
  - Positive 20-day return.
  - Price above or near 50-day and 200-day moving averages.
  - Moderate or low 20-day volatility.
  - Limited drawdown.

- `volatile_recovery`
  - Recent returns improving after a drawdown.
  - Price may still be below the 200-day moving average.
  - Volatility remains elevated.
  - Short-term momentum is positive.

- `sideways_defensive`
  - Low or mixed 20-day return.
  - Price near moving averages.
  - Moderate volatility.
  - No severe drawdown.

- `stress_selloff`
  - Negative 20-day return.
  - Price below key moving averages.
  - Elevated volatility and/or significant drawdown.

Implementation expectations:

- Rules should be centralized in `src/labeling.py`.
- Thresholds should be configurable constants.
- Labels should be reproducible.
- Tests should cover representative cases for all four regimes.
- Documentation should state that these labels are heuristic, not ground truth.

### D. Supervised Model Training

The MVP should train baseline classifiers to predict the rule-based regime labels from engineered features.

Required models:

- Logistic Regression baseline.
- Random Forest baseline.

Training requirements:

- Use a concrete time-based train/test split: sort by date, train on the earliest 80%, test on the latest 20%, and never shuffle rows.
- Save trained model artifact under `models/`.
- Save feature column order with the model artifact metadata.
- Save metrics to JSON under `reports/`.
- Save file-based experiment metadata to `reports/experiments.json`.
- Support training by ticker or by a combined ticker universe.
- Use deterministic random seeds where applicable.
- Include a training smoke test with a small dataset.

Saved model artifact contract:

The MVP artifact may be a single `joblib` file containing a dictionary, or a model file plus adjacent metadata JSON. In either case, loading code must be able to recover:

- `model`: fitted scikit-learn estimator or pipeline.
- `feature_columns`: ordered list of feature columns expected at prediction time.
- `label_names`: ordered list of supported class labels.
- `model_type`: model family such as `logistic_regression` or `random_forest`.
- `training_date_range`: start and end dates used for training.
- `ticker_universe`: tickers included in the training run.
- `feature_version`: stable feature-set version string.
- `created_at`: timestamp when the artifact was created.

Latest model resolution:

- Read `reports/experiments.json`.
- Filter to experiments with `status` equal to `completed`.
- Require a non-empty `artifact_path`.
- Require that `artifact_path` exists on disk.
- Select the most recent valid experiment by `created_at`.
- Return a clear no-model error if no valid completed experiment exists.

Important constraints:

- No temporal leakage.
- No future returns used as features for current labels in the MVP.
- Scalers, imputers, and model pipelines must be fit only on training data.

### E. Evaluation

The MVP should provide honest, readable evaluation output.

Required metrics:

- Accuracy.
- Macro F1.
- Confusion matrix.
- Per-class precision, recall, and F1.
- Feature importance for tree-based models.

Evaluation notes to include in reports:

- Class imbalance may make accuracy misleading.
- Macro F1 is important because minority regimes matter.
- Rule-based labels are heuristic labels, not objective truth.
- Time-series splits are required to reduce leakage.
- The MVP split policy is earliest 80% train and latest 20% test after sorting by date, with no shuffle.
- Results should not be interpreted as trading performance.

### F. FastAPI Backend

The MVP API should expose model health, predictions, history, metrics, and experiment metadata. Training and evaluation remain CLI-first. `POST /train` may be included for local demonstration, but it is optional and should call the same training code used by the CLI.

#### `GET /health`

Purpose:

- Confirm the API is running.
- Report basic service and model availability status.

Request parameters:

- None.

Response shape:

```json
{
  "status": "ok",
  "service": "regimelab-api",
  "model_loaded": true,
  "version": "0.1.0"
}
```

Expected errors:

- Should rarely fail unless the service is unavailable.

#### `POST /train`

Purpose:

- Optionally trigger a local demonstration training run for selected tickers, date range, feature set, and model type.
- This endpoint is not the primary MVP training workflow; CLI training is canonical.

Request parameters:

- `tickers`: list of ticker symbols.
- `start_date`: optional ISO date.
- `end_date`: optional ISO date.
- `model_type`: `logistic_regression` or `random_forest`.
- `force_refresh`: optional boolean for data reload.

Response shape:

```json
{
  "experiment_id": "2026-05-18T120000_random_forest",
  "status": "completed",
  "model_type": "random_forest",
  "tickers": ["SPY", "QQQ"],
  "metrics": {
    "accuracy": 0.78,
    "macro_f1": 0.71
  },
  "artifact_path": "models/random_forest_2026-05-18T120000.joblib"
}
```

Expected errors:

- `400` for invalid ticker, invalid date range, or unsupported model type.
- `422` for request schema validation errors.
- `500` for unexpected training failures.

#### `GET /regime/{ticker}`

Purpose:

- Return the latest predicted regime for a ticker.

Request parameters:

- Path: `ticker`.
- Query: optional `refresh_data`.

Response shape:

```json
{
  "ticker": "SPY",
  "as_of": "2026-05-15",
  "regime": "stable_growth",
  "confidence": 0.82,
  "probabilities": {
    "stable_growth": 0.82,
    "volatile_recovery": 0.08,
    "sideways_defensive": 0.07,
    "stress_selloff": 0.03
  },
  "key_signals": {
    "return_20d": 0.034,
    "volatility_20d": 0.011,
    "ma_50_distance": 0.018,
    "drawdown_60d": -0.012
  },
  "model_id": "random_forest_2026-05-18T120000",
  "warnings": []
}
```

Expected errors:

- `404` if no data is available for the ticker.
- `409` if no trained model is available.
- `422` for invalid ticker format.
- `500` for unexpected prediction failures.

#### `GET /history/{ticker}`

Purpose:

- Return recent historical labels, predicted regimes, and selected signals for a ticker.
- When both rule-based labels and model predictions are available, the response must distinguish `rule_label` from `predicted_regime`.

Request parameters:

- Path: `ticker`.
- Query: optional `start_date`, `end_date`, `limit`.

Response shape:

```json
{
  "ticker": "SPY",
  "rows": [
    {
      "date": "2026-05-15",
      "rule_label": "stable_growth",
      "predicted_regime": "stable_growth",
      "close": 525.1,
      "return_20d": 0.034,
      "volatility_20d": 0.011,
      "drawdown_60d": -0.012
    }
  ],
  "count": 1
}
```

Expected errors:

- `404` if no history is available.
- `400` for invalid date ranges.
- `422` for invalid query parameters.

#### `GET /metrics`

Purpose:

- Return latest model evaluation metrics.

Request parameters:

- Optional `experiment_id`.

Response shape:

```json
{
  "experiment_id": "2026-05-18T120000_random_forest",
  "model_type": "random_forest",
  "accuracy": 0.78,
  "macro_f1": 0.71,
  "per_class": {
    "stable_growth": {
      "precision": 0.81,
      "recall": 0.86,
      "f1": 0.83
    }
  },
  "confusion_matrix": [[42, 3, 4, 1], [5, 18, 2, 3], [6, 2, 20, 1], [1, 4, 2, 13]]
}
```

Expected errors:

- `404` if requested metrics are unavailable.
- `500` if metrics cannot be loaded.

#### `GET /experiments`

Purpose:

- List recorded training runs.
- For the MVP, read from file-based metadata in `reports/experiments.json`.

Request parameters:

- Optional `limit`, `model_type`, `ticker`.

Response shape:

```json
{
  "experiments": [
    {
      "experiment_id": "2026-05-18T120000_random_forest",
      "created_at": "2026-05-18T12:00:00",
      "model_type": "random_forest",
      "tickers": ["SPY", "QQQ"],
      "accuracy": 0.78,
      "macro_f1": 0.71,
      "artifact_path": "models/random_forest_2026-05-18T120000.joblib"
    }
  ],
  "count": 1
}
```

Expected errors:

- `500` if experiment metadata cannot be loaded.

### G. Testing

The MVP should include a focused pytest suite.

Required tests:

- Feature calculations:
  - Returns are computed correctly.
  - Rolling volatility uses the expected window.
  - Moving average distance features use historical data only.
  - Drawdown calculation behaves correctly.
  - Rolling features use only current and prior data.

- Labeling rules:
  - Each regime can be produced from representative inputs.
  - Boundary cases are handled consistently.
  - Missing values do not silently produce misleading labels.

- Model training smoke test:
  - Training runs on a small synthetic or fixture dataset.
  - A model artifact or in-memory model is produced.
  - Metrics include expected keys.
  - Time-based split sorts by date, uses earliest 80% for training, latest 20% for testing, and does not shuffle.
  - Artifact loading restores model object, feature column order, label names, model type, training date range, ticker universe, feature version, and created timestamp.

- API health endpoint:
  - `GET /health` returns status `ok`.
  - No-model API behavior returns a clear `409` for prediction requests when no valid completed experiment artifact exists.

- Prediction response schema:
  - Prediction responses validate against Pydantic schemas.
  - Required fields are present.
  - Regime values are constrained to known labels.

### H. Documentation

Required documentation:

- `README.md`
- Setup instructions.
- Example CLI commands for data ingestion, feature generation, training, evaluation, and API startup.
- Example API calls using `curl`.
- Explanation of the four regimes.
- Explanation of feature engineering and labeling assumptions.
- Limitations and disclaimer.
- Notes about avoiding temporal leakage.
- Description of repository structure.

Required disclaimer:

RegimeLab is an educational and research-oriented ML engineering project only. It is not financial advice, does not provide trading recommendations, does not place trades, and does not guarantee profitable predictions.

## 6. MVP User Flow

1. User runs CLI commands to ingest data, generate features, train a model, and evaluate it.
2. User requests the latest regime for a ticker through `GET /regime/{ticker}`.
3. System fetches or loads cached OHLCV data.
4. System computes the latest feature row using historical data.
5. System resolves the latest completed experiment from `reports/experiments.json`.
6. System loads the trained model artifact if needed.
7. Model predicts the current regime.
8. API returns regime, confidence, class probabilities when available, key signals, model metadata, and warnings.

## 7. Example API Responses

### `GET /health`

```json
{
  "status": "ok",
  "service": "regimelab-api",
  "model_loaded": true,
  "version": "0.1.0"
}
```

### `GET /regime/SPY`

```json
{
  "ticker": "SPY",
  "as_of": "2026-05-15",
  "regime": "stable_growth",
  "confidence": 0.82,
  "probabilities": {
    "stable_growth": 0.82,
    "volatile_recovery": 0.08,
    "sideways_defensive": 0.07,
    "stress_selloff": 0.03
  },
  "key_signals": {
    "return_1d": 0.004,
    "return_5d": 0.012,
    "return_20d": 0.034,
    "volatility_20d": 0.011,
    "ma_50_distance": 0.018,
    "ma_200_distance": 0.074,
    "drawdown_60d": -0.012,
    "volume_change_20d": -0.06
  },
  "model_id": "random_forest_2026-05-18T120000",
  "warnings": []
}
```

### `GET /history/SPY`

```json
{
  "ticker": "SPY",
  "rows": [
    {
      "date": "2026-05-13",
      "rule_label": "stable_growth",
      "predicted_regime": "stable_growth",
      "close": 522.4,
      "return_20d": 0.028,
      "volatility_20d": 0.012,
      "drawdown_60d": -0.018
    },
    {
      "date": "2026-05-14",
      "rule_label": "stable_growth",
      "predicted_regime": "stable_growth",
      "close": 523.9,
      "return_20d": 0.031,
      "volatility_20d": 0.011,
      "drawdown_60d": -0.015
    },
    {
      "date": "2026-05-15",
      "rule_label": "stable_growth",
      "predicted_regime": "stable_growth",
      "close": 525.1,
      "return_20d": 0.034,
      "volatility_20d": 0.011,
      "drawdown_60d": -0.012
    }
  ],
  "count": 3
}
```

### `GET /metrics`

```json
{
  "experiment_id": "2026-05-18T120000_random_forest",
  "model_type": "random_forest",
  "tickers": ["SPY", "QQQ", "AAPL", "NVDA"],
  "train_period": {
    "start_date": "2015-01-01",
    "end_date": "2023-12-31"
  },
  "test_period": {
    "start_date": "2024-01-01",
    "end_date": "2026-05-15"
  },
  "accuracy": 0.78,
  "macro_f1": 0.71,
  "per_class": {
    "stable_growth": {
      "precision": 0.81,
      "recall": 0.86,
      "f1": 0.83
    },
    "volatile_recovery": {
      "precision": 0.69,
      "recall": 0.61,
      "f1": 0.65
    },
    "sideways_defensive": {
      "precision": 0.72,
      "recall": 0.68,
      "f1": 0.70
    },
    "stress_selloff": {
      "precision": 0.76,
      "recall": 0.66,
      "f1": 0.71
    }
  },
  "confusion_matrix": [[42, 3, 4, 1], [5, 18, 2, 3], [6, 2, 20, 1], [1, 4, 2, 13]],
  "feature_importance": [
    {
      "feature": "drawdown_60d",
      "importance": 0.21
    },
    {
      "feature": "volatility_20d",
      "importance": 0.18
    },
    {
      "feature": "return_20d",
      "importance": 0.16
    }
  ]
}
```

## 8. Suggested Repository Structure

```text
regime-lab/
  app/
    main.py
    schemas.py
    routes/
      health.py
      regime.py
      training.py
      metrics.py
  src/
    data.py
    features.py
    labeling.py
    train.py
    evaluate.py
    predict.py
    experiments.py
  models/
  data/
    raw/
    processed/
  reports/
  tests/
  notebooks/
  README.md
  PRODUCT_REQUIREMENTS.md
  Dockerfile
  pyproject.toml
```

Directory responsibilities:

- `app/`: FastAPI application, routes, and Pydantic schemas.
- `src/`: Core ML pipeline modules independent of the API layer.
- `models/`: Saved model artifacts and metadata.
- `data/raw/`: Cached raw OHLCV data.
- `data/processed/`: Engineered feature datasets.
- `reports/`: Metrics, evaluation summaries, and analysis outputs.
- `tests/`: Unit and integration tests.
- `notebooks/`: Optional exploratory notebooks.

## 9. Advanced Version 1: Hidden Markov Model Regime Detection

Advanced Version 1 should add unsupervised latent regime detection using Hidden Markov Models.

Purpose:

- Infer market states directly from return and volatility sequences.
- Compare unsupervised HMM states against rule-based and supervised labels.
- Evaluate whether latent regimes are stable, interpretable, and useful for analysis.

Implementation notes:

- Use `hmmlearn` if it is stable enough for the project environment.
- Start with features such as daily returns and rolling volatility.
- Fit models with 3 to 5 hidden states.
- Map hidden states to descriptive regimes after inspecting statistics.
- Compare state transitions over time against known stress periods.
- Save HMM artifacts separately from supervised models.
- Add an HMM report under `reports/`.
- Optionally expose HMM-inferred regimes through an endpoint such as `GET /hmm/regime/{ticker}` or include them in history reports.

Success criteria:

- HMM training runs on at least `SPY`.
- Inferred states can be summarized with average return, volatility, drawdown, and frequency.
- State sequences are stable enough to discuss in documentation.
- HMM results are compared with rule-based labels.
- The project clearly explains that HMM state names are interpreted after training, not known ahead of time.

## 10. Advanced Version 2: Forward Return and Risk Analysis

Advanced Version 2 should analyze what historically happened after each predicted or labeled regime.

Purpose:

- Explain regimes by measuring future return and risk profiles after regime observations.
- Support market research and model interpretation without presenting trading advice.

Required analysis:

- For each regime, compute forward 5-day, 20-day, and 60-day returns.
- Compute average forward return by regime.
- Compute forward volatility by regime.
- Compute max drawdown after regime occurrence.
- Compute hit rate, defined as the percentage of forward periods with positive return.
- Compare results across tickers.

Implementation notes:

- Keep forward returns out of current-regime feature generation to avoid leakage.
- Forward returns must not be included in model training features, prediction inputs, scalers, or model artifacts as input columns.
- Store analysis outputs under `reports/`.
- Add charts or tables showing regime-conditioned forward outcomes.
- Make the documentation explicit that this is retrospective analysis, not a trading recommendation.
- Consider adding an endpoint such as `GET /analysis/forward-returns`.

Success criteria:

- Forward return analysis runs on at least `SPY`, `QQQ`, `AAPL`, and `NVDA`.
- Reports include return, volatility, drawdown, and hit rate by regime.
- The analysis is reproducible from source data.
- Leakage boundaries are documented.

## 11. Advanced Version 3: Experiment Tracking and Model Registry

Advanced Version 3 should add structured experiment tracking and a lightweight model registry.

Purpose:

- Make training runs comparable and reproducible.
- Track which model artifact is active.
- Support cleaner API behavior and better technical review.

Required metadata:

- Experiment ID.
- Model type.
- Ticker universe.
- Feature set version.
- Labeling rule version.
- Train/test date ranges.
- Hyperparameters.
- Metrics.
- Artifact path.
- Created timestamp.
- Git commit hash if available.
- Notes or tags.

Implementation notes:

- MVP experiment metadata should remain file-based in `reports/experiments.json`.
- Add SQLite in this advanced version when file-based metadata becomes too limiting.
- Add `src/experiments.py` for experiment persistence.
- Add `GET /experiments`.
- Add `GET /experiments/{id}`.
- Add a way to mark the active model.
- Keep MLflow optional for a later version.

Success criteria:

- Every training run records structured metadata.
- API can list experiments.
- API can return one experiment by ID.
- Latest or active model can be resolved predictably.
- Metrics and artifacts can be traced back to a training configuration.
- SQLite-backed storage is available or clearly documented as the next step after file-based metadata.

## 12. Optional Future Enhancements

- Add XGBoost model support.
- Add a minimal Streamlit dashboard.
- Add PostgreSQL for persistent experiment metadata.
- Add scheduled data refresh.
- Add Docker Compose.
- Add CI with GitHub Actions.
- Add more assets, ETFs, sectors, and macro indicators.
- Add SHAP explanations for model interpretability.
- Add sector-level regime comparison.
- Add richer data validation with Great Expectations or Pandera.
- Add walk-forward validation.
- Add model cards for trained artifacts.
- Add configuration files for feature sets and labeling thresholds.

## 13. Success Criteria

The MVP is complete when:

- Historical data can be downloaded or loaded from cache.
- Feature engineering runs successfully for supported tickers.
- Rule-based labels are generated for the four MVP regimes.
- At least one supervised model trains successfully.
- Evaluation metrics are saved and readable.
- Model artifact is saved and loadable.
- FastAPI app runs locally.
- `GET /health` returns a valid response.
- `GET /regime/{ticker}` returns a valid prediction response.
- `GET /metrics` returns latest evaluation metrics.
- Tests pass with `pytest`.
- README clearly explains setup, usage, regimes, limitations, and disclaimers.
- Docker build and run instructions are documented.
- Project is polished enough to show on a resume and GitHub.

## 14. Resume Positioning

Possible resume bullets for the finished MVP:

- Built RegimeLab, a Python/FastAPI financial ML system that ingests historical OHLCV data, engineers time-series features, classifies market regimes, and serves predictions through typed API endpoints.
- Implemented reproducible model training and evaluation pipelines using pandas, scikit-learn, pytest, and Docker, with time-based validation, saved artifacts, metrics reporting, and leakage-aware documentation.
- Developed rule-based and supervised market regime classifiers for assets including SPY, QQQ, AAPL, and NVDA, exposing regime predictions, confidence scores, feature signals, and experiment metadata through a backend API.

Possible advanced-version resume bullets:

- Added unsupervised Hidden Markov Model regime detection and compared latent states against supervised and rule-based market regimes for interpretability and stability analysis.
- Built forward return and risk analysis reports measuring 5-day, 20-day, and 60-day outcomes by regime, including volatility, drawdown, and hit-rate summaries.
- Designed a lightweight experiment tracking and model registry layer with SQLite-backed metadata, model artifact versioning, and API endpoints for comparing training runs.

## 15. Implementation Milestones

### Milestone 1: PRD and Repo Setup

Checklist:

- [ ] Create `PRODUCT_REQUIREMENTS.md`.
- [ ] Create initial `README.md`.
- [ ] Add `pyproject.toml`.
- [ ] Add project folders.
- [ ] Add initial pytest setup.
- [ ] Add `.gitignore`.

Future Codex prompt:

```text
Implement Milestone 1 from PRODUCT_REQUIREMENTS.md.
```

### Milestone 2: Data Ingestion

Checklist:

- [ ] Implement pluggable data loader.
- [ ] Add yfinance-based loader.
- [ ] Support configured ticker list.
- [ ] Save raw data under `data/raw/`.
- [ ] Load cached data when available.
- [ ] Validate expected OHLCV columns.
- [ ] Add data ingestion tests with fixtures or mocked data.

Future Codex prompt:

```text
Implement Milestone 2 from PRODUCT_REQUIREMENTS.md.
```

### Milestone 3: Feature Engineering

Checklist:

- [ ] Implement return features.
- [ ] Implement rolling volatility.
- [ ] Implement moving average distance features.
- [ ] Implement drawdown feature.
- [ ] Implement volume change feature.
- [ ] Add feature column documentation.
- [ ] Add unit tests for feature calculations.

Future Codex prompt:

```text
Implement Milestone 3 from PRODUCT_REQUIREMENTS.md.
```

### Milestone 4: Rule-Based Labeling

Checklist:

- [ ] Define four regime labels.
- [ ] Implement configurable labeling thresholds.
- [ ] Add labeling function.
- [ ] Add representative tests for all regimes.
- [ ] Document heuristic assumptions.

Future Codex prompt:

```text
Implement Milestone 4 from PRODUCT_REQUIREMENTS.md.
```

### Milestone 5: Model Training

Checklist:

- [ ] Implement concrete time-based split: sort by date, earliest 80% train, latest 20% test, no shuffle.
- [ ] Train Logistic Regression baseline.
- [ ] Train Random Forest baseline.
- [ ] Save model artifacts.
- [ ] Save feature column metadata.
- [ ] Add model training smoke test.

Future Codex prompt:

```text
Implement Milestone 5 from PRODUCT_REQUIREMENTS.md.
```

### Milestone 6: Evaluation

Checklist:

- [ ] Compute accuracy.
- [ ] Compute macro F1.
- [ ] Compute per-class precision, recall, and F1.
- [ ] Compute confusion matrix.
- [ ] Compute feature importance for tree models.
- [ ] Save metrics JSON.
- [ ] Add evaluation tests where practical.

Future Codex prompt:

```text
Implement Milestone 6 from PRODUCT_REQUIREMENTS.md.
```

### Milestone 7: FastAPI Backend

Checklist:

- [ ] Create FastAPI app.
- [ ] Add Pydantic schemas.
- [ ] Implement `GET /health`.
- [ ] Optionally implement `POST /train` for local demonstration.
- [ ] Implement `GET /regime/{ticker}`.
- [ ] Implement `GET /history/{ticker}`.
- [ ] Implement `GET /metrics`.
- [ ] Implement `GET /experiments`.
- [ ] Add API tests for core endpoints.

Future Codex prompt:

```text
Implement Milestone 7 from PRODUCT_REQUIREMENTS.md.
```

### Milestone 8: Testing

Checklist:

- [ ] Expand unit tests for features and labeling.
- [ ] Add model smoke tests.
- [ ] Add API tests with FastAPI test client.
- [ ] Add prediction schema tests.
- [ ] Add time-based split tests.
- [ ] Add artifact loading tests.
- [ ] Add no-model API behavior tests.
- [ ] Ensure `pytest` passes locally.

Future Codex prompt:

```text
Implement Milestone 8 from PRODUCT_REQUIREMENTS.md.
```

### Milestone 9: Docker and Documentation

Checklist:

- [ ] Add Dockerfile.
- [ ] Document local setup.
- [ ] Document Docker usage.
- [ ] Add example training command.
- [ ] Add example API calls.
- [ ] Add limitations and disclaimer.
- [ ] Polish README for GitHub review.

Future Codex prompt:

```text
Implement Milestone 9 from PRODUCT_REQUIREMENTS.md.
```

### Milestone 10: Advanced HMM

Checklist:

- [ ] Add HMM dependency or optional extras.
- [ ] Implement HMM training pipeline.
- [ ] Generate HMM regime reports.
- [ ] Compare HMM states with rule-based labels.
- [ ] Document interpretation and limitations.

Future Codex prompt:

```text
Implement Milestone 10 from PRODUCT_REQUIREMENTS.md.
```

### Milestone 11: Forward Return Analysis

Checklist:

- [ ] Compute forward 5-day returns.
- [ ] Compute forward 20-day returns.
- [ ] Compute forward 60-day returns.
- [ ] Summarize return and risk by regime.
- [ ] Save reports and visualizations.
- [ ] Document leakage boundaries and disclaimer.

Future Codex prompt:

```text
Implement Milestone 11 from PRODUCT_REQUIREMENTS.md.
```

### Milestone 12: Experiment Tracking

Checklist:

- [ ] Add structured experiment metadata storage.
- [ ] Track model type, tickers, dates, features, metrics, and artifact path.
- [ ] Move beyond MVP file-based metadata when needed.
- [ ] Add active model resolution.
- [ ] Implement `GET /experiments/{id}`.
- [ ] Add tests for experiment persistence.
- [ ] Document experiment comparison workflow.

Future Codex prompt:

```text
Implement Milestone 12 from PRODUCT_REQUIREMENTS.md.
```
