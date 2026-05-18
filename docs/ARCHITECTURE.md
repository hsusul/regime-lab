# RegimeLab Architecture

RegimeLab is organized as a CLI-first ML pipeline with a cached-artifact FastAPI layer and separate offline analysis modules.

```text
data/raw -> data/processed -> labeled features -> models/reports -> FastAPI
                                         \-> HMM reports
                                         \-> forward return reports
                                         \-> walk-forward reports
```

## Pipeline Stages

1. `src.data`
   - Downloads daily OHLCV data through a pluggable provider.
   - Normalizes columns to `date, open, high, low, close, adj_close, volume, ticker`.
   - Caches raw CSVs under `data/raw/`.

2. `src.features`
   - Computes leakage-aware rolling features separately per ticker.
   - Writes processed feature CSVs under `data/processed/`.

3. `src.labeling`
   - Adds deterministic heuristic `rule_label` values.
   - Writes labeled processed CSVs under `data/processed/`.

4. `src.train`
   - Loads labeled data.
   - Uses a shared chronological 80/20 split.
   - Trains `logistic_regression` or `random_forest`.
   - Saves joblib artifacts under `models/`.
   - Appends metadata to `reports/experiments.json`.
   - Mirrors metadata into the optional SQLite registry at `reports/regimelab.db`.

5. `src.evaluate`
   - Reloads artifacts.
   - Reconstructs the same chronological test split.
   - Saves metrics under `reports/`.

6. `app/`
   - Serves cached predictions, history, metrics, and experiments through FastAPI.
   - Does not download live market data or train models inside request handlers.

7. `src.hmm`
   - Optional unsupervised latent-state analysis using `return_1d` and `volatility_20d`.
   - Saves separate HMM artifacts under `models/hmm/`.
   - Writes exploratory state summaries under `reports/`.

8. `src.forward_returns`
   - Computes retrospective forward returns for offline analysis only.
   - Writes JSON reports and optional CSV summaries under `reports/`.
   - Does not modify `FEATURE_COLUMNS`, training inputs, or supervised artifacts.

9. `src.experiment_registry`
   - Maintains an optional SQLite registry for experiment records.
   - Supports `list`, `show`, and `activate` CLI commands.
   - Provides an active-model pointer while preserving `experiments.json` fallback.

10. `src.walk_forward`
    - Runs optional expanding-window validation across chronological folds.
    - Reuses supervised baseline model definitions and `FEATURE_COLUMNS`.
    - Writes JSON reports and optional CSV fold summaries under `reports/`.
    - Does not update model artifacts, experiment registry state, or API behavior.

## Artifact Contract

Training artifacts are joblib dictionaries containing:

- `model`
- `feature_columns`
- `label_names`
- `model_type`
- `training_date_range`
- `test_date_range`
- `ticker_universe`
- `feature_version`
- `labeling_version`
- `created_at`
- `experiment_id`
- environment metadata such as Python, scikit-learn, pandas, and NumPy versions

Generated local data and reports are ignored by git. Source code, docs, tests, and `.gitkeep` placeholders are tracked.

## Experiment Resolution

The model-serving path resolves experiments in this order:

1. If `reports/regimelab.db` exists and contains an active experiment with a valid `artifact_path`, use that experiment.
2. Otherwise, read `reports/experiments.json` and select the most recent completed experiment with a valid `artifact_path`.

This keeps the original file-based MVP behavior intact while allowing explicit active-model selection through SQLite.
