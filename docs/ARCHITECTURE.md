# RegimeLab Architecture

RegimeLab is organized as a CLI-first ML pipeline with a cached-artifact FastAPI layer.

```text
data/raw -> data/processed -> labeled features -> models/reports -> FastAPI
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

5. `src.evaluate`
   - Reloads artifacts.
   - Reconstructs the same chronological test split.
   - Saves metrics under `reports/`.

6. `app/`
   - Serves cached predictions, history, metrics, and experiments through FastAPI.
   - Does not download live market data or train models inside request handlers.

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
