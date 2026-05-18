# RegimeLab Results

This page records one reproducible sample run using cached real market data for `SPY`, `QQQ`, `AAPL`, and `NVDA`.

RegimeLab is educational and research software only. These results are not financial advice, trading recommendations, or evidence of a profitable strategy.

## Sample Run

- Tickers: `SPY`, `QQQ`, `AAPL`, `NVDA`
- Labeled feature date range: `2015-10-16` to `2026-05-14`
- Rows: `10,636` total, `2,659` per ticker
- Supervised model: `random_forest`
- Train period: `2015-10-16` to `2024-04-01`
- Test period: `2024-04-02` to `2026-05-14`
- Accuracy: `0.9948`
- Macro F1: `0.9918`

The metrics above measure agreement with heuristic `rule_label` values. They do not measure return prediction, trading profitability, or investment usefulness.

## Combined Label Distribution

| Regime | Count | Percent |
| --- | ---: | ---: |
| `stable_growth` | 3,468 | 32.6% |
| `volatile_recovery` | 903 | 8.5% |
| `sideways_defensive` | 4,450 | 41.8% |
| `stress_selloff` | 1,815 | 17.1% |

## Label Distribution by Ticker

| Ticker | `stable_growth` | `volatile_recovery` | `sideways_defensive` | `stress_selloff` |
| --- | ---: | ---: | ---: | ---: |
| `SPY` | 1,080 (40.6%) | 119 (4.5%) | 1,189 (44.7%) | 271 (10.2%) |
| `QQQ` | 1,131 (42.5%) | 148 (5.6%) | 993 (37.3%) | 387 (14.6%) |
| `AAPL` | 896 (33.7%) | 197 (7.4%) | 990 (37.2%) | 576 (21.7%) |
| `NVDA` | 361 (13.6%) | 439 (16.5%) | 1,278 (48.1%) | 581 (21.9%) |

## Latest Regimes

Latest available labeled row: `2026-05-14`.

| Ticker | Rule Label | Predicted Regime |
| --- | --- | --- |
| `SPY` | `stable_growth` | `stable_growth` |
| `QQQ` | `stable_growth` | `stable_growth` |
| `AAPL` | `stable_growth` | `stable_growth` |
| `NVDA` | `sideways_defensive` | `sideways_defensive` |

## Walk-Forward Validation

Walk-forward validation was run with expanding chronological folds from `2018` through partial `2026`. Each fold trains only on earlier dates and tests on the next chronological period, so no fold trains on future rows.

- Model: `random_forest`
- Tickers: `SPY`, `QQQ`, `AAPL`, `NVDA`
- Fold range: `2018` through partial `2026`
- Accuracy range: about `0.977` to `0.999`
- Macro F1 range: about `0.959` to `0.998`

The `2026` fold has fewer rows because the cached dataset ends on `2026-05-14`, so it is a partial year. These walk-forward metrics test whether the model consistently reproduces heuristic regime labels across time. They do not measure trading profitability or investment usefulness.

## Optional Analysis Paths

- HMM analysis: `python -m src.hmm --tickers SPY QQQ AAPL NVDA --n-states 4`
- Forward return analysis: `python -m src.forward_returns --tickers SPY QQQ AAPL NVDA --horizons 5 20 60`
- Walk-forward validation: `python -m src.walk_forward --tickers SPY QQQ AAPL NVDA --model-type random_forest --start-year 2018 --test-window-years 1`

HMM analysis requires the optional `hmmlearn` dependency. Forward-return reports are retrospective summaries of what happened after observed regimes; they are not trading signals.
Walk-forward validation evaluates whether agreement with heuristic labels is stable across multiple chronological folds.

## How to Interpret Results

- Regime labels are deterministic heuristics based on returns, volatility, moving-average distance, and drawdown.
- High model accuracy means the supervised model learned to reproduce those heuristic labels.
- High model accuracy does not mean the system predicts profitable trades.
- Forward returns use future data only for offline retrospective analysis, never for model inputs.
- HMM states are latent clusters whose names are interpreted after training from state statistics.
- Walk-forward validation uses expanding chronological folds and never trains a fold on future rows.
- If an artifact compatibility warning appears, retrain the model in the current environment before relying on local predictions.

## Resume Bullets

- Built a backend-first financial ML system in Python that ingests historical OHLCV data, engineers leakage-aware time-series features, labels market regimes, trains supervised classifiers, and serves cached predictions through FastAPI.
- Implemented reproducible model artifacts, file-based experiment metadata, chronological train/test evaluation, diagnostics, Docker packaging, and pytest coverage across data, features, labeling, training, evaluation, and API behavior.
- Added optional unsupervised HMM regime analysis, retrospective forward-return reporting, and walk-forward validation to compare heuristic, predicted, latent, and temporally validated regimes without framing outputs as trading advice.
