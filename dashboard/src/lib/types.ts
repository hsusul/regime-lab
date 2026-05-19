/**
 * TypeScript types for RegimeLab FastAPI responses.
 * These mirror the Pydantic schemas defined in app/schemas.py.
 */

export interface HealthResponse {
  status: string;
  service: string;
  model_loaded: boolean;
  version: string;
  latest_experiment_id: string | null;
}

export interface RegimeResponse {
  ticker: string;
  as_of: string;
  regime: string;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  key_signals: Record<string, number | null>;
  experiment_id: string;
  model_id: string;
  warnings: string[];
}

export interface HistoryRow {
  date: string;
  rule_label: string | null;
  predicted_regime: string | null;
  close: number | null;
  return_20d: number | null;
  volatility_20d: number | null;
  drawdown_60d: number | null;
}

export interface HistoryResponse {
  ticker: string;
  rows: HistoryRow[];
  count: number;
  total_available: number | null;
  warnings: string[];
}

export interface MetricsResponse {
  experiment_id: string;
  model_type: string | null;
  tickers: string[];
  train_period: Record<string, string> | null;
  test_period: Record<string, string> | null;
  accuracy: number | null;
  macro_f1: number | null;
  per_class: Record<string, Record<string, number>>;
  confusion_matrix: number[][];
  label_names: string[];
  feature_importance: Array<Record<string, number | string>>;
  warnings: string[];
}

export interface ExperimentsResponse {
  experiments: Array<Record<string, unknown>>;
  count: number;
}

/* ── Report endpoint response types ────────────────────────────────── */

export interface HmmStateSummary {
  state: number;
  frequency: number;
  pct: number;
  average_return_1d: number;
  average_return_20d: number;
  average_volatility_20d: number;
  average_drawdown_60d: number;
  interpreted_regime: string;
}

export interface HmmReport {
  hmm_run_id: string;
  created_at: string;
  analysis_type: string;
  n_states: number;
  tickers: string[];
  rows: number;
  date_range: { start_date: string; end_date: string };
  state_summary: Record<string, HmmStateSummary>;
  warnings: string[];
}

export interface ForwardReturnHorizonStats {
  valid_count: number;
  average_forward_return: number;
  median_forward_return: number;
  forward_volatility: number;
  worst_forward_return: number;
  best_forward_return: number;
}

export interface ForwardReturnReport {
  analysis_type: string;
  created_at: string;
  tickers: string[];
  rows: number;
  date_range: { start_date: string; end_date: string };
  horizons: number[];
  summaries: {
    combined: {
      rule_label: Record<string, {
        observation_count: number;
        horizons: Record<string, ForwardReturnHorizonStats>;
      }>;
    };
  };
  warnings: string[];
}

export interface WalkForwardFold {
  fold_index: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  train_rows: number;
  test_rows: number;
  accuracy: number;
  macro_f1: number;
  per_class_f1: Record<string, number>;
}

export interface WalkForwardReport {
  analysis_type: string;
  created_at: string;
  model_type: string;
  tickers: string[];
  rows: number;
  folds: WalkForwardFold[];
  summary: {
    fold_count: number;
    mean_accuracy: number;
    mean_macro_f1: number;
    min_macro_f1: number;
    max_macro_f1: number;
  };
  warnings: string[];
}

export interface ModelComparisonReport {
  analysis_type: string;
  created_at: string;
  tickers: string[];
  models: Array<Record<string, unknown>>;
  best_model_type: string;
  warnings: string[];
}

export interface ExplainReport {
  analysis_type: string;
  created_at: string;
  experiment_id: string;
  model_type: string;
  feature_importance: Array<Record<string, number | string>>;
  warnings: string[];
}

/** Navigation page identifiers. */
export type PageId = "overview" | "regimes" | "history" | "metrics" | "reports";

