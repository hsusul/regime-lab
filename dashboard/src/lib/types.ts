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

/** Navigation page identifiers. */
export type PageId = "overview" | "regimes" | "history" | "metrics" | "reports";
