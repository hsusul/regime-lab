import type {
  HealthResponse,
  RegimeResponse,
  MetricsResponse,
  ExperimentsResponse,
  HistoryResponse,
  HmmReport,
  ForwardReturnReport,
  WalkForwardReport,
  ModelComparisonReport,
  ExplainReport,
} from "./types";

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText} — ${path}`);
  }
  return res.json() as Promise<T>;
}

/** Fetch JSON, returning null on 404 (report not generated yet). */
async function fetchJSONOrNull<T>(path: string): Promise<T | null> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText} — ${path}`);
  }
  return res.json() as Promise<T>;
}

/** GET /health — service status and model availability. */
export function getHealth(): Promise<HealthResponse> {
  return fetchJSON<HealthResponse>("/health");
}

/** GET /metrics — evaluation metrics for the latest or requested experiment. */
export function getMetrics(experimentId = "latest"): Promise<MetricsResponse> {
  return fetchJSON<MetricsResponse>(`/metrics?experiment_id=${encodeURIComponent(experimentId)}`);
}

/** GET /experiments — list recorded training runs. */
export function getExperiments(limit?: number): Promise<ExperimentsResponse> {
  const params = limit != null ? `?limit=${limit}` : "";
  return fetchJSON<ExperimentsResponse>(`/experiments${params}`);
}

/** GET /regime/:ticker — latest cached regime prediction. */
export function getRegime(ticker: string): Promise<RegimeResponse> {
  return fetchJSON<RegimeResponse>(`/regime/${encodeURIComponent(ticker)}`);
}

/** GET /history/:ticker — historical labels and predictions. */
export function getHistory(ticker: string, limit?: number): Promise<HistoryResponse> {
  const params = limit != null ? `?limit=${limit}` : "";
  return fetchJSON<HistoryResponse>(`/history/${encodeURIComponent(ticker)}${params}`);
}

/* ── Report endpoints (return null on 404) ─────────────────────────── */

/** GET /reports/hmm/latest */
export function getHmmReport(): Promise<HmmReport | null> {
  return fetchJSONOrNull<HmmReport>("/reports/hmm/latest");
}

/** GET /reports/forward-returns/latest */
export function getForwardReturnsReport(): Promise<ForwardReturnReport | null> {
  return fetchJSONOrNull<ForwardReturnReport>("/reports/forward-returns/latest");
}

/** GET /reports/walk-forward/latest */
export function getWalkForwardReport(): Promise<WalkForwardReport | null> {
  return fetchJSONOrNull<WalkForwardReport>("/reports/walk-forward/latest");
}

/** GET /reports/model-comparison/latest */
export function getModelComparisonReport(): Promise<ModelComparisonReport | null> {
  return fetchJSONOrNull<ModelComparisonReport>("/reports/model-comparison/latest");
}

/** GET /reports/explain/latest */
export function getExplainReport(): Promise<ExplainReport | null> {
  return fetchJSONOrNull<ExplainReport>("/reports/explain/latest");
}
