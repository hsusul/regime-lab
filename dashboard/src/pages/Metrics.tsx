import { useState, useEffect, useMemo } from "react";
import { BarChart3, Target, Percent, AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell,
} from "recharts";
import { getMetrics } from "@/lib/api";
import type { MetricsResponse } from "@/lib/types";

/* ── chart tooltip ─────────────────────────────────────────────────── */
function BarTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value?: number; name?: string; fill?: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-border bg-card px-3 py-2 text-xs shadow-lg">
      <p className="font-mono text-muted-foreground mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} className="font-mono" style={{ color: p.fill }}>
          {p.name}: {p.value != null ? p.value.toFixed(4) : "—"}
        </p>
      ))}
    </div>
  );
}

/* ── heatmap cell color ────────────────────────────────────────────── */
function heatmapBg(value: number, maxVal: number, isDiag: boolean): string {
  if (maxVal === 0) return "transparent";
  const ratio = value / maxVal;
  if (isDiag) {
    // Green gradient for diagonal (correct predictions)
    const alpha = 0.15 + ratio * 0.55;
    return `rgba(52, 211, 153, ${alpha.toFixed(2)})`;  // emerald
  }
  // Red gradient for off-diagonal (misclassifications)
  if (value === 0) return "transparent";
  const alpha = 0.1 + ratio * 0.4;
  return `rgba(248, 113, 113, ${alpha.toFixed(2)})`;  // red
}

/* ── main component ────────────────────────────────────────────────── */
export function Metrics() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getMetrics()
      .then((m) => { if (!cancelled) setData(m); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "Failed"); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const fmt = (n: number | null | undefined, d = 3) =>
    n != null ? n.toFixed(d) : "—";

  /* Feature importance sorted for bar chart */
  const featureData = useMemo(() => {
    if (!data) return [];
    return data.feature_importance
      .map((f) => ({
        name: String(f.feature ?? f.name ?? ""),
        importance: Number(f.importance ?? f.value ?? 0),
      }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 12);
  }, [data]);

  /* Per-class F1 data for bar chart */
  const f1Data = useMemo(() => {
    if (!data) return [];
    return Object.entries(data.per_class).map(([label, scores]) => ({
      label,
      precision: scores.precision ?? 0,
      recall: scores.recall ?? 0,
      f1: scores["f1-score"] ?? 0,
    }));
  }, [data]);

  /* Confusion matrix max value (for heatmap normalization) */
  const cmMax = useMemo(() => {
    if (!data || data.confusion_matrix.length === 0) return 0;
    return Math.max(...data.confusion_matrix.flat());
  }, [data]);

  /* Bar palette */
  const BAR_COLORS = ["#60a5fa", "#34d399", "#f59e0b", "#a78bfa", "#f87171", "#38bdf8"];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">Model Metrics</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Evaluation metrics, per-class performance, confusion matrix, and feature importance.
        </p>
      </div>

      {error && (
        <Alert className="border-red-500/20 bg-red-500/5">
          <AlertCircle className="h-4 w-4 text-red-400" />
          <AlertTitle className="text-red-300">API Error</AlertTitle>
          <AlertDescription className="text-red-300/70">{error}</AlertDescription>
        </Alert>
      )}

      {/* ── top summary cards ───────────────────────────────────── */}
      <div className="grid gap-4 sm:grid-cols-3">
        {loading ? (
          Array.from({ length: 3 }).map((_, i) => (
            <Card key={i}><CardHeader className="pb-2"><Skeleton className="h-4 w-24" /></CardHeader><CardContent><Skeleton className="h-7 w-20" /></CardContent></Card>
          ))
        ) : (
          <>
            <MetricCard title="Accuracy" value={data?.accuracy != null ? `${(data.accuracy * 100).toFixed(1)}%` : "—"} icon={Percent} iconColor="text-sky-400" />
            <MetricCard title="Macro F1" value={fmt(data?.macro_f1)} icon={Target} iconColor="text-emerald-400" />
            <MetricCard title="Model Type" value={data?.model_type ?? "—"} icon={BarChart3} iconColor="text-violet-400" />
          </>
        )}
      </div>

      {/* ── charts row ──────────────────────────────────────────── */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Feature importance bar chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Feature Importance</CardTitle>
          </CardHeader>
          <CardContent className="h-72">
            {loading ? (
              <Skeleton className="h-full w-full" />
            ) : featureData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureData} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                    tickLine={false}
                    axisLine={false}
                    width={110}
                  />
                  <Tooltip content={<BarTooltip />} cursor={{ fill: "hsl(var(--accent))", opacity: 0.15 }} />
                  <Bar dataKey="importance" name="Importance" radius={[0, 4, 4, 0]} animationDuration={600}>
                    {featureData.map((_, i) => (
                      <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} fillOpacity={0.75} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="flex h-full items-center justify-center text-sm text-muted-foreground">No feature importance data.</p>
            )}
          </CardContent>
        </Card>

        {/* Per-class F1 bar chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Per-Class Performance</CardTitle>
          </CardHeader>
          <CardContent className="h-72">
            {loading ? (
              <Skeleton className="h-full w-full" />
            ) : f1Data.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={f1Data} margin={{ top: 0, right: 16, bottom: 0, left: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
                  <XAxis
                    dataKey="label"
                    tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                    tickLine={false}
                    axisLine={false}
                    interval={0}
                    angle={-15}
                    textAnchor="end"
                    height={50}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                    tickLine={false}
                    axisLine={false}
                    domain={[0, 1]}
                    width={36}
                  />
                  <Tooltip content={<BarTooltip />} cursor={{ fill: "hsl(var(--accent))", opacity: 0.15 }} />
                  <Bar dataKey="precision" name="Precision" fill="#60a5fa" fillOpacity={0.7} radius={[4, 4, 0, 0]} animationDuration={600} />
                  <Bar dataKey="recall" name="Recall" fill="#34d399" fillOpacity={0.7} radius={[4, 4, 0, 0]} animationDuration={600} />
                  <Bar dataKey="f1" name="F1" fill="#f59e0b" fillOpacity={0.7} radius={[4, 4, 0, 0]} animationDuration={600} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="flex h-full items-center justify-center text-sm text-muted-foreground">No per-class data.</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── confusion matrix heatmap ────────────────────────────── */}
      {!loading && data && data.confusion_matrix.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Confusion Matrix</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs font-mono">
                <thead>
                  <tr>
                    <th className="px-3 py-2 text-left text-muted-foreground font-normal border-b border-border">
                      Actual ↓ \ Pred →
                    </th>
                    {data.label_names.map((l) => (
                      <th key={l} className="px-3 py-2 text-center text-muted-foreground font-normal border-b border-border whitespace-nowrap">
                        {l}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.confusion_matrix.map((row, ri) => (
                    <tr key={ri}>
                      <td className="px-3 py-2 text-muted-foreground whitespace-nowrap border-b border-border/50">
                        {data.label_names[ri] ?? ri}
                      </td>
                      {row.map((cell, ci) => {
                        const isDiag = ri === ci;
                        return (
                          <td
                            key={ci}
                            className="px-3 py-2 text-center border-b border-border/50 transition-colors"
                            style={{
                              backgroundColor: heatmapBg(cell, cmMax, isDiag),
                              color: isDiag ? "#34d399" : cell > 0 ? "#fca5a5" : "hsl(var(--muted-foreground))",
                              fontWeight: isDiag ? 600 : 400,
                            }}
                          >
                            {cell}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-3 flex items-center gap-6 text-[10px] text-muted-foreground/60">
              <div className="flex items-center gap-1.5">
                <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: "rgba(52, 211, 153, 0.5)" }} />
                Correct predictions
              </div>
              <div className="flex items-center gap-1.5">
                <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: "rgba(248, 113, 113, 0.35)" }} />
                Misclassifications
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── period info ─────────────────────────────────────────── */}
      {!loading && data && (
        <div className="text-xs text-muted-foreground/60 space-y-0.5">
          {data.train_period && <p>Train: {data.train_period.start} → {data.train_period.end}</p>}
          {data.test_period && <p>Test: {data.test_period.start} → {data.test_period.end}</p>}
          <p>Experiment: {data.experiment_id}</p>
        </div>
      )}
    </div>
  );
}
