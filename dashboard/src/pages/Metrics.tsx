import { useState, useEffect } from "react";
import { BarChart3, Target, Percent, AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { getMetrics } from "@/lib/api";
import type { MetricsResponse } from "@/lib/types";

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

      {/* Top summary cards */}
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

      <div className="grid gap-4 md:grid-cols-2">
        {/* Per-class performance */}
        <Card>
          <CardHeader><CardTitle className="text-sm font-medium">Per-Class Performance</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            {loading ? (
              Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="flex items-center justify-between rounded-md border border-border px-3 py-2">
                  <Skeleton className="h-4 w-16" /><Skeleton className="h-4 w-24" />
                </div>
              ))
            ) : data && Object.keys(data.per_class).length > 0 ? (
              Object.entries(data.per_class).map(([label, scores]) => (
                <div key={label} className="flex items-center justify-between rounded-md border border-border px-3 py-2">
                  <Badge variant="outline" className="font-mono text-xs">{label}</Badge>
                  <div className="flex gap-4 text-xs font-mono">
                    <span className="text-muted-foreground">P <span className="text-foreground">{fmt(scores.precision)}</span></span>
                    <span className="text-muted-foreground">R <span className="text-foreground">{fmt(scores.recall)}</span></span>
                    <span className="text-muted-foreground">F1 <span className="text-foreground">{fmt(scores["f1-score"])}</span></span>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-sm text-muted-foreground">No per-class data.</p>
            )}
          </CardContent>
        </Card>

        {/* Feature importance */}
        <Card>
          <CardHeader><CardTitle className="text-sm font-medium">Feature Importance</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            {loading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="flex items-center gap-3"><Skeleton className="h-4 w-28" /><Skeleton className="h-3 flex-1" /></div>
              ))
            ) : data && data.feature_importance.length > 0 ? (
              (() => {
                const items = data.feature_importance
                  .map((f) => ({ name: String(f.feature ?? f.name ?? ""), importance: Number(f.importance ?? f.value ?? 0) }))
                  .sort((a, b) => b.importance - a.importance)
                  .slice(0, 12);
                const max = Math.max(...items.map((i) => i.importance), 1e-9);
                return items.map((item) => (
                  <div key={item.name} className="flex items-center gap-3">
                    <span className="w-32 truncate text-xs text-muted-foreground font-mono">{item.name}</span>
                    <div className="h-2 flex-1 rounded-full bg-muted">
                      <div className="h-2 rounded-full bg-primary/50" style={{ width: `${(item.importance / max * 100).toFixed(0)}%` }} />
                    </div>
                    <span className="w-12 text-right font-mono text-xs text-muted-foreground">{item.importance.toFixed(3)}</span>
                  </div>
                ));
              })()
            ) : (
              <p className="text-sm text-muted-foreground">No feature importance data.</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Confusion matrix */}
      {!loading && data && data.confusion_matrix.length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-sm font-medium">Confusion Matrix</CardTitle></CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="text-xs font-mono">
                <thead>
                  <tr>
                    <th className="px-2 py-1 text-left text-muted-foreground">Actual \ Pred</th>
                    {data.label_names.map((l) => (
                      <th key={l} className="px-2 py-1 text-center text-muted-foreground">{l}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.confusion_matrix.map((row, ri) => (
                    <tr key={ri}>
                      <td className="px-2 py-1 text-muted-foreground">{data.label_names[ri] ?? ri}</td>
                      {row.map((cell, ci) => {
                        const isDiag = ri === ci;
                        return (
                          <td key={ci} className={`px-2 py-1 text-center ${isDiag ? "text-emerald-400 font-semibold" : "text-muted-foreground"}`}>
                            {cell}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Period info */}
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
