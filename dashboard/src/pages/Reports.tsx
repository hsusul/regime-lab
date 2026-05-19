import { useState, useEffect } from "react";
import { FileText, FlaskConical, AlertCircle, CalendarDays } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { getExperiments } from "@/lib/api";
import type { ExperimentsResponse } from "@/lib/types";

export function Reports() {
  const [data, setData] = useState<ExperimentsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getExperiments(20)
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "Failed"); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">Experiment Reports</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Browse recorded training experiments, parameters, and evaluation summaries.
        </p>
      </div>

      {error && (
        <Alert className="border-red-500/20 bg-red-500/5">
          <AlertCircle className="h-4 w-4 text-red-400" />
          <AlertTitle className="text-red-300">API Error</AlertTitle>
          <AlertDescription className="text-red-300/70">{error}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-sm font-medium">
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-muted-foreground" />
              Recent Experiments
            </div>
            {data && (
              <span className="text-xs font-normal text-muted-foreground">
                {data.count} experiment{data.count !== 1 ? "s" : ""}
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {loading ? (
            Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="flex items-center gap-4 rounded-md border border-border px-4 py-3">
                <FlaskConical className="h-4 w-4 shrink-0 text-muted-foreground/40" />
                <div className="flex-1 space-y-1.5">
                  <Skeleton className="h-4 w-48" />
                  <Skeleton className="h-3 w-32" />
                </div>
                <Skeleton className="h-5 w-16" />
              </div>
            ))
          ) : data && data.experiments.length > 0 ? (
            data.experiments.map((exp, i) => (
              <ExperimentRow key={String(exp.experiment_id ?? i)} exp={exp} />
            ))
          ) : (
            <p className="text-center text-sm text-muted-foreground py-8">
              No experiments recorded yet.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ExperimentRow({ exp }: { exp: Record<string, unknown> }) {
  const id = String(exp.experiment_id ?? "—");
  const model = String(exp.model_type ?? exp.model ?? "—");
  const tickers = Array.isArray(exp.tickers) ? (exp.tickers as string[]).join(", ") : "—";
  const acc = typeof exp.accuracy === "number" ? `${(exp.accuracy * 100).toFixed(1)}%` : null;
  const f1 = typeof exp.macro_f1 === "number" ? exp.macro_f1.toFixed(3) : null;
  const ts = typeof exp.timestamp === "string" ? exp.timestamp : typeof exp.created_at === "string" ? exp.created_at : null;

  return (
    <div className="rounded-md border border-border px-4 py-3 hover:border-muted-foreground/30 transition-colors">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3 min-w-0">
          <FlaskConical className="h-4 w-4 shrink-0 text-violet-400 mt-0.5" />
          <div className="min-w-0">
            <p className="font-mono text-sm font-medium truncate">{id}</p>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1 text-xs text-muted-foreground">
              <span>Model: <span className="text-foreground">{model}</span></span>
              <span>Tickers: <span className="text-foreground">{tickers}</span></span>
              {ts && (
                <span className="flex items-center gap-1">
                  <CalendarDays className="h-3 w-3" />
                  {ts}
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {acc && <Badge variant="outline" className="text-xs font-mono text-emerald-400 border-emerald-500/25">Acc {acc}</Badge>}
          {f1 && <Badge variant="outline" className="text-xs font-mono text-sky-400 border-sky-500/25">F1 {f1}</Badge>}
        </div>
      </div>
    </div>
  );
}
