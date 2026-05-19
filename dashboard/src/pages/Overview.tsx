import { useState, useEffect } from "react";
import {
  Activity,
  BarChart3,
  Target,
  Layers,
  FlaskConical,
  AlertCircle,
} from "lucide-react";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { RegimeBadge } from "@/components/dashboard/RegimeBadge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { getMetrics, getRegime } from "@/lib/api";
import type { MetricsResponse, RegimeResponse } from "@/lib/types";

export function Overview() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [regimes, setRegimes] = useState<Map<string, RegimeResponse>>(new Map());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);

      try {
        const m = await getMetrics();
        if (cancelled) return;
        setMetrics(m);

        // Fetch regime for each ticker from the metrics response
        const tickers = m.tickers ?? [];
        const results = await Promise.allSettled(
          tickers.map((t) => getRegime(t))
        );

        if (cancelled) return;

        const map = new Map<string, RegimeResponse>();
        results.forEach((r, i) => {
          if (r.status === "fulfilled") {
            map.set(tickers[i], r.value);
          }
        });
        setRegimes(map);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load data");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="space-y-6">
      {/* Page intro */}
      <div>
        <h2 className="text-xl font-semibold tracking-tight">
          Research Overview
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Model performance and regime classification summary.
        </p>
      </div>

      {/* Error banner */}
      {error && (
        <Alert className="border-red-500/20 bg-red-500/5">
          <AlertCircle className="h-4 w-4 text-red-400" />
          <AlertTitle className="text-red-300">API Error</AlertTitle>
          <AlertDescription className="text-red-300/70">
            {error}
          </AlertDescription>
        </Alert>
      )}

      {/* Metric cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {loading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <CardHeader className="pb-2">
                <Skeleton className="h-4 w-24" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-7 w-20" />
                <Skeleton className="mt-1 h-3 w-32" />
              </CardContent>
            </Card>
          ))
        ) : (
          <>
            <MetricCard
              title="Active Model"
              value={metrics?.model_type ?? "—"}
              description="Latest experiment model type"
              icon={FlaskConical}
              iconColor="text-violet-400"
            />
            <MetricCard
              title="Macro F1"
              value={metrics?.macro_f1 != null ? metrics.macro_f1.toFixed(3) : "—"}
              description="Weighted across regime classes"
              icon={Target}
              iconColor="text-emerald-400"
            />
            <MetricCard
              title="Accuracy"
              value={
                metrics?.accuracy != null
                  ? `${(metrics.accuracy * 100).toFixed(1)}%`
                  : "—"
              }
              description="Test-set overall accuracy"
              icon={BarChart3}
              iconColor="text-sky-400"
            />
            <MetricCard
              title="Tracked Tickers"
              value={String(metrics?.tickers?.length ?? 0)}
              description={metrics?.tickers?.join(" · ") ?? ""}
              icon={Layers}
              iconColor="text-amber-400"
            />
          </>
        )}
      </div>

      <Separator />

      {/* Bottom row */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Regime snapshot */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Activity className="h-4 w-4 text-muted-foreground" />
              Recent Regime Snapshot
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {loading ? (
              Array.from({ length: 4 }).map((_, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between rounded-md border border-border px-3 py-2"
                >
                  <Skeleton className="h-4 w-12" />
                  <Skeleton className="h-5 w-16" />
                </div>
              ))
            ) : metrics?.tickers?.length ? (
              metrics.tickers.map((ticker) => {
                const r = regimes.get(ticker);
                return (
                  <div
                    key={ticker}
                    className="flex items-center justify-between rounded-md border border-border px-3 py-2"
                  >
                    <span className="font-mono text-sm font-medium">
                      {ticker}
                    </span>
                    {r ? (
                      <div className="flex items-center gap-3">
                        <RegimeBadge regime={r.regime} />
                        {r.confidence != null && (
                          <span className="text-xs text-muted-foreground font-mono">
                            {(r.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    ) : (
                      <span className="text-xs text-muted-foreground">
                        unavailable
                      </span>
                    )}
                  </div>
                );
              })
            ) : (
              <p className="text-sm text-muted-foreground">
                No tickers available.
              </p>
            )}
          </CardContent>
        </Card>

        {/* About card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <FlaskConical className="h-4 w-4 text-muted-foreground" />
              About RegimeLab
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              RegimeLab is an{" "}
              <strong className="text-foreground">
                educational research platform
              </strong>{" "}
              for studying market regime classification using machine learning.
            </p>
            <p>
              It trains supervised models on historical market data to identify
              macro regimes (e.g. bull, bear, crisis, recovery) and evaluates
              classification performance rigorously.
            </p>
            <p className="text-xs text-muted-foreground/60">
              This tool is for research purposes only and does not constitute
              financial advice. All predictions are retrospective model outputs
              — not forward-looking signals.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
