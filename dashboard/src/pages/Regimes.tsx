import { useState, useEffect } from "react";
import { AlertCircle, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RegimeBadge } from "@/components/dashboard/RegimeBadge";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { getMetrics, getRegime } from "@/lib/api";
import type { RegimeResponse } from "@/lib/types";

interface TickerState {
  loading: boolean;
  data: RegimeResponse | null;
  error: string | null;
}

export function Regimes() {
  const [tickers, setTickers] = useState<string[]>([]);
  const [states, setStates] = useState<Map<string, TickerState>>(new Map());
  const [globalError, setGlobalError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const m = await getMetrics();
        if (cancelled) return;
        const tickerList = m.tickers ?? [];
        setTickers(tickerList);

        // Initialize loading states
        const initial = new Map<string, TickerState>();
        tickerList.forEach((t) =>
          initial.set(t, { loading: true, data: null, error: null })
        );
        setStates(initial);

        // Fetch each regime
        const results = await Promise.allSettled(
          tickerList.map((t) => getRegime(t))
        );

        if (cancelled) return;

        const updated = new Map<string, TickerState>();
        results.forEach((r, i) => {
          const ticker = tickerList[i];
          if (r.status === "fulfilled") {
            updated.set(ticker, { loading: false, data: r.value, error: null });
          } else {
            updated.set(ticker, {
              loading: false,
              data: null,
              error: r.reason instanceof Error ? r.reason.message : "Failed",
            });
          }
        });
        setStates(updated);
      } catch (err) {
        if (!cancelled) {
          setGlobalError(
            err instanceof Error ? err.message : "Failed to load tickers"
          );
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">
          Current Regimes
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Latest model regime predictions per ticker, with confidence and key
          signals.
        </p>
      </div>

      {globalError && (
        <Alert className="border-red-500/20 bg-red-500/5">
          <AlertCircle className="h-4 w-4 text-red-400" />
          <AlertTitle className="text-red-300">API Error</AlertTitle>
          <AlertDescription className="text-red-300/70">
            {globalError}
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {tickers.length === 0 && !globalError
          ? Array.from({ length: 3 }).map((_, i) => (
              <Card key={i}>
                <CardHeader className="pb-2">
                  <Skeleton className="h-5 w-16" />
                </CardHeader>
                <CardContent className="space-y-3">
                  <Skeleton className="h-3 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                  <Skeleton className="h-3 w-2/3" />
                </CardContent>
              </Card>
            ))
          : tickers.map((ticker) => {
              const s = states.get(ticker);
              return (
                <RegimeCard key={ticker} ticker={ticker} state={s ?? null} />
              );
            })}
      </div>
    </div>
  );
}

function RegimeCard({
  ticker,
  state,
}: {
  ticker: string;
  state: TickerState | null;
}) {
  if (!state || state.loading) {
    return (
      <Card className="transition-colors">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="font-mono text-sm font-semibold">
            {ticker}
          </CardTitle>
          <Skeleton className="h-5 w-16" />
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-3 w-3/4" />
          <Skeleton className="h-3 w-1/2" />
          <Skeleton className="h-3 w-2/3" />
        </CardContent>
      </Card>
    );
  }

  if (state.error) {
    return (
      <Card className="border-red-500/10 transition-colors">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="font-mono text-sm font-semibold">
            {ticker}
          </CardTitle>
          <Badge variant="outline" className="text-xs text-red-400 border-red-500/25">
            Error
          </Badge>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">{state.error}</p>
        </CardContent>
      </Card>
    );
  }

  const d = state.data!;
  const signals = Object.entries(d.key_signals);

  return (
    <Card className="transition-colors hover:border-muted-foreground/30">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="font-mono text-sm font-semibold">
          {ticker}
        </CardTitle>
        <RegimeBadge regime={d.regime} />
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Confidence */}
        {d.confidence != null && (
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Confidence</span>
            <span className="font-mono font-medium">
              {(d.confidence * 100).toFixed(1)}%
            </span>
          </div>
        )}

        {/* Probabilities */}
        {d.probabilities && (
          <div className="space-y-1">
            {Object.entries(d.probabilities).map(([label, prob]) => (
              <div key={label} className="flex items-center gap-2">
                <span className="w-16 truncate text-xs text-muted-foreground">
                  {label}
                </span>
                <div className="h-1.5 flex-1 rounded-full bg-muted">
                  <div
                    className="h-1.5 rounded-full bg-primary/60"
                    style={{ width: `${(prob * 100).toFixed(0)}%` }}
                  />
                </div>
                <span className="w-10 text-right font-mono text-xs text-muted-foreground">
                  {(prob * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Key signals */}
        {signals.length > 0 && (
          <div className="space-y-1 border-t border-border pt-2">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-1">
              <TrendingUp className="h-3 w-3" />
              Key Signals
            </div>
            {signals.map(([key, val]) => (
              <div
                key={key}
                className="flex items-center justify-between text-xs"
              >
                <span className="text-muted-foreground">{key}</span>
                <span className="font-mono">
                  {val != null ? val.toFixed(4) : "—"}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* As-of date */}
        <p className="text-[10px] text-muted-foreground/50 pt-1">
          as of {d.as_of}
        </p>
      </CardContent>
    </Card>
  );
}
