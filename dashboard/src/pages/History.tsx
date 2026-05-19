import { useState, useEffect, useCallback } from "react";
import { Clock, AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { RegimeBadge } from "@/components/dashboard/RegimeBadge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { getMetrics, getHistory } from "@/lib/api";
import type { HistoryResponse } from "@/lib/types";

export function History() {
  const [tickers, setTickers] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics()
      .then((m) => {
        const list = m.tickers ?? [];
        setTickers(list);
        if (list.length > 0) setSelected(list[0]);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed"));
  }, []);

  const load = useCallback(async (ticker: string) => {
    setLoading(true);
    setError(null);
    try {
      setHistory(await getHistory(ticker, 100));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load history");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selected) load(selected);
  }, [selected, load]);

  const fmt = (n: number | null | undefined, d = 4) =>
    n != null ? n.toFixed(d) : "—";

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">Historical Data</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Browse historical regime labels, predictions, and market indicators.
        </p>
      </div>

      {error && (
        <Alert className="border-red-500/20 bg-red-500/5">
          <AlertCircle className="h-4 w-4 text-red-400" />
          <AlertTitle className="text-red-300">API Error</AlertTitle>
          <AlertDescription className="text-red-300/70">{error}</AlertDescription>
        </Alert>
      )}

      {tickers.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {tickers.map((t) => (
            <Button
              key={t}
              variant={t === selected ? "secondary" : "outline"}
              size="sm"
              className={`font-mono text-xs ${t === selected ? "bg-accent text-accent-foreground" : "text-muted-foreground"}`}
              onClick={() => setSelected(t)}
            >
              {t}
            </Button>
          ))}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-sm font-medium">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              {selected ? `${selected} History` : "History"}
            </div>
            {history && (
              <span className="text-xs font-normal text-muted-foreground">
                {history.count} rows
                {history.total_available != null && ` of ${history.total_available}`}
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-3">
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} className="flex items-center gap-4">
                  <Skeleton className="h-4 w-20" />
                  <Skeleton className="h-4 flex-1" />
                  <Skeleton className="h-4 w-16" />
                </div>
              ))}
            </div>
          ) : history && history.rows.length > 0 ? (
            <div className="overflow-x-auto -mx-6">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-xs">Date</TableHead>
                    <TableHead className="text-xs">Rule Label</TableHead>
                    <TableHead className="text-xs">Predicted</TableHead>
                    <TableHead className="text-xs text-right">Close</TableHead>
                    <TableHead className="text-xs text-right">Ret 20d</TableHead>
                    <TableHead className="text-xs text-right">Vol 20d</TableHead>
                    <TableHead className="text-xs text-right">DD 60d</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {history.rows.map((row, i) => (
                    <TableRow key={`${row.date}-${i}`}>
                      <TableCell className="font-mono text-xs">{row.date}</TableCell>
                      <TableCell>
                        {row.rule_label ? <RegimeBadge regime={row.rule_label} /> : <span className="text-xs text-muted-foreground">—</span>}
                      </TableCell>
                      <TableCell>
                        {row.predicted_regime ? <RegimeBadge regime={row.predicted_regime} /> : <span className="text-xs text-muted-foreground">—</span>}
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs">{row.close != null ? row.close.toFixed(2) : "—"}</TableCell>
                      <TableCell className="text-right font-mono text-xs">{fmt(row.return_20d)}</TableCell>
                      <TableCell className="text-right font-mono text-xs">{fmt(row.volatility_20d)}</TableCell>
                      <TableCell className="text-right font-mono text-xs">{fmt(row.drawdown_60d)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <p className="text-center text-sm text-muted-foreground py-8">
              No history data available for this ticker.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
