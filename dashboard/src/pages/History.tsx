import { useState, useEffect, useCallback, useMemo } from "react";
import { Clock, AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { RegimeBadge } from "@/components/dashboard/RegimeBadge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
} from "recharts";
import { getMetrics, getHistory } from "@/lib/api";
import type { HistoryResponse } from "@/lib/types";

/* ── constants ─────────────────────────────────────────────────────── */
const TICKERS = ["SPY", "QQQ", "AAPL", "NVDA"];
const LIMITS = [100, 250, 500, 1000] as const;

const CHART_COLORS = {
  close: "#60a5fa",        // sky-400
  return_20d: "#34d399",   // emerald-400
  volatility_20d: "#f59e0b", // amber-500
  drawdown_60d: "#f87171", // red-400
} as const;

/* ── tooltip ───────────────────────────────────────────────────────── */
function ChartTooltip({ active, payload, label, valueKey }: {
  active?: boolean;
  payload?: Array<{ value?: number }>;
  label?: string;
  valueKey: string;
}) {
  if (!active || !payload?.length) return null;
  const v = payload[0]?.value;
  return (
    <div className="rounded-md border border-border bg-card px-3 py-2 text-xs shadow-lg">
      <p className="font-mono text-muted-foreground">{label}</p>
      <p className="font-mono font-medium text-foreground">
        {valueKey}: {v != null ? v.toFixed(4) : "—"}
      </p>
    </div>
  );
}

/* ── mini chart wrapper ────────────────────────────────────────────── */
function MiniChart({ data, dataKey, color, title }: {
  data: Array<Record<string, unknown>>;
  dataKey: string;
  color: string;
  title: string;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs font-medium text-muted-foreground">{title}</CardTitle>
      </CardHeader>
      <CardContent className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              tickLine={false}
              axisLine={false}
              minTickGap={40}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              tickLine={false}
              axisLine={false}
              width={52}
              tickFormatter={(v: number) => {
                if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(0)}k`;
                if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1);
                return v.toFixed(2);
              }}
            />
            <Tooltip content={<ChartTooltip valueKey={dataKey} />} />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={1.5}
              dot={false}
              animationDuration={600}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

/* ── main component ────────────────────────────────────────────────── */
export function History() {
  const [tickers, setTickers] = useState<string[]>([]);
  const [selected, setSelected] = useState<string>(TICKERS[0]);
  const [limit, setLimit] = useState<number>(250);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  /* Load ticker list from /metrics on mount */
  useEffect(() => {
    getMetrics()
      .then((m) => {
        const list = m.tickers ?? [];
        setTickers(list);
        if (list.length > 0 && !TICKERS.includes(selected)) setSelected(list[0]);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed"));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /* Fetch history whenever ticker/limit changes */
  const load = useCallback(async (ticker: string, lim: number) => {
    setLoading(true);
    setError(null);
    try {
      setHistory(await getHistory(ticker, lim));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load history");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selected) load(selected, limit);
  }, [selected, limit, load]);

  /* Chart data: reverse rows so oldest-first */
  const chartData = useMemo(() => {
    if (!history) return [];
    return [...history.rows].reverse().map((r) => ({
      date: r.date,
      close: r.close,
      return_20d: r.return_20d,
      volatility_20d: r.volatility_20d,
      drawdown_60d: r.drawdown_60d,
    }));
  }, [history]);

  /* Unique tickers = merge TICKERS constants with backend list */
  const allTickers = useMemo(() => {
    const set = new Set([...TICKERS, ...tickers]);
    return Array.from(set);
  }, [tickers]);

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

      {/* ── selectors ───────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex flex-wrap gap-2">
          <span className="text-xs text-muted-foreground self-center mr-1">Ticker</span>
          {allTickers.map((t) => (
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
        <div className="h-5 w-px bg-border" />
        <div className="flex flex-wrap gap-2">
          <span className="text-xs text-muted-foreground self-center mr-1">Rows</span>
          {LIMITS.map((l) => (
            <Button
              key={l}
              variant={l === limit ? "secondary" : "outline"}
              size="sm"
              className={`font-mono text-xs ${l === limit ? "bg-accent text-accent-foreground" : "text-muted-foreground"}`}
              onClick={() => setLimit(l)}
            >
              {l}
            </Button>
          ))}
        </div>
      </div>

      {/* ── charts ──────────────────────────────────────────────── */}
      {loading ? (
        <div className="grid gap-4 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <CardHeader className="pb-2"><Skeleton className="h-4 w-24" /></CardHeader>
              <CardContent><Skeleton className="h-48 w-full" /></CardContent>
            </Card>
          ))}
        </div>
      ) : chartData.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2">
          <MiniChart data={chartData} dataKey="close" color={CHART_COLORS.close} title="Close Price" />
          <MiniChart data={chartData} dataKey="return_20d" color={CHART_COLORS.return_20d} title="Return (20d)" />
          <MiniChart data={chartData} dataKey="volatility_20d" color={CHART_COLORS.volatility_20d} title="Volatility (20d)" />
          <MiniChart data={chartData} dataKey="drawdown_60d" color={CHART_COLORS.drawdown_60d} title="Drawdown (60d)" />
        </div>
      ) : (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            No chart data available for this ticker.
          </CardContent>
        </Card>
      )}

      {/* ── data table ──────────────────────────────────────────── */}
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
