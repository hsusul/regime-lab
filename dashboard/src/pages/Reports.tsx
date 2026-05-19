import { useState, useEffect, useMemo } from "react";
import {
  Activity, TrendingUp, GitBranch, Scale, Lightbulb, AlertCircle, Info,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
} from "recharts";
import {
  getHmmReport, getForwardReturnsReport, getWalkForwardReport,
  getModelComparisonReport, getExplainReport,
} from "@/lib/api";
import type {
  HmmReport, ForwardReturnReport, WalkForwardReport,
  ModelComparisonReport, ExplainReport,
} from "@/lib/types";

/* ── shared tooltip ────────────────────────────────────────────────── */
function ChartTooltip({ active, payload, label }: {
  active?: boolean; payload?: Array<{ value?: number; name?: string; fill?: string; stroke?: string }>; label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-border bg-card px-3 py-2 text-xs shadow-lg">
      <p className="font-mono text-muted-foreground mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} className="font-mono" style={{ color: p.fill || p.stroke }}>
          {p.name}: {p.value != null ? p.value.toFixed(4) : "—"}
        </p>
      ))}
    </div>
  );
}

/* ── "not generated" banner ────────────────────────────────────────── */
function NotGenerated({ name }: { name: string }) {
  return (
    <Alert className="border-amber-500/20 bg-amber-500/5">
      <Info className="h-4 w-4 text-amber-400" />
      <AlertTitle className="text-amber-300">Not Available</AlertTitle>
      <AlertDescription className="text-amber-300/70">
        The <span className="font-mono">{name}</span> CLI report has not been generated yet.
        Run the corresponding pipeline command to populate this section.
      </AlertDescription>
    </Alert>
  );
}

/* ── section skeleton ──────────────────────────────────────────────── */
function SectionSkeleton() {
  return <Card><CardHeader><Skeleton className="h-4 w-40" /></CardHeader><CardContent><Skeleton className="h-48 w-full" /></CardContent></Card>;
}

const fmt = (n: number | null | undefined, d = 4) => n != null ? n.toFixed(d) : "—";
const pct = (n: number | null | undefined) => n != null ? `${(n * 100).toFixed(1)}%` : "—";

/* ── main component ────────────────────────────────────────────────── */
export function Reports() {
  const [hmm, setHmm] = useState<HmmReport | null | undefined>(undefined);
  const [fwd, setFwd] = useState<ForwardReturnReport | null | undefined>(undefined);
  const [wf, setWf] = useState<WalkForwardReport | null | undefined>(undefined);
  const [mc, setMc] = useState<ModelComparisonReport | null | undefined>(undefined);
  const [expl, setExpl] = useState<ExplainReport | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const safe = <T,>(fn: () => Promise<T | null>, set: (v: T | null) => void) =>
      fn().then(d => { if (!cancelled) set(d); }).catch(e => { if (!cancelled) { set(null); setError(prev => prev ?? (e instanceof Error ? e.message : "Failed")); }});
    safe(getHmmReport, setHmm);
    safe(getForwardReturnsReport, setFwd);
    safe(getWalkForwardReport, setWf);
    safe(getModelComparisonReport, setMc);
    safe(getExplainReport, setExpl);
    return () => { cancelled = true; };
  }, []);

  const loading = hmm === undefined || fwd === undefined || wf === undefined;

  /* Walk-forward chart data */
  const wfChartData = useMemo(() => {
    if (!wf) return [];
    return wf.folds.map(f => ({
      name: `Fold ${f.fold_index}`,
      accuracy: f.accuracy,
      macro_f1: f.macro_f1,
    }));
  }, [wf]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">Analysis Reports</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Latest generated analysis reports from the CLI pipeline. Run pipeline commands to populate missing sections.
        </p>
      </div>

      {error && (
        <Alert className="border-red-500/20 bg-red-500/5">
          <AlertCircle className="h-4 w-4 text-red-400" />
          <AlertTitle className="text-red-300">API Error</AlertTitle>
          <AlertDescription className="text-red-300/70">{error}</AlertDescription>
        </Alert>
      )}

      {/* ── 1. HMM State Summary ──────────────────────────────────── */}
      {loading ? <SectionSkeleton /> : hmm === null ? <NotGenerated name="HMM" /> : (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Activity className="h-4 w-4 text-violet-400" /> HMM State Summary
              <Badge variant="outline" className="ml-auto text-xs font-mono text-muted-foreground">{hmm.n_states} states · {hmm.rows.toLocaleString()} rows</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs font-mono">
                <thead>
                  <tr className="border-b border-border">
                    {["State", "Regime", "Count", "Pct", "Avg Ret 1d", "Avg Ret 20d", "Avg Vol 20d", "Avg DD 60d"].map(h => (
                      <th key={h} className="px-3 py-2 text-left text-muted-foreground font-normal whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.values(hmm.state_summary).map(s => (
                    <tr key={s.state} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                      <td className="px-3 py-2 text-center">{s.state}</td>
                      <td className="px-3 py-2"><Badge variant="outline" className="text-[10px]">{s.interpreted_regime}</Badge></td>
                      <td className="px-3 py-2 text-right">{s.frequency.toLocaleString()}</td>
                      <td className="px-3 py-2 text-right">{pct(s.pct)}</td>
                      <td className="px-3 py-2 text-right" style={{ color: s.average_return_1d >= 0 ? "#34d399" : "#f87171" }}>{fmt(s.average_return_1d)}</td>
                      <td className="px-3 py-2 text-right" style={{ color: s.average_return_20d >= 0 ? "#34d399" : "#f87171" }}>{fmt(s.average_return_20d)}</td>
                      <td className="px-3 py-2 text-right">{fmt(s.average_volatility_20d)}</td>
                      <td className="px-3 py-2 text-right text-red-400">{fmt(s.average_drawdown_60d)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-[10px] text-muted-foreground/60">
              Tickers: {hmm.tickers.join(", ")} · {hmm.date_range.start_date} → {hmm.date_range.end_date} · {hmm.created_at}
            </p>
          </CardContent>
        </Card>
      )}

      {/* ── 2. Forward Returns ────────────────────────────────────── */}
      {loading ? <SectionSkeleton /> : fwd === null ? <NotGenerated name="Forward Returns" /> : (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <TrendingUp className="h-4 w-4 text-emerald-400" /> Forward Return Summary by Regime
              <Badge variant="outline" className="ml-auto text-xs font-mono text-muted-foreground">Horizons: {fwd.horizons.join(", ")}d</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs font-mono">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-3 py-2 text-left text-muted-foreground font-normal">Regime</th>
                    <th className="px-3 py-2 text-right text-muted-foreground font-normal">Count</th>
                    {fwd.horizons.map(h => (
                      <th key={h} colSpan={2} className="px-3 py-2 text-center text-muted-foreground font-normal border-l border-border/40">{h}d Avg / Vol</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(fwd.summaries.combined.rule_label).map(([label, data]) => (
                    <tr key={label} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                      <td className="px-3 py-2"><Badge variant="outline" className="text-[10px]">{label}</Badge></td>
                      <td className="px-3 py-2 text-right">{data.observation_count.toLocaleString()}</td>
                      {fwd.horizons.map(h => {
                        const hs = data.horizons[String(h)];
                        return hs ? (
                          <>
                            <td key={`${h}-r`} className="px-3 py-2 text-right border-l border-border/40" style={{ color: hs.average_forward_return >= 0 ? "#34d399" : "#f87171" }}>
                              {fmt(hs.average_forward_return)}
                            </td>
                            <td key={`${h}-v`} className="px-3 py-2 text-right text-muted-foreground">{fmt(hs.forward_volatility)}</td>
                          </>
                        ) : <td key={h} colSpan={2} className="px-3 py-2 text-center border-l border-border/40">—</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-[10px] text-muted-foreground/60">
              {fwd.rows.toLocaleString()} rows · {fwd.date_range.start_date} → {fwd.date_range.end_date}
            </p>
          </CardContent>
        </Card>
      )}

      {/* ── 3. Walk-Forward ───────────────────────────────────────── */}
      {loading ? <SectionSkeleton /> : wf === null ? <NotGenerated name="Walk-Forward Validation" /> : (
        <>
          {/* summary cards */}
          <div className="grid gap-4 sm:grid-cols-4">
            <Card><CardHeader className="pb-1"><CardTitle className="text-xs text-muted-foreground font-normal">Folds</CardTitle></CardHeader><CardContent><p className="text-2xl font-semibold tabular-nums">{wf.summary.fold_count}</p></CardContent></Card>
            <Card><CardHeader className="pb-1"><CardTitle className="text-xs text-muted-foreground font-normal">Mean Accuracy</CardTitle></CardHeader><CardContent><p className="text-2xl font-semibold tabular-nums text-sky-400">{pct(wf.summary.mean_accuracy)}</p></CardContent></Card>
            <Card><CardHeader className="pb-1"><CardTitle className="text-xs text-muted-foreground font-normal">Mean Macro F1</CardTitle></CardHeader><CardContent><p className="text-2xl font-semibold tabular-nums text-emerald-400">{fmt(wf.summary.mean_macro_f1, 4)}</p></CardContent></Card>
            <Card><CardHeader className="pb-1"><CardTitle className="text-xs text-muted-foreground font-normal">F1 Range</CardTitle></CardHeader><CardContent><p className="text-2xl font-semibold tabular-nums text-amber-400">{fmt(wf.summary.min_macro_f1, 3)}–{fmt(wf.summary.max_macro_f1, 3)}</p></CardContent></Card>
          </div>

          {/* chart + table side by side */}
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader><CardTitle className="flex items-center gap-2 text-sm font-medium"><GitBranch className="h-4 w-4 text-sky-400" /> Fold Performance</CardTitle></CardHeader>
              <CardContent className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={wfChartData} margin={{ top: 4, right: 16, bottom: 0, left: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
                    <XAxis dataKey="name" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickLine={false} axisLine={false} />
                    <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickLine={false} axisLine={false} domain={[0.9, 1]} width={36} />
                    <Tooltip content={<ChartTooltip />} cursor={{ fill: "hsl(var(--accent))", opacity: 0.15 }} />
                    <Bar dataKey="accuracy" name="Accuracy" fill="#60a5fa" fillOpacity={0.7} radius={[4, 4, 0, 0]} animationDuration={600} />
                    <Bar dataKey="macro_f1" name="Macro F1" fill="#34d399" fillOpacity={0.7} radius={[4, 4, 0, 0]} animationDuration={600} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle className="text-sm font-medium">Fold Details</CardTitle></CardHeader>
              <CardContent>
                <div className="overflow-x-auto max-h-56 overflow-y-auto">
                  <table className="w-full border-collapse text-xs font-mono">
                    <thead className="sticky top-0 bg-card">
                      <tr className="border-b border-border">
                        {["Fold", "Train", "Test", "Rows", "Acc", "F1"].map(h => (
                          <th key={h} className="px-2 py-1.5 text-left text-muted-foreground font-normal whitespace-nowrap">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {wf.folds.map(f => (
                        <tr key={f.fold_index} className="border-b border-border/50 hover:bg-muted/30">
                          <td className="px-2 py-1.5">{f.fold_index}</td>
                          <td className="px-2 py-1.5 text-muted-foreground">{f.train_start.slice(0,7)}→{f.train_end.slice(0,7)}</td>
                          <td className="px-2 py-1.5 text-muted-foreground">{f.test_start.slice(0,7)}→{f.test_end.slice(0,7)}</td>
                          <td className="px-2 py-1.5 text-right">{f.test_rows}</td>
                          <td className="px-2 py-1.5 text-right text-sky-400">{pct(f.accuracy)}</td>
                          <td className="px-2 py-1.5 text-right text-emerald-400">{fmt(f.macro_f1, 4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>

          <p className="text-[10px] text-muted-foreground/60">
            Model: {wf.model_type} · {wf.tickers.join(", ")} · {wf.created_at}
          </p>
        </>
      )}

      {/* ── 4. Model Comparison ───────────────────────────────────── */}
      {mc === undefined ? <SectionSkeleton /> : mc === null ? <NotGenerated name="Model Comparison" /> : (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Scale className="h-4 w-4 text-amber-400" /> Model Comparison
              <Badge variant="outline" className="ml-auto text-xs font-mono text-emerald-400 border-emerald-500/25">Best: {mc.best_model_type}</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs font-mono">
                <thead>
                  <tr className="border-b border-border">
                    {Object.keys(mc.models[0] ?? {}).map(k => (
                      <th key={k} className="px-3 py-2 text-left text-muted-foreground font-normal whitespace-nowrap">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {mc.models.map((m, i) => (
                    <tr key={i} className="border-b border-border/50 hover:bg-muted/30">
                      {Object.values(m).map((v, j) => (
                        <td key={j} className="px-3 py-2 whitespace-nowrap">
                          {typeof v === "number" ? v.toFixed(4) : String(v ?? "—")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── 5. Explainability ─────────────────────────────────────── */}
      {expl === undefined ? <SectionSkeleton /> : expl === null ? <NotGenerated name="Explainability" /> : (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Lightbulb className="h-4 w-4 text-yellow-400" /> Model Explanation
              <Badge variant="outline" className="ml-auto text-xs font-mono text-muted-foreground">{expl.model_type} · {expl.experiment_id}</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="h-72">
            {expl.feature_importance?.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={expl.feature_importance.map(f => ({ name: String(f.feature ?? f.name ?? ""), importance: Number(f.importance ?? f.value ?? 0) })).sort((a, b) => b.importance - a.importance).slice(0, 15)}
                  layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 4 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickLine={false} axisLine={false} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} tickLine={false} axisLine={false} width={110} />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: "hsl(var(--accent))", opacity: 0.15 }} />
                  <Bar dataKey="importance" name="Importance" fill="#facc15" fillOpacity={0.7} radius={[0, 4, 4, 0]} animationDuration={600} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="flex h-full items-center justify-center text-sm text-muted-foreground">No feature importance data in explanation report.</p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
