import { Badge } from "@/components/ui/badge";

/** Map regime labels to visual colour classes. */
const REGIME_COLORS: Record<string, string> = {
  bull: "bg-emerald-500/15 text-emerald-400 border-emerald-500/25",
  bear: "bg-rose-500/15 text-rose-400 border-rose-500/25",
  recovery: "bg-amber-500/15 text-amber-400 border-amber-500/25",
  crisis: "bg-red-500/15 text-red-300 border-red-500/25",
};

const DEFAULT_COLOR = "bg-zinc-500/15 text-zinc-400 border-zinc-500/25";

interface RegimeBadgeProps {
  regime: string;
  className?: string;
}

export function RegimeBadge({ regime, className = "" }: RegimeBadgeProps) {
  const colorClasses = REGIME_COLORS[regime.toLowerCase()] ?? DEFAULT_COLOR;
  return (
    <Badge
      variant="outline"
      className={`font-mono text-xs uppercase tracking-wider ${colorClasses} ${className}`}
    >
      {regime}
    </Badge>
  );
}
