import type { LucideIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface MetricCardProps {
  title: string;
  value: string;
  description?: string;
  icon: LucideIcon;
  /** Optional accent colour class applied to the icon, e.g. "text-emerald-400". */
  iconColor?: string;
}

export function MetricCard({
  title,
  value,
  description,
  icon: Icon,
  iconColor = "text-muted-foreground",
}: MetricCardProps) {
  return (
    <Card className="transition-colors hover:border-muted-foreground/30">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        <Icon className={`h-4 w-4 ${iconColor}`} />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold tracking-tight">{value}</div>
        {description && (
          <p className="mt-1 text-xs text-muted-foreground">{description}</p>
        )}
      </CardContent>
    </Card>
  );
}
