import { useState, useEffect, useCallback } from "react";
import type { ReactNode } from "react";
import {
  LayoutDashboard,
  Layers,
  Clock,
  BarChart3,
  FileText,
  FlaskConical,
  Wifi,
  WifiOff,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { API_BASE_URL, getHealth } from "@/lib/api";
import type { PageId } from "@/lib/types";

interface NavItem {
  id: PageId;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "regimes", label: "Current Regimes", icon: Layers },
  { id: "history", label: "History", icon: Clock },
  { id: "metrics", label: "Metrics", icon: BarChart3 },
  { id: "reports", label: "Reports", icon: FileText },
];

interface DashboardLayoutProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  children: ReactNode;
}

export function DashboardLayout({
  activePage,
  onNavigate,
  children,
}: DashboardLayoutProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      {/* ─── Sidebar ─── */}
      <aside
        className={`flex flex-col border-r border-border bg-card transition-[width] duration-200 ${
          collapsed ? "w-16" : "w-60"
        }`}
      >
        {/* Brand */}
        <div className="flex h-14 items-center gap-2.5 px-4">
          <FlaskConical className="h-5 w-5 shrink-0 text-primary" />
          {!collapsed && (
            <span className="text-sm font-semibold tracking-tight">
              RegimeLab
            </span>
          )}
        </div>

        <Separator />

        {/* Nav */}
        <nav className="flex-1 space-y-1 px-2 py-3">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
            const active = activePage === id;
            return (
              <Button
                key={id}
                variant={active ? "secondary" : "ghost"}
                className={`w-full justify-start gap-3 ${
                  collapsed ? "px-3" : "px-3"
                } ${
                  active
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
                onClick={() => onNavigate(id)}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {!collapsed && (
                  <span className="truncate text-sm">{label}</span>
                )}
              </Button>
            );
          })}
        </nav>

        <Separator />

        {/* Collapse toggle */}
        <div className="flex items-center justify-center py-3">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-muted-foreground"
            onClick={() => setCollapsed((c) => !c)}
          >
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>
      </aside>

      {/* ─── Main Area ─── */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header bar */}
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-card px-6">
          <div className="flex items-center gap-3">
            <h1 className="text-sm font-semibold tracking-tight">
              {NAV_ITEMS.find((n) => n.id === activePage)?.label ?? "Dashboard"}
            </h1>
          </div>

          <div className="flex items-center gap-4">
            {/* API base indicator */}
            <span className="hidden text-xs text-muted-foreground sm:inline-block font-mono">
              {API_BASE_URL}
            </span>

            {/* Status placeholder */}
            <StatusPill />
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}

/* ─── Inline API status indicator ─── */

function StatusPill() {
  const [connected, setConnected] = useState(false);

  const check = useCallback(() => {
    getHealth()
      .then(() => setConnected(true))
      .catch(() => setConnected(false));
  }, []);

  useEffect(() => {
    check();
    const id = setInterval(check, 10_000);
    return () => clearInterval(id);
  }, [check]);

  return (
    <Badge
      variant="outline"
      className={`gap-1.5 text-xs font-normal ${
        connected
          ? "border-emerald-500/30 text-emerald-400"
          : "border-muted-foreground/30 text-muted-foreground"
      }`}
    >
      {connected ? (
        <Wifi className="h-3 w-3" />
      ) : (
        <WifiOff className="h-3 w-3" />
      )}
      {connected ? "API Connected" : "API Offline"}
    </Badge>
  );
}
