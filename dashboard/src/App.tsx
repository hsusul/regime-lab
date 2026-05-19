import { useState } from "react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Overview } from "@/pages/Overview";
import { Regimes } from "@/pages/Regimes";
import { History } from "@/pages/History";
import { Metrics } from "@/pages/Metrics";
import { Reports } from "@/pages/Reports";
import type { PageId } from "@/lib/types";

const PAGE_COMPONENTS: Record<PageId, React.ComponentType> = {
  overview: Overview,
  regimes: Regimes,
  history: History,
  metrics: Metrics,
  reports: Reports,
};

function App() {
  const [activePage, setActivePage] = useState<PageId>("overview");
  const ActiveComponent = PAGE_COMPONENTS[activePage];

  return (
    <DashboardLayout activePage={activePage} onNavigate={setActivePage}>
      <ActiveComponent />
    </DashboardLayout>
  );
}

export default App;
