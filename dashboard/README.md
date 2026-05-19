# RegimeLab Dashboard

React + TypeScript + Vite dashboard for the RegimeLab ML research platform.

## Tech Stack

- **React 19** + **TypeScript**
- **Vite** dev server & build
- **Tailwind CSS v4** (via `@tailwindcss/vite`)
- **shadcn/ui** component library (radix-nova style, dark neutral theme)
- **lucide-react** icons

## Quick Start

```bash
cd dashboard
npm install
npm run dev
```

The dev server starts at `http://localhost:5173` by default.

## Environment

Copy the example env file and adjust if needed:

```bash
cp .env.example .env
```

| Variable             | Default                    | Description                    |
| -------------------- | -------------------------- | ------------------------------ |
| `VITE_API_BASE_URL`  | `http://127.0.0.1:8000`   | FastAPI backend base URL       |

## Build

```bash
npm run build
```

Production output is written to `dist/`.

## Project Structure

```
src/
├── lib/
│   ├── api.ts          # API client (getHealth, getMetrics, etc.)
│   ├── types.ts        # TypeScript types mirroring backend schemas
│   └── utils.ts        # cn() utility for Tailwind class merging
├── components/
│   ├── ui/             # shadcn/ui primitives (Card, Badge, Button, …)
│   ├── layout/
│   │   └── DashboardLayout.tsx
│   └── dashboard/
│       ├── MetricCard.tsx
│       └── RegimeBadge.tsx
├── pages/
│   ├── Overview.tsx
│   ├── Regimes.tsx
│   ├── History.tsx
│   ├── Metrics.tsx
│   └── Reports.tsx
├── App.tsx
├── main.tsx
└── index.css
```

## Notes

- **No router** — page switching uses local React state for now.
- **No live API calls** — pages display polished placeholders until API wiring is added in a future milestone.
- This is an **educational / research tool**, not financial advice.
