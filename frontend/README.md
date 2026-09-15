# PyRIT Frontend

Modern TypeScript + React frontend for PyRIT, built with Fluent UI.

## Development

```bash
# Install dependencies
npm install

# Start both backend and frontend (cross-platform)
python dev.py start
# OR use npm script
npm run start

# Start backend only (with airt initializer by default)
python dev.py backend

# Start frontend only (backend must be started separately)
python dev.py frontend
# OR
npm run dev

# Restart both servers
python dev.py restart
# OR
npm run restart

# Stop all servers
python dev.py stop
# OR
npm run stop

# Build for production
npm run build

# Preview production build
npm run preview
```

### Backend CLI

The backend uses `pyrit_backend` CLI which supports initializers:

```bash
# Start with default airt initializer (loads targets from env vars)
pyrit_backend --initializers airt

# Start without initializers
pyrit_backend

# Start with custom initialization script
pyrit_backend --initialization-scripts ./my_targets.py

# List available initializers
pyrit_backend --list-initializers

# Custom host/port
pyrit_backend --host 127.0.0.1 --port 8080
```

**Development Mode**: The `dev.py` script sets `PYRIT_DEV_MODE=true` so the backend expects the frontend to run separately on port 3000.

**Production Mode**: When installed from PyPI, the backend serves the bundled frontend and will exit if frontend files are missing.

## Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Fluent UI v9** - Microsoft design system
- **Vite** - Fast build tool
- **Axios** - HTTP client

## Testing

```bash
# Unit & Integration Tests (Jest + React Testing Library)
npm test              # Run all tests
npm run test:watch    # Watch mode for development
npm run test:coverage # Run with coverage report (85%+ threshold)

# End-to-End Tests (Playwright)
npm run test:e2e          # Run headless (auto-starts frontend + backend via dev.py)
npm run test:e2e:headed   # Run with visible browser windows (requires display)
npm run test:e2e:ui       # Interactive UI mode (requires display)
```

### E2E Test Modes

E2E flow tests run in two modes controlled by Playwright projects and an environment variable:

- **Seeded** (`--project seeded`, default for CI): Messages are stored directly in the database with `send: false` using dummy credentials. No real API keys needed. Tests cover the full UI flow (display, branching, conversation switching, promoting) without calling any external service.

- **Live** (`--project live`, requires `E2E_LIVE_MODE=true`): Messages are sent to real OpenAI endpoints with `send: true`. Each target variant requires endpoint and model environment variables plus either an API key or an Azure endpoint accessible through the current Entra identity. Variants without a usable configuration are automatically skipped. Tests verify that real target responses render correctly.

```bash
# Seeded integration (no credentials needed)
npx playwright test --project seeded

# Live integration (uses API keys when present, otherwise Entra authentication)
E2E_LIVE_MODE=true npx playwright test --project live

# Run both
E2E_LIVE_MODE=true npx playwright test
```

The mock and seeded projects run in the **GitHub Actions** pull-request workflow. The live project is intended for a protected pipeline with an Entra identity or API keys.

E2E tests use `dev.py` to automatically start both frontend and backend servers. If servers are already running, they will be reused.

> **Note**: `test:e2e:ui` and `test:e2e:headed` require a graphical display and won't work in headless environments like devcontainers. Use `npm run test:e2e` for CI/headless testing.

## Configuration

The frontend proxies API requests to `http://localhost:8000` in development.
Configure this in `vite.config.ts` if needed.

## Saved-result analytics

Open **Analytics** (`/analytics`) to explore saved AttackResults across operations,
operators, persisted targets/models, targeted harms, attack types, request/response
converters, scenario runs, custom labels, and outcomes. It starts with all saved
results, not the current operation or active target. The optional **Last updated**
range describes edits to stored results, not attack execution dates.

Counts, outcome shares, decided share, ASR, grouping, and matching result pages
come from the analytics API. ASR is attacker successes among decided results;
errors and undetermined results are not defensive failures. An unavailable ASR
is not 0%. An outcome restriction applies to the whole dashboard and adds
**ASR\*** with an explanation, including when the rate is unavailable.

Group and heatmap selections append the server's drill-down predicates. Separate
filter chips are ANDed, even for the same converter dimension. Values within a
chip use ANY matching, or explicit ALL matching for converters. Missing metadata
and known-empty converter pipelines are distinct from literal values such as
`Unknown`. Overlapping groups must not be added together.

The active chart is an outcome composition, success-rate comparison, or bounded
heatmap. Each has a semantic data table, and heatmap cells are keyboard-operable.
**Show all groups** browses paginated groups; heatmap truncation is reported
explicitly. Custom-label values are fetched only after entering a key. Facets
are searched and paged only while their control is open.

Filters and chart settings are shareable in the URL and restore with browser
Back. **Reload** retains them, resets pagination, and refreshes the report and
opened facet. A failed reload keeps the last successful report with a stale
warning and Retry. Result pagination uses only the results endpoint and has
its own read time; it does not advance the report's **Last refreshed** time.
Opening a result uses the existing attack route and its read-only guards.

Focused frontend verification (PowerShell):

```powershell
npm run type-check
npm run lint
npm test -- --runInBand --testPathPatterns "Analytics|attackAnalytics"
$env:E2E_FRONTEND_PORT = '4177'
npm run test:e2e -- analytics.spec.ts --project=mock --workers=1
```

The analytics browser tests mock every API request and use a dedicated,
automatically stopped Vite server. They do not start attacks or access a database.
