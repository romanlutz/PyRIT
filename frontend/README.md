# PyRIT Frontend

Modern TypeScript + React frontend for PyRIT, built with Fluent UI.

## Appearance

The **Theme** menu at the bottom of the sidebar offers System, Light, Dark,
Raccoon, Jimothy, Pirate, Seattle Rain, Evergreen, Blueprint, and Night Sky.
Each named preset combines a fixed palette with a decorative workspace
background. Content panels remain solid for readability.

Your choice is saved in this browser. System follows the operating system's
light/dark preference. High-contrast mode overrides every palette and hides
decorations without forgetting the selected preset.

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

## Chat converters

Chat keeps one ordered converter pipeline per input modality in memory. Closing
the converter panel does not clear these pipelines. Sending a message clears
its conversion results, but keeps the pipelines for the next message.
Use the arrow keys on a stage's reorder button to move it. Focus stays on that
stage, including when the same converter occurs more than once.

**Convert** processes each input piece separately, including multiple attachments
of the same type. **Add converted value** replaces the applied selection with the
current successful results. Failed pieces remain unconverted and show an error.
Changing an input or its pipeline clears the affected results and selections;
late responses cannot restore them.

Send uses the applied pieces' exact message indexes and runs their configured
converters on the backend. A nondeterministic converter can produce a different
value at Send than the value shown in the converter panel.

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

Jest's shared setup in `src/setupTests.ts` supplies the minimal layout signals
Fluent UI needs for dialog focus. No per-suite layout mocks are needed. Hidden
and detached elements remain excluded. Await role queries after dialog
transitions, including when returning to background controls. This is not a
layout engine; use Playwright for assertions about element dimensions or positioning.

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
heatmap. Bar charts show labeled counts without a duplicate aggregate table.
**First groups** and **Next groups** browse all groups, 15 at a time. The heatmap
uses a semantic table with keyboard-operable cells and reports truncated axes.
Custom-label values are fetched only after entering a key. Facets are searched
and paged only while their control is open. The SDK ignores that dimension's own
predicates when finding facet alternatives, but keeps the other filters. This
does not remove any applied filter from the dashboard.

Filters and chart settings are shareable in the URL and restore with browser
Back. Result cursors, group-page offsets, and unfinished filter edits are local
state, not part of the link. Changing heatmap color or switching between outcome
and success-rate bars only changes presentation: it does not request data or
reset the current pages. Changing filters or active axes requests a new report.

**Reload** retains the view, resets pagination, and refreshes the report and
opened facet. A failed reload keeps the last successful report with a stale
warning and Retry. Result pagination uses only the results endpoint and has
its own read time; it does not advance the report's **Last refreshed** time.
Opening a result uses the existing attack route and its read-only guards.

### Analytics implementation boundaries

| Module | Responsibility |
| --- | --- |
| `src/utils/attackAnalytics.ts` | Validate/serialize URL state, preserve typed identities, project active settings onto the API query, and format SDK values. |
| `src/components/Analytics/AnalyticsPage.tsx` | Coordinate validated URL edits, local group pagination, drill-down, and the report hook. |
| `src/hooks/useAttackAnalytics.ts` | Debounce/cancel report reads, retain coherent stale reports, and manage independent cursor-based result reads. |
| `src/hooks/useAttackAnalyticsFacet.ts` | Read only the opened dimension, debounce search, reset its offset for a new scope, and retry the same failed page. |
| `src/components/Analytics/AnalyticsFilters.tsx` | Keep selections/date input as drafts until Apply, preserve selections across facet pages, and guard against editing a chip replaced by browser Back. |
| `AnalyticsStats`, `AnalyticsGroups`, `AnalyticsHeatmap`, `AnalyticsResultsTable` | Render server-provided values and emit user actions; do not aggregate results or infer drill-down predicates. |
| `src/services/api.ts` and `src/types/index.ts` | Keep the transport and shared wire contracts separate from browser state. |
| `src/styles/outcomePalette.ts` and `OutcomeBadge` | Keep outcome identities and fills consistent across charts, History, Home, and Scanner without replacing Fluent theme tokens. |

Two identities serve different purposes in the report hook. The **report scope**
includes filters and axes but excludes group pagination, allowing the previous
coherent report to remain visible during a group read or failed Reload. The
**request key** includes the full server query and refresh generation. That one
key drives fetching, loading/error state, and result-page ownership. A new
JavaScript object with the same serialized payload is not a new request.
The hook keeps a stable identity for consecutive equal keys; returning to an
earlier query after another query creates a new identity rather than reviving
that query's cancelled request or old results page.
Each effect captures its typed payload once before debouncing; cancelled requests
cannot publish late responses. Failed reads never invent a newer timestamp.

Do not merge repeated dimension predicates or identify options by display label.
Dimension identity includes a custom-label key or converter direction, and value
identity includes its kind. Malformed links render a reset action and issue no
analytics request: silently using defaults would broaden the requested cohort.

Focused frontend verification (PowerShell):

```powershell
npm run type-check
npm run lint
npm test -- --runInBand --testPathPatterns "Analytics|attackAnalytics|OutcomeBadge|HistoryPagination|api.test"
$env:E2E_FRONTEND_PORT = '4177'
npm run test:e2e -- analytics.spec.ts --project=mock --workers=1
```

The analytics browser tests mock every API request and use a dedicated,
automatically stopped Vite server. They do not start attacks or access a database.

## Adding a theme preset

The catalog in `src/themes/themePresets.ts` is the source of truth for preset
IDs, labels, palettes, backgrounds, menu entries, and stored-value validation.

1. Draw a new, self-contained SVG in `public/backgrounds/`. Use a transparent
   background and keep prominent artwork away from the upper-left reading area.
   Do not embed scripts, external resources, fonts, or raster images.
2. Add one entry to `THEME_PRESETS`, using a unique, stable ID. For example:

   ```ts
   'my-background': {
     label: 'My Background',
     resolved: 'light',
     theme: webLightTheme,
     background: {
       imageUrl: '/backgrounds/my-background.svg',
       opacity: 0.08,
     },
   },
   ```

3. For a coordinated palette, follow a nearby preset's `createPaletteTheme`
   definition instead of changing colors in individual components. Keep
   `resolved` consistent with the palette's light/dark base. Its status
   foregrounds cover custom surfaces while preserving Fluent's semantic
   backgrounds and borders.
4. Document how the artwork was made and keep the palette accessibility tests
   passing. They check neutral/status text and button contrast, including the
   strongest possible artwork at the configured opacity, plus semantic
   foreground/background pairs used by badges and messages.

No hook, menu switch, or page-specific background needs to be added for a new
preset. Existing page canvases share one decorative layer; controls, dialogs,
cards, tables, and message bubbles continue using opaque Fluent UI tokens.
An unknown or removed stored preset returns to System.

### Background artwork provenance

All seven SVGs in `public/backgrounds/` were newly drawn from scratch for this
change with Copilot assistance and are provided under this repository's MIT
license. No artist's illustration, photograph, or stock wallpaper was copied,
traced, vectorized, or used as image-generation input.

The Jimothy drawing uses the real Seattle raccoon's distinctive compact,
rounded appearance. [Know Your Meme](https://knowyourmeme.com/memes/jimothy-the-raccoon)
and [Wikipedia](https://en.wikipedia.org/wiki/Jimothy_(Raccoon)) were consulted
for factual descriptions only. Their displayed artwork and photographs were
not reused. The existing CoPyRIT logo is unchanged.
