import { expect, test, type Page, type Route } from '@playwright/test'

import {
  ANALYTICS_EMPTY_STATISTICS, ANALYTICS_OPERATION_FILTER, ANALYTICS_PAGE_TIME, ANALYTICS_TEST_TIME,
  makeAnalyticsFacets, makeAnalyticsReport, makeAnalyticsResults, makeAnalyticsRow,
} from '../src/test-utils/analyticsFixtures'
import type {
  AttackAnalyticsFilter, AttackAnalyticsFilters, AttackAnalyticsStatistics, AttackOutcome,
} from '../src/types'
import {
  DEFAULT_ANALYTICS_VIEW, analyticsViewFromSearchParams, analyticsViewToSearchParams,
} from '../src/utils/attackAnalytics'

const ALL_STATISTICS: AttackAnalyticsStatistics = {
  total_results: 100, successes: 40, failures: 20, errors: 10, undetermined: 30,
  total_decided: 60, success_rate: 0.6666666667, decided_share: 0.6,
  outcome_shares: { success: 0.4, failure: 0.2, error: 0.1, undetermined: 0.3 },
}
const OUTCOME_STATISTICS: Record<AttackOutcome, AttackAnalyticsStatistics> = {
  success: { ...ANALYTICS_EMPTY_STATISTICS, total_results: 40, successes: 40, total_decided: 40, success_rate: 1, decided_share: 1, outcome_shares: { success: 1, failure: 0, error: 0, undetermined: 0 } },
  failure: { ...ANALYTICS_EMPTY_STATISTICS, total_results: 20, failures: 20, total_decided: 20, success_rate: 0, decided_share: 1, outcome_shares: { success: 0, failure: 1, error: 0, undetermined: 0 } },
  error: { ...ANALYTICS_EMPTY_STATISTICS, total_results: 10, errors: 10, decided_share: 0, outcome_shares: { success: 0, failure: 0, error: 1, undetermined: 0 } },
  undetermined: { ...ANALYTICS_EMPTY_STATISTICS, total_results: 30, undetermined: 30, decided_share: 0, outcome_shares: { success: 0, failure: 0, error: 0, undetermined: 1 } },
}
const HARM_FILTER: AttackAnalyticsFilter = {
  dimension: { name: 'targeted_harm_category' },
  values: [{ kind: 'value', value: 'category-A' }],
  match_mode: 'any',
}
const ATTACK_TYPE_FILTER: AttackAnalyticsFilter = {
  dimension: { name: 'attack_type' },
  values: [{ kind: 'value', value: 'PromptSendingAttack' }],
  match_mode: 'any',
}

function deferred() {
  let release: (() => void) | undefined
  const promise = new Promise<void>((resolve: () => void) => { release = resolve })
  return { promise, release: (): void => { if (release) release() } }
}

function rgbChannels(color: string): number[] {
  const channels = color.match(/[\d.]+/g)?.slice(0, 3).map(Number)
  if (!channels || channels.length !== 3) throw new Error(`Expected a computed RGB color, got ${color}`)
  return channels
}

function colorContrast(foreground: string, background: string): number {
  const luminance = (color: string): number => {
    const channels = rgbChannels(color).map((channel: number) => {
      const value = channel / 255
      return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
    })
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722
  }
  const first = luminance(foreground)
  const second = luminance(background)
  return (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05)
}

interface AnalyticsMocks {
  reportRequests: Array<Record<string, unknown>>
  resultRequests: Array<Record<string, unknown>>
  facetRequests: Array<Record<string, unknown>>
  unexpectedRequests: string[]
  failNextReport: boolean
  failNextResults: boolean
  blockNextReport: ReturnType<typeof deferred> | null
  computedAt: string
  completedReports: number
}

function requestBody(route: Route): Record<string, unknown> {
  const body: unknown = route.request().postDataJSON()
  if (typeof body !== 'object' || body === null || Array.isArray(body)) throw new Error('Expected an analytics query object')
  return Object.fromEntries(Object.entries(body))
}

function queryView(body: Record<string, unknown>) {
  const parsed = analyticsViewFromSearchParams(new URLSearchParams({
    analytics: JSON.stringify({
      ...DEFAULT_ANALYTICS_VIEW,
      filters: body.filters ?? DEFAULT_ANALYTICS_VIEW.filters,
      groupBy: body.group_by ?? DEFAULT_ANALYTICS_VIEW.groupBy,
      heatmapRow: body.compare_by ? body.group_by : DEFAULT_ANALYTICS_VIEW.heatmapRow,
      heatmapColumn: body.compare_by ?? DEFAULT_ANALYTICS_VIEW.heatmapColumn,
    }),
  }))
  if (parsed.error) throw new Error(parsed.error)
  return parsed.view
}

function resultPage(filters: AttackAnalyticsFilters, next: boolean) {
  const outcome = filters.outcomes.length === 1 ? filters.outcomes[0] : 'success'
  const statistics = filters.outcomes.length === 1 ? OUTCOME_STATISTICS[outcome] : ALL_STATISTICS
  return makeAnalyticsResults({
    items: Array.from({ length: Math.min(25, statistics.total_results) }, (_unused: unknown, index: number) => makeAnalyticsRow({
      attack_result_id: `result-${(next ? 25 : 0) + index + 1}`,
      objective_preview: `Saved ${outcome} result ${(next ? 25 : 0) + index + 1}`,
      outcome,
    })),
    has_more: !next && statistics.total_results > 25,
    next_cursor: !next && statistics.total_results > 25 ? 'next-results' : null,
    computed_at: next ? ANALYTICS_PAGE_TIME : ANALYTICS_TEST_TIME,
  })
}

async function mockAnalytics(page: Page): Promise<AnalyticsMocks> {
  const state: AnalyticsMocks = {
    reportRequests: [], resultRequests: [], facetRequests: [], unexpectedRequests: [],
    failNextReport: false, failNextResults: false, blockNextReport: null, computedAt: ANALYTICS_TEST_TIME, completedReports: 0,
  }
  // Catch every API request. A missing fixture must never reach a real backend.
  await page.route('**/api/**', async (route: Route) => {
    const path = new URL(route.request().url()).pathname
    let response: unknown
    if (path === '/api/auth/config') response = { clientId: '', tenantId: '', allowedGroupIds: '' }
    else if (path === '/api/auth/access') response = { isAdmin: false }
    else if (path === '/api/health') response = { status: 'healthy' }
    else if (path === '/api/version') response = { version: 'test', display: 'PyRIT test', default_labels: { operation: 'Current operation', operator: 'Current operator' } }
    else if (path === '/api/targets' || path === '/api/converters') response = { items: [], pagination: { has_more: false, next_cursor: null, limit: 200 } }
    else if (path === '/api/converters/catalog') response = { items: [] }
    else if (path === '/api/labels') response = { source: 'attacks', labels: {}, operators: ['Alice'], operations: ['Nightly'] }
    else if (path === '/api/analytics/attacks/query') {
      expect(route.request().method()).toBe('POST')
      const body = requestBody(route)
      state.reportRequests.push(body)
      const view = queryView(body)
      const computedAt = state.computedAt
      const blocker = state.blockNextReport
      state.blockNextReport = null
      if (blocker) await blocker.promise
      if (state.failNextReport) {
        state.failNextReport = false
        await route.fulfill({ status: 503, json: { detail: 'Analytics is busy. Retry this read.' } })
        return
      }
      const statistics = view.filters.outcomes.length === 1 ? OUTCOME_STATISTICS[view.filters.outcomes[0]] : ALL_STATISTICS
      response = makeAnalyticsReport({
        filters: view.filters,
        group_by: body.compare_by ? view.heatmapRow : view.groupBy,
        compare_by: body.compare_by ? view.heatmapColumn : null,
        summary: statistics,
        outcome_filter_applied: view.filters.outcomes.length > 0 && view.filters.outcomes.length < 4,
        groups_overlap: Boolean(body.compare_by),
        groups: [{
          key: { kind: 'value', value: 'Nightly' }, label: 'Nightly',
          statistics, drilldown_filters: [ANALYTICS_OPERATION_FILTER],
        }],
        rows: body.compare_by ? [{ key: { kind: 'value', value: 'category-A' }, label: 'category-A' }] : [],
        columns: body.compare_by ? [{ key: { kind: 'value', value: 'PromptSendingAttack' }, label: 'PromptSendingAttack' }] : [],
        cells: body.compare_by ? [{
          row: { kind: 'value', value: 'category-A' }, column: { kind: 'value', value: 'PromptSendingAttack' },
          statistics, drilldown_filters: [HARM_FILTER, ATTACK_TYPE_FILTER],
        }] : [],
        results: resultPage(view.filters, false),
        computed_at: computedAt,
        warnings: ['SQLite journaling may affect read performance.'],
      })
    } else if (path === '/api/analytics/attacks/results') {
      expect(route.request().method()).toBe('POST')
      const body = requestBody(route)
      state.resultRequests.push(body)
      if (state.failNextResults) {
        state.failNextResults = false
        await route.fulfill({ status: 503, json: { detail: 'Results page timed out.' } })
        return
      }
      response = resultPage(queryView(body).filters, Boolean(body.cursor))
    } else if (path === '/api/analytics/attacks/facets') {
      expect(route.request().method()).toBe('POST')
      const body = requestBody(route)
      state.facetRequests.push(body)
      response = makeAnalyticsFacets()
    } else if (/^\/api\/attacks\/result-\d+$/.test(path)) {
      expect(route.request().method()).toBe('GET')
      const id = path.split('/').pop()
      response = {
        attack_result_id: id, conversation_id: 'persisted-conversation',
        attack_type: 'PromptSendingAttack', objective: 'Saved analytics objective', outcome: 'success',
        target: { target_type: 'OpenAIChatTarget', model_name: 'Retired model', identifier_hash: 'retired-target-hash' },
        converters: [], message_count: 0, related_conversation_ids: [], labels: {},
        operation: 'Nightly', operator: 'Alice', created_at: ANALYTICS_TEST_TIME, updated_at: ANALYTICS_TEST_TIME,
      }
    } else if (/^\/api\/attacks\/result-\d+\/conversations$/.test(path)) {
      response = { attack_result_id: 'result-1', main_conversation_id: 'persisted-conversation', conversations: [] }
    } else if (/^\/api\/attacks\/result-\d+\/messages$/.test(path)) {
      expect(route.request().method()).toBe('GET')
      response = { conversation_id: 'persisted-conversation', messages: [] }
    } else {
      state.unexpectedRequests.push(`${route.request().method()} ${path}`)
      await route.fulfill({ status: 501, json: { detail: `Unmocked request: ${path}` } })
      return
    }
    await route.fulfill({ status: 200, json: response })
    if (path === '/api/analytics/attacks/query') state.completedReports += 1
  })
  return state
}

test.describe('Saved AttackResult analytics', () => {
  for (const theme of ['light', 'dark']) {
    test(`uses matching accessible outcome colors in ${theme} badges, icons, bars and swatches`, async ({ page }) => {
      await mockAnalytics(page)
      await page.addInitScript((mode: string) => { localStorage.setItem('pyrit.themeMode', mode) }, theme)
      await page.goto('/analytics')
      const summary = page.getByRole('region', { name: 'Outcome summary' })
      const markerColors: Partial<Record<AttackOutcome, string>> = {}
      for (const outcome of ['success', 'failure', 'error', 'undetermined'] as const) {
        const badge = summary.getByText(outcome, { exact: true })
        await expect(badge).toBeVisible()
        const colors = await badge.evaluate((element: HTMLElement) => ({
          color: getComputedStyle(element).color,
          background: getComputedStyle(element).backgroundColor,
          icon: getComputedStyle(element.querySelector('svg') ?? element).color,
        }))
        markerColors[outcome] = colors.color
        const bar = page.getByRole('button', { name: new RegExp(`Nightly: ${outcome} segment;`) })
        const swatch = page.getByRole('button', { name: new RegExp(`Nightly: \\d+ ${outcome};`) })
          .locator('[aria-hidden="true"]').first()
        expect(await bar.evaluate((element: HTMLElement) => getComputedStyle(element).backgroundColor)).toBe(colors.color)
        expect(await swatch.evaluate((element: HTMLElement) => getComputedStyle(element).backgroundColor)).toBe(colors.color)
        expect(colors.icon).toBe(colors.color)
        expect(colorContrast(colors.color, colors.background)).toBeGreaterThanOrEqual(4.5)
        await bar.hover()
        expect(await bar.evaluate((element: HTMLElement) => getComputedStyle(element).backgroundColor)).toBe(colors.color)
      }
      const blue = rgbChannels(markerColors.error ?? '')
      const red = rgbChannels(markerColors.failure ?? '')
      expect(blue[2]).toBeGreaterThan(blue[0])
      expect(blue[2]).toBeGreaterThan(blue[1])
      expect(red[0]).toBeGreaterThan(red[1])
      expect(red[0]).toBeGreaterThan(red[2])
      for (const outcome of ['failure', 'error'] as const) {
        await summary.getByRole('button', { name: new RegExp(`Filter to ${outcome}:`) }).click()
        const badge = page.getByRole('table', { name: 'Saved attack results' }).getByText(outcome, { exact: true }).first()
        await expect(badge).toBeVisible()
        const colors = await badge.evaluate((element: HTMLElement) => ({
          color: getComputedStyle(element).color,
          background: getComputedStyle(element).backgroundColor,
          icon: getComputedStyle(element.querySelector('svg') ?? element).color,
        }))
        expect(colors.background).toBe(markerColors[outcome])
        expect(colors.icon).toBe(colors.color)
        expect(colorContrast(colors.color, colors.background)).toBeGreaterThanOrEqual(4.5)
      }
    })
  }

  test('drills into the actual outcome segment and success-rate bar', async ({ page }) => {
    const mocks = await mockAnalytics(page)
    await page.goto('/analytics')
    await page.getByRole('button', { name: 'Nightly: success segment; 40 results; filter dashboard' }).click()
    await expect.poll(() => mocks.reportRequests.at(-1)?.filters).toMatchObject({
      dimensions: [ANALYTICS_OPERATION_FILTER], outcomes: ['success'],
    })
    await expect(page.getByRole('region', { name: 'Outcome summary' }).getByText('100%*', { exact: true })).toBeVisible()
    await page.getByLabel('Chart view', { exact: true }).selectOption('success-rate')
    const bar = page.getByRole('button', { name: /Inspect Nightly success rate: 40 results/ })
    await bar.focus()
    await page.keyboard.press('Enter')
    await expect.poll(() => mocks.reportRequests.at(-1)?.filters).toMatchObject({
      dimensions: [ANALYTICS_OPERATION_FILTER, ANALYTICS_OPERATION_FILTER], outcomes: ['success'],
    })
    expect(mocks.unexpectedRequests).toEqual([])
  })

  test('filters, drills into a keyboard heatmap cell, and opens the existing guarded attack route', async ({ page }) => {
    const mocks = await mockAnalytics(page)
    await page.goto('/analytics')
    await expect(page.getByRole('heading', { name: 'Analytics', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Open result result-1', exact: true })).toBeVisible()
    expect(mocks.reportRequests).toHaveLength(1)
    expect(mocks.facetRequests).toHaveLength(0)
    expect(mocks.resultRequests).toHaveLength(0)
    expect(mocks.reportRequests[0].filters).toEqual({ dimensions: [], outcomes: [] })
    await page.getByRole('button', { name: 'Operation', exact: true }).click()
    await page.getByRole('checkbox', { name: 'Nightly', exact: true }).check()
    await page.getByRole('button', { name: 'Apply filter' }).click()
    await expect.poll(() => mocks.reportRequests.length).toBe(2)
    await page.getByLabel('Chart view', { exact: true }).selectOption('heatmap')
    await expect(page.getByRole('region', { name: 'Heatmap', exact: true })).toBeVisible()
    expect(mocks.reportRequests.at(-1)).toMatchObject({
      filters: { dimensions: [ANALYTICS_OPERATION_FILTER] },
      group_by: { name: 'targeted_harm_category' }, compare_by: { name: 'attack_type' },
    })
    const cell = page.getByRole('button', { name: /category-A \/ PromptSendingAttack: 100 results/ })
    await cell.focus()
    await page.keyboard.press('Enter')
    await expect.poll(() => mocks.reportRequests.at(-1)?.filters).toMatchObject({
      dimensions: [ANALYTICS_OPERATION_FILTER, HARM_FILTER, ATTACK_TYPE_FILTER],
    })
    await expect(page.getByRole('button', { name: 'Open result result-1', exact: true })).toBeVisible()
    const analyticsUrl = page.url()
    await page.getByRole('button', { name: 'Open result result-1', exact: true }).click()
    await expect(page).toHaveURL(/\/attacks\/result-1$/)
    await expect(page.getByText('Saved analytics objective', { exact: true })).toBeVisible()
    await expect(page.getByText(/read.only/i).first()).toBeVisible()
    await page.getByRole('navigation', { name: 'Primary' }).getByRole('button', { name: 'Analytics' }).click()
    await expect(page).toHaveURL(analyticsUrl)
    await expect(page.getByRole('region', { name: 'Heatmap', exact: true })).toBeVisible()
    expect(mocks.unexpectedRequests).toEqual([])
  })

  test('marks ASR everywhere for a success-only cohort and keeps error-only ASR unavailable', async ({ page }) => {
    await mockAnalytics(page)
    await page.goto('/analytics')
    await page.getByRole('button', { name: 'Filter to success: 40', exact: true }).click()
    const summary = page.getByRole('region', { name: 'Outcome summary' })
    await expect(summary.getByText('100%*', { exact: true })).toBeVisible()
    await expect(page.getByRole('note', { name: 'Outcome-filtered ASR' })).toHaveText(/may be 100%/)
    await summary.getByRole('button', { name: /ASR\*:.*Outcome filter active/ }).focus()
    await expect(page.getByRole('tooltip').filter({ hasText: 'Outcome filter active' })).toBeVisible()
    await page.getByLabel('Chart view', { exact: true }).selectOption('success-rate')
    await expect(page.getByRole('region', { name: 'Success rate* by Operation' })).toBeVisible()
    await page.getByLabel('Chart view', { exact: true }).selectOption('heatmap')
    await expect(page.getByLabel('Heatmap legend').getByRole('button', { name: /ASR\*/ })).toBeVisible()
    await page.getByRole('button', { name: 'Filter to error: 0', exact: true }).click()
    await expect(summary.getByText('Unavailable*', { exact: true })).toBeVisible()
    await expect(page.getByRole('region', { name: 'Heatmap', exact: true }).getByText('Unavailable*', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: 'Clear all filters' }).click()
    await expect(summary.getByText('66.7%', { exact: true })).toBeVisible()
    await expect(page.getByRole('note', { name: 'Outcome-filtered ASR' })).toHaveCount(0)
  })

  test('isolates result pagination, retries failed pages, and leaves report freshness unchanged', async ({ page }) => {
    const mocks = await mockAnalytics(page)
    await page.goto('/analytics')
    const results = page.getByRole('region', { name: 'Matching AttackResults' })
    await expect(results.getByText('Page 1', { exact: true })).toBeVisible()
    const refreshed = await page.getByText(/Last refreshed:/).innerText()
    mocks.failNextResults = true
    await results.getByRole('button', { name: 'Next', exact: true }).click()
    await expect(results.getByText(/Still showing page 1/)).toBeVisible()
    await results.getByRole('button', { name: 'Retry results' }).click()
    await expect(results.getByText('Page 2', { exact: true })).toBeVisible()
    await expect(results.getByRole('button', { name: 'Open result result-26', exact: true })).toBeVisible()
    expect(mocks.reportRequests).toHaveLength(1)
    expect(mocks.facetRequests).toHaveLength(0)
    expect(mocks.resultRequests).toHaveLength(2)
    expect(mocks.resultRequests[0]).toEqual(mocks.resultRequests[1])
    expect(await page.getByText(/Last refreshed:/).innerText()).toBe(refreshed)
    await expect(results.locator('time[datetime="2026-09-14T12:05:00Z"]')).toBeVisible()
    await page.getByRole('button', { name: 'Reload', exact: true }).click()
    await expect(results.getByText('Page 1', { exact: true })).toBeVisible()
    await expect.poll(() => mocks.reportRequests.length).toBe(2)
    expect(mocks.resultRequests).toHaveLength(2)
    expect(mocks.unexpectedRequests).toEqual([])
  })

  test('preserves a failed Reload and ignores a superseded refresh after filters change', async ({ page }) => {
    const mocks = await mockAnalytics(page)
    await page.goto('/analytics')
    await expect(page.getByRole('region', { name: 'Outcome summary' })).toBeVisible()
    const refreshed = await page.getByText(/Last refreshed:/).innerText()
    mocks.failNextReport = true
    await page.getByRole('button', { name: 'Reload', exact: true }).click()
    await expect(page.getByText(/Reload failed.*data may be stale/)).toBeVisible()
    expect(await page.getByText(/Last refreshed:/).innerText()).toBe(refreshed)
    await expect(page.getByRole('button', { name: 'Open result result-1', exact: true })).toBeVisible()
    mocks.computedAt = ANALYTICS_PAGE_TIME
    await page.getByRole('button', { name: 'Retry', exact: true }).click()
    await expect(page.getByText(/data may be stale/)).toHaveCount(0)
    await expect(page.locator('header time[datetime="2026-09-14T12:05:00Z"]')).toBeVisible()
    const blocked = deferred()
    mocks.blockNextReport = blocked
    await page.getByRole('button', { name: 'Reload', exact: true }).click()
    await expect.poll(() => mocks.reportRequests.length).toBe(4)
    await page.getByRole('button', { name: 'Filter to success: 40', exact: true }).click()
    await expect(page.getByRole('region', { name: 'Outcome summary' }).getByText('100%*', { exact: true })).toBeVisible()
    const completedReports = mocks.completedReports
    blocked.release()
    await expect.poll(() => mocks.completedReports).toBe(completedReports + 1)
    await expect(page.getByRole('note', { name: 'Outcome-filtered ASR' })).toBeVisible()
    await expect(page.getByRole('region', { name: 'Outcome summary' }).getByText('100%*', { exact: true })).toBeVisible()
    expect(mocks.unexpectedRequests).toEqual([])
  })

  test('restores URL predicates, response direction, last-updated bounds and browser Back', async ({ page }) => {
    const mocks = await mockAnalytics(page)
    const converter: AttackAnalyticsFilter = {
      dimension: { name: 'converter_type', converter_direction: 'response' },
      values: [{ kind: 'value', value: 'A' }, { kind: 'value', value: 'B' }],
      match_mode: 'any',
    }
    const filters: AttackAnalyticsFilters = {
      dimensions: [converter, { ...converter, values: [{ kind: 'missing', value: null }] }],
      outcomes: ['success'], updated_after: '2026-01-01T00:00:00-07:00', updated_before: '2026-09-15T00:00:00Z',
    }
    const path = `/analytics?${analyticsViewToSearchParams({ ...DEFAULT_ANALYTICS_VIEW, groupBy: converter.dimension, filters })}`
    await page.goto(path)
    await expect(page.getByLabel('Group by', { exact: true })).toHaveValue('response_converters')
    await expect.poll(() => mocks.reportRequests.at(-1)?.filters).toEqual(filters)
    await page.getByLabel('Group by', { exact: true }).selectOption('model')
    await expect(page.getByLabel('Group by', { exact: true })).toHaveValue('model')
    await page.goBack()
    await expect(page.getByLabel('Group by', { exact: true })).toHaveValue('response_converters')
    await page.reload()
    await expect.poll(() => mocks.reportRequests.at(-1)?.filters).toEqual(filters)
    await page.getByRole('button', { name: 'Last updated', exact: true }).click()
    await page.getByLabel('Last updated after', { exact: true }).fill('2026-02-01T09:00')
    await page.getByRole('button', { name: 'Apply range' }).click()
    await expect.poll(() => mocks.reportRequests.at(-1)?.filters).toMatchObject({
      dimensions: filters.dimensions, outcomes: ['success'], updated_after: expect.stringMatching(/Z$/),
    })
  })

  for (const theme of ['light', 'dark', 'high-contrast'] as const) {
    test(`keeps the operational chart readable in ${theme}`, async ({ page }, testInfo) => {
      const mocks = await mockAnalytics(page)
      await page.emulateMedia({ colorScheme: theme === 'dark' ? 'dark' : 'light', forcedColors: theme === 'high-contrast' ? 'active' : 'none' })
      await page.goto('/analytics')
      await expect(page.getByRole('region', { name: 'Outcome breakdown by Operation' })).toBeVisible()
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme)
      await page.screenshot({ path: testInfo.outputPath(`analytics-${theme}.png`) })
      expect(mocks.unexpectedRequests).toEqual([])
    })
  }

  test.describe('narrow touch viewport', () => {
    test.use({ viewport: { width: 390, height: 844 }, hasTouch: true })
    test('reflows filters and keeps heatmap controls keyboard and touch accessible', async ({ page }, testInfo) => {
      await mockAnalytics(page)
      await page.goto('/analytics')
      await page.getByLabel('Chart view', { exact: true }).selectOption('heatmap')
      const cell = page.getByRole('button', { name: /category-A \/ PromptSendingAttack: 100 results/ })
      await expect(cell).toBeVisible()
      const box = await cell.boundingBox()
      expect(box?.height).toBeGreaterThanOrEqual(44)
      const dimensions = await page.evaluate(() => ({
        width: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth,
      }))
      expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.width)
      await page.screenshot({ path: testInfo.outputPath('analytics-mobile.png') })
    })
  })
})
