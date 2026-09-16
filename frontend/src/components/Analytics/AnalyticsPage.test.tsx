import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation, useNavigate } from 'react-router'

import { analyticsApi } from '@/services/api'
import {
  ANALYTICS_EMPTY_STATISTICS, ANALYTICS_OPERATION_FILTER, ANALYTICS_PAGE_TIME,
  ANALYTICS_TEST_STATISTICS, ANALYTICS_TEST_TIME,
  makeAnalyticsFacets, makeAnalyticsReport, makeAnalyticsResults, makeAnalyticsRow,
} from '@/test-utils/analyticsFixtures'
import type { AttackAnalyticsFilter, AttackAnalyticsQuery, AttackAnalyticsReport, AttackAnalyticsViewState } from '@/types'
import {
  ANALYTICS_ASR_NOTE, DEFAULT_ANALYTICS_VIEW, analyticsViewToSearchParams, formatAnalyticsTime,
} from '@/utils/attackAnalytics'

import AnalyticsPage from './AnalyticsPage'

jest.mock('@/services/api', () => ({
  analyticsApi: { query: jest.fn(), results: jest.fn(), facets: jest.fn() },
}))

const mockQuery = jest.mocked(analyticsApi.query)
const mockResults = jest.mocked(analyticsApi.results)
const mockFacets = jest.mocked(analyticsApi.facets)

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
)

function NavigationProbe() {
  const navigate = useNavigate()
  const location = useLocation()
  return <><button onClick={() => { void navigate(-1) }}>Browser back</button><output aria-label="Current URL">{location.search}</output></>
}

function renderPage(view?: AttackAnalyticsViewState, onOpenAttack: (id: string) => void = jest.fn()) {
  const path = view ? `/analytics?${analyticsViewToSearchParams(view)}` : '/analytics'
  return render(
    <TestWrapper><MemoryRouter initialEntries={[path]}>
      <AnalyticsPage onOpenAttack={onOpenAttack} /><NavigationProbe />
    </MemoryRouter></TestWrapper>,
  )
}

function deferred<T>() {
  let resolve: ((value: T) => void) | undefined
  let reject: ((reason: unknown) => void) | undefined
  const promise = new Promise<T>((onResolve: (value: T | PromiseLike<T>) => void, onReject: (reason: unknown) => void) => {
    resolve = onResolve
    reject = onReject
  })
  return {
    promise,
    resolve: (value: T): void => { if (resolve) resolve(value) },
    reject: (reason: unknown): void => { if (reject) reject(reason) },
  }
}

function echoQuery(query: AttackAnalyticsQuery): AttackAnalyticsReport {
  return makeAnalyticsReport({
    filters: query.filters ?? { dimensions: [], outcomes: [] },
    group_by: query.group_by ?? { name: 'operation' },
    compare_by: query.compare_by ?? null,
  })
}

describe('AnalyticsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockQuery.mockReset().mockImplementation(async (query: AttackAnalyticsQuery) => echoQuery(query))
    mockResults.mockReset().mockResolvedValue(makeAnalyticsResults())
    mockFacets.mockReset().mockResolvedValue(makeAnalyticsFacets())
  })

  afterEach(() => { jest.useRealTimers() })

  it('should load all saved results once, use server statistics and reuse the included first page', async () => {
    mockQuery.mockResolvedValue(makeAnalyticsReport({
      summary: { ...ANALYTICS_TEST_STATISTICS, success_rate: 0.625, decided_share: 0.375 },
    }))
    const onOpenAttack = jest.fn()
    const user = userEvent.setup()
    renderPage(undefined, onOpenAttack)
    expect(screen.getByRole('status', { name: 'Loading analytics for current filters' })).toBeInTheDocument()
    expect(await screen.findByText('Inspect a saved response')).toBeInTheDocument()
    const summary = screen.getByRole('region', { name: 'Outcome summary' })
    expect(within(summary).getByText('62.5%')).toBeInTheDocument()
    expect(within(summary).getByText('37.5%')).toBeInTheDocument()
    expect(within(summary).getByText('4 success / 6 decided')).toBeInTheDocument()
    expect(within(summary).getByText('10 total')).toBeInTheDocument()
    expect(mockQuery).toHaveBeenCalledTimes(1)
    expect(mockQuery).toHaveBeenCalledWith(expect.objectContaining({
      filters: { dimensions: [], outcomes: [] },
      group_by: { name: 'operation' }, compare_by: null, result_limit: 25,
    }), expect.any(AbortSignal))
    expect(mockResults).not.toHaveBeenCalled()
    expect(mockFacets).not.toHaveBeenCalled()
    await user.click(screen.getByRole('row', { name: 'Open result result-1' }))
    expect(onOpenAttack).toHaveBeenCalledWith('result-1')
  })

  it('should paginate only results, keep report freshness, and reset the table on Reload', async () => {
    const user = userEvent.setup()
    mockQuery.mockResolvedValue(makeAnalyticsReport({
      results: makeAnalyticsResults({ has_more: true, next_cursor: 'opaque-next' }),
    }))
    mockResults.mockResolvedValue(makeAnalyticsResults({
      items: [makeAnalyticsRow({ attack_result_id: 'result-2', objective_preview: 'Second saved result' })],
      computed_at: ANALYTICS_PAGE_TIME,
    }))
    renderPage()
    await screen.findByText('Inspect a saved response')
    const refreshed = screen.getByText(/Last refreshed:/).textContent
    const results = screen.getByRole('region', { name: 'Matching AttackResults' })
    await user.click(within(results).getByRole('button', { name: 'Next' }))
    expect(await screen.findByText('Second saved result')).toBeInTheDocument()
    expect(within(results).getByText('Page 2')).toBeInTheDocument()
    expect(screen.getByText(/Last refreshed:/).textContent).toBe(refreshed)
    expect(within(results).getByText(formatAnalyticsTime(ANALYTICS_PAGE_TIME))).toBeInTheDocument()
    expect(mockResults).toHaveBeenCalledWith({
      filters: { dimensions: [], outcomes: [] }, cursor: 'opaque-next', limit: 25,
    }, expect.any(AbortSignal))
    expect(mockQuery).toHaveBeenCalledTimes(1)
    expect(mockFacets).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: 'Reload' }))
    expect(await screen.findByText('Inspect a saved response')).toBeInTheDocument()
    await waitFor(() => { expect(mockQuery).toHaveBeenCalledTimes(2) })
    expect(within(results).getByText('Page 1')).toBeInTheDocument()
    expect(mockResults).toHaveBeenCalledTimes(1)
  })

  it('should keep a failed Reload visibly stale with its original timestamp and support Retry', async () => {
    const user = userEvent.setup()
    mockQuery.mockResolvedValueOnce(makeAnalyticsReport())
      .mockRejectedValueOnce(new Error('Analytics is busy'))
      .mockResolvedValueOnce(makeAnalyticsReport({ computed_at: ANALYTICS_PAGE_TIME }))
    renderPage()
    await screen.findByText('Inspect a saved response')
    await user.click(screen.getByRole('button', { name: 'Reload' }))
    expect(await screen.findByText(/Reload failed.*data may be stale/)).toBeInTheDocument()
    expect(screen.getByText('Inspect a saved response')).toBeInTheDocument()
    expect(screen.getByText(/Last refreshed:/)).toHaveTextContent(formatAnalyticsTime(ANALYTICS_TEST_TIME))
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => { expect(screen.getByText(/Last refreshed:/)).toHaveTextContent(formatAnalyticsTime(ANALYTICS_PAGE_TIME)) })
    expect(screen.queryByText(/data may be stale/)).not.toBeInTheDocument()
  })

  it('should never show old results under changed filters, including when the new cohort fails', async () => {
    const user = userEvent.setup()
    const pending = deferred<AttackAnalyticsReport>()
    mockQuery.mockResolvedValueOnce(makeAnalyticsReport()).mockReturnValueOnce(pending.promise)
    renderPage()
    await screen.findByText('Inspect a saved response')
    await user.click(screen.getByRole('button', { name: 'Filter to error: 1' }))
    expect(screen.queryByText('Inspect a saved response')).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Outcome summary' })).not.toBeInTheDocument()
    await waitFor(() => { expect(mockQuery).toHaveBeenCalledTimes(2) })
    await act(async () => { pending.reject(new Error('Cannot read this cohort')) })
    expect(await screen.findByText(/Could not load analytics for these filters/)).toBeInTheDocument()
    expect(screen.queryByText('Inspect a saved response')).not.toBeInTheDocument()
  })

  it('should abort superseded reports and ignore a late successful response', async () => {
    const user = userEvent.setup()
    const pending = deferred<AttackAnalyticsReport>()
    mockQuery.mockReturnValueOnce(pending.promise).mockResolvedValueOnce(makeAnalyticsReport({
      group_by: { name: 'model' },
      results: makeAnalyticsResults({ items: [makeAnalyticsRow({ objective_preview: 'Latest cohort' })] }),
    }))
    renderPage()
    await waitFor(() => { expect(mockQuery).toHaveBeenCalledTimes(1) })
    const signal = mockQuery.mock.calls[0][1]
    await user.selectOptions(screen.getByLabelText('Group by'), 'model')
    expect(await screen.findByText('Latest cohort')).toBeInTheDocument()
    expect(signal?.aborted).toBe(true)
    await act(async () => { pending.resolve(makeAnalyticsReport()) })
    expect(screen.getByText('Latest cohort')).toBeInTheDocument()
    expect(screen.queryByText('Inspect a saved response')).not.toBeInTheDocument()
  })

  it('should abort an in-flight results page on Reload so it cannot replace the refreshed first page', async () => {
    const user = userEvent.setup()
    const pending = deferred<ReturnType<typeof makeAnalyticsResults>>()
    mockQuery.mockResolvedValue(makeAnalyticsReport({
      results: makeAnalyticsResults({ has_more: true, next_cursor: 'page-2' }),
    }))
    mockResults.mockReturnValueOnce(pending.promise)
    renderPage()
    await screen.findByText('Inspect a saved response')
    await user.click(within(screen.getByRole('region', { name: 'Matching AttackResults' })).getByRole('button', { name: 'Next' }))
    const signal = mockResults.mock.calls[0][1]
    await user.click(screen.getByRole('button', { name: 'Reload' }))
    await waitFor(() => { expect(mockQuery).toHaveBeenCalledTimes(2) })
    expect(signal?.aborted).toBe(true)
    await act(async () => {
      pending.resolve(makeAnalyticsResults({ items: [makeAnalyticsRow({ objective_preview: 'Obsolete page' })] }))
    })
    expect(screen.queryByText('Obsolete page')).not.toBeInTheDocument()
    expect(screen.getByText('Page 1')).toBeInTheDocument()
  })

  it('should preserve report and current rows on a results-only error, then retry that cursor', async () => {
    const user = userEvent.setup()
    mockQuery.mockResolvedValue(makeAnalyticsReport({
      results: makeAnalyticsResults({ has_more: true, next_cursor: 'retry-cursor' }),
    }))
    mockResults.mockRejectedValueOnce(new Error('Page deadline exceeded')).mockResolvedValueOnce(makeAnalyticsResults())
    renderPage()
    await screen.findByText('Inspect a saved response')
    await user.click(within(screen.getByRole('region', { name: 'Matching AttackResults' })).getByRole('button', { name: 'Next' }))
    expect(await screen.findByText(/Still showing page 1/)).toBeInTheDocument()
    expect(screen.getByText('Inspect a saved response')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Retry results' }))
    await screen.findByText('Page 2')
    expect(mockResults.mock.calls[1][0].cursor).toBe('retry-cursor')
    expect(mockQuery).toHaveBeenCalledTimes(1)
  })

  it('should apply typed group predicates in addition to existing converter, outcome and date filters', async () => {
    const user = userEvent.setup()
    const converters: AttackAnalyticsFilter = {
      dimension: { name: 'converter_type', converter_direction: 'response' },
      values: [{ kind: 'value', value: 'A' }, { kind: 'value', value: 'B' }],
      match_mode: 'any',
    }
    const added: AttackAnalyticsFilter = { ...converters, values: [{ kind: 'missing', value: null }] }
    const view: AttackAnalyticsViewState = {
      ...DEFAULT_ANALYTICS_VIEW,
      groupBy: converters.dimension,
      filters: {
        dimensions: [ANALYTICS_OPERATION_FILTER, converters], outcomes: ['success', 'failure'],
        updated_after: '2026-01-01T00:00:00Z', updated_before: '2026-10-01T00:00:00Z',
      },
    }
    mockQuery.mockImplementation(async (query: AttackAnalyticsQuery) => makeAnalyticsReport({
      ...echoQuery(query), outcome_filter_applied: true, groups_overlap: true,
      groups: [{ key: { kind: 'missing', value: null }, label: 'Unknown', statistics: ANALYTICS_TEST_STATISTICS, drilldown_filters: [added] }],
    }))
    renderPage(view)
    await user.click(await screen.findByRole('button', { name: 'Filter to Unknown (missing metadata)' }))
    await waitFor(() => {
      expect(mockQuery).toHaveBeenLastCalledWith(expect.objectContaining({
        filters: { ...view.filters, dimensions: [...view.filters.dimensions, added] },
      }), expect.any(AbortSignal))
    })
    expect(screen.getByText(/Groups overlap/)).toBeInTheDocument()
  })

  it('should restore filter and chart choices with browser Back', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole('button', { name: 'Filter to success: 4' }))
    await user.selectOptions(screen.getByLabelText('Chart view'), 'success-rate')
    expect(screen.getByLabelText('Chart view')).toHaveValue('success-rate')
    await user.click(screen.getByRole('button', { name: 'Browser back' }))
    expect(screen.getByLabelText('Chart view')).toHaveValue('outcomes')
    expect(screen.getByRole('button', { name: 'Remove outcome success' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Browser back' }))
    expect(screen.queryByRole('button', { name: 'Remove outcome success' })).not.toBeInTheDocument()
    await waitFor(() => { expect(mockQuery).toHaveBeenLastCalledWith(expect.objectContaining({ filters: { dimensions: [], outcomes: [] } }), expect.any(AbortSignal)) })
  })

  it('should invalidate a stale filter editor when browser Back restores a different predicate', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole('button', { name: 'Filter to Nightly' }))
    await user.click(screen.getByRole('button', { name: /^Operation \(ANY\): Nightly/ }))
    expect(screen.getByRole('region', { name: 'Filter editor' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Browser back' }))
    expect(screen.queryByRole('region', { name: 'Filter editor' })).not.toBeInTheDocument()
    expect(screen.getByText(/The edited filter changed/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Operation' }))
    expect(screen.queryByRole('group', { name: 'Selected filter values' })).not.toBeInTheDocument()
  })

  it('should browse every group with server offsets and not mistake a group page for a results page', async () => {
    const user = userEvent.setup()
    mockQuery.mockResolvedValue(makeAnalyticsReport({ has_more_groups: true, next_group_offset: 50 }))
    renderPage()
    await user.click(await screen.findByRole('button', { name: 'Show all groups' }))
    expect(await screen.findByText('Browse all groups, 50 at a time.')).toBeInTheDocument()
    await waitFor(() => { expect(mockQuery).toHaveBeenLastCalledWith(expect.objectContaining({ group_limit: 50, group_offset: 0 }), expect.any(AbortSignal)) })
    await waitFor(() => { expect(screen.getByRole('button', { name: 'Next groups' })).toBeEnabled() })
    await user.click(screen.getByRole('button', { name: 'Next groups' }))
    await waitFor(() => { expect(mockQuery).toHaveBeenLastCalledWith(expect.objectContaining({ group_limit: 50, group_offset: 50 }), expect.any(AbortSignal)) })
    expect(mockResults).not.toHaveBeenCalled()
  })

  it('should reset group pagination on Reload while keeping a failed refresh coherent', async () => {
    const user = userEvent.setup()
    mockQuery.mockResolvedValue(makeAnalyticsReport({ has_more_groups: true, next_group_offset: 15 }))
    renderPage()
    await user.click(await screen.findByRole('button', { name: 'Next groups' }))
    await waitFor(() => { expect(screen.getByRole('button', { name: 'First groups' })).toBeEnabled() })
    mockQuery.mockRejectedValueOnce(new Error('Reload deadline exceeded'))
    await user.click(screen.getByRole('button', { name: 'Reload' }))
    await screen.findByText(/Reload failed.*data may be stale/)
    expect(mockQuery).toHaveBeenLastCalledWith(expect.objectContaining({ group_offset: 0 }), expect.any(AbortSignal))
    expect(screen.getByText('Inspect a saved response')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'First groups' })).toBeEnabled()
  })

  it.each(['error', 'undetermined'] as const)('should show unavailable ASR* for %s-only results', async (outcome: 'error' | 'undetermined') => {
    mockQuery.mockResolvedValue(makeAnalyticsReport({
      outcome_filter_applied: true,
      summary: { ...ANALYTICS_EMPTY_STATISTICS, total_results: 2, [outcome === 'error' ? 'errors' : 'undetermined']: 2 },
    }))
    renderPage({ ...DEFAULT_ANALYTICS_VIEW, filters: { dimensions: [], outcomes: [outcome] } })
    const summary = await screen.findByRole('region', { name: 'Outcome summary' })
    expect(within(summary).getByText('Unavailable*')).toBeInTheDocument()
    expect(within(summary).queryByText('0%')).not.toBeInTheDocument()
    expect(screen.getByRole('note', { name: 'Outcome-filtered ASR' })).toHaveTextContent(ANALYTICS_ASR_NOTE)
    expect(screen.getByRole('note', { name: 'Outcome-filtered ASR' })).toBeVisible()
    expect(within(summary).getByRole('button', { name: /ASR\*:.*Outcome filter active/ })).toBeInTheDocument()
  })

  it.each([false, true])('should distinguish empty storage from a filtered empty cohort (%s)', async (filtered: boolean) => {
    mockQuery.mockResolvedValue(makeAnalyticsReport({
      summary: ANALYTICS_EMPTY_STATISTICS, groups: [], results: makeAnalyticsResults({ items: [] }),
    }))
    renderPage(filtered ? { ...DEFAULT_ANALYTICS_VIEW, filters: { dimensions: [ANALYTICS_OPERATION_FILTER], outcomes: [] } } : undefined)
    expect(await screen.findByRole('heading', { name: filtered ? 'No results match these filters' : 'No saved AttackResults yet' })).toBeInTheDocument()
  })

  it('should render backend warnings as text and refuse malformed shared URLs', async () => {
    const warning = '<img src=x onerror="alert(1)"> SQLite reads may block writes.'
    mockQuery.mockResolvedValue(makeAnalyticsReport({ warnings: [warning] }))
    const { unmount } = renderPage()
    expect(await screen.findByText(warning)).toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    unmount()
    mockQuery.mockClear()
    render(<TestWrapper><MemoryRouter initialEntries={['/analytics?analytics=not-json']}>
      <AnalyticsPage onOpenAttack={jest.fn()} />
    </MemoryRouter></TestWrapper>)
    expect(screen.getByText(/Cannot restore this analytics link/)).toBeInTheDocument()
    expect(mockQuery).not.toHaveBeenCalled()
  })
})
