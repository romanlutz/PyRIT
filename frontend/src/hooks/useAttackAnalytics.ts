import { useEffect, useEffectEvent, useMemo, useRef, useState } from 'react'

import { analyticsApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type { AttackAnalyticsQuery, AttackAnalyticsReport, AttackAnalyticsResults } from '@/types'
import { ANALYTICS_DEBOUNCE_MS, ANALYTICS_PAGE_SIZE } from '@/utils/attackAnalytics'

interface ReportRequestIdentity {
  readonly scopeKey: string
  readonly key: string
}

interface ReportState {
  readonly request: ReportRequestIdentity
  readonly report: AttackAnalyticsReport | null
  readonly groupOffset: number
  readonly error: string | null
}

interface ResultsState {
  readonly request: ReportRequestIdentity
  readonly results: AttackAnalyticsResults
  readonly page: number
  readonly requestedPage: number
  readonly cursor: string | null
  readonly loading: boolean
  readonly error: string | null
}

/**
 * Owns report and result-page reads, never aggregation. A null query disables reads
 * (including for invalid shared URLs). Equal serialized payloads are one request;
 * Reload explicitly starts another generation. There is no polling.
 *
 * Keep a failed same-scope refresh's report and timestamps intact. Filters or axes
 * changing invalidate it immediately, so old results cannot describe a new cohort.
 */
export function useAttackAnalytics(query: AttackAnalyticsQuery | null) {
  const [refreshVersion, setRefreshVersion] = useState(0)
  const [settled, setSettled] = useState<ReportState | null>(null)
  const [resultState, setResultState] = useState<ResultsState | null>(null)
  const resultController = useRef<AbortController | null>(null)
  // Group pages share a report scope, but each page/refresh is a distinct request.
  const scopeKey = JSON.stringify(query && { ...query, group_offset: undefined, group_limit: undefined })
  const requestKey = JSON.stringify([query, refreshVersion])
  // Equal consecutive keys reuse one identity. A -> B -> A creates a new one:
  // revisiting A must not revive a cancelled/older A request or its result page.
  const request = useMemo(() => ({ scopeKey, key: requestKey }), [scopeKey, requestKey])
  const report = settled?.request.scopeKey === scopeKey ? settled.report : null
  const loading = query !== null && settled?.request !== request
  const error = settled?.request === request ? settled.error : null
  // A cancelled page belongs to the previous request even if a failed refresh
  // retains the very same report object. Never revive that page's loading state.
  const currentResults = resultState?.request === request && !loading ? resultState : null
  const results = currentResults?.results ?? report?.results ?? null
  const page = currentResults?.page ?? 0
  const captureQuery = useEffectEvent((): AttackAnalyticsQuery | null => query)

  useEffect(() => {
    // The serialized request key owns reactivity, not the caller's object identity.
    // Capture its typed payload once, before debouncing; equal new objects must not
    // restart this effect or cancel an independently loading results page.
    const payload = captureQuery()
    if (payload === null) return
    const controller = new AbortController()
    resultController.current?.abort()
    const timer = setTimeout(() => {
      analyticsApi.query(payload, controller.signal)
        .then((response: AttackAnalyticsReport) => {
          if (controller.signal.aborted) return
          setSettled({ request, report: response, groupOffset: payload.group_offset ?? 0, error: null })
        })
        .catch((reason: unknown) => {
          if (controller.signal.aborted) return
          setSettled((previous: ReportState | null) => ({
            request,
            report: previous?.request.scopeKey === request.scopeKey ? previous.report : null,
            groupOffset: previous?.request.scopeKey === request.scopeKey ? previous.groupOffset : 0,
            error: toApiError(reason).detail,
          }))
        })
    }, ANALYTICS_DEBOUNCE_MS)
    return () => {
      clearTimeout(timer)
      controller.abort()
      resultController.current?.abort()
    }
  }, [request])

  function reload(): void {
    resultController.current?.abort()
    setRefreshVersion((version: number) => version + 1)
  }

  /** Read only rows for the displayed report's filters, leaving its aggregates and timestamp untouched. */
  async function loadResults(cursor: string | null, requestedPage: number): Promise<void> {
    if (!report || !results || loading) return
    resultController.current?.abort()
    const controller = new AbortController()
    resultController.current = controller
    // Keep the displayed page until success; retry uses the requested cursor/page,
    // not the displayed page's next cursor (which could be a different request).
    const requestState: ResultsState = {
      request, results, page, requestedPage, cursor, loading: true, error: null,
    }
    setResultState(requestState)
    try {
      const response = await analyticsApi.results(
        { filters: report.filters, cursor, limit: ANALYTICS_PAGE_SIZE },
        controller.signal,
      )
      if (controller.signal.aborted) return
      setResultState({ ...requestState, results: response, page: requestedPage, loading: false })
    } catch (reason: unknown) {
      if (controller.signal.aborted) return
      setResultState({ ...requestState, loading: false, error: toApiError(reason).detail })
    }
  }

  return {
    report,
    reportGroupOffset: settled?.request.scopeKey === scopeKey ? settled.groupOffset : 0,
    loading,
    error,
    refreshVersion,
    reload,
    results,
    page,
    resultsLoading: currentResults?.loading ?? false,
    resultsError: currentResults?.error ?? null,
    nextPage: (): void => {
      if (results?.has_more && results.next_cursor && !currentResults?.loading) {
        void loadResults(results.next_cursor, page + 1)
      }
    },
    firstPage: (): void => { void loadResults(null, 0) },
    retryResults: (): void => {
      if (currentResults) void loadResults(currentResults.cursor, currentResults.requestedPage)
    },
  }
}
