import { useEffect, useRef, useState } from 'react'

import { analyticsApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type { AttackAnalyticsQuery, AttackAnalyticsReport, AttackAnalyticsResults } from '@/types'
import { ANALYTICS_DEBOUNCE_MS, ANALYTICS_PAGE_SIZE } from '@/utils/attackAnalytics'

interface ReportState {
  readonly queryKey: string
  readonly requestKey: string
  readonly report: AttackAnalyticsReport | null
  readonly groupOffset: number
  readonly error: string | null
}

interface ResultsState {
  readonly report: AttackAnalyticsReport
  readonly results: AttackAnalyticsResults
  readonly page: number
  readonly requestedPage: number
  readonly cursor: string | null
  readonly loading: boolean
  readonly error: string | null
}

export function useAttackAnalytics(query: AttackAnalyticsQuery | null) {
  const [refreshVersion, setRefreshVersion] = useState(0)
  const [settled, setSettled] = useState<ReportState | null>(null)
  const [resultState, setResultState] = useState<ResultsState | null>(null)
  const resultController = useRef<AbortController | null>(null)
  // Group pagination does not change the cohort. Keep the last coherent report
  // during those reads, but never reuse it under different filters or axes.
  const queryKey = JSON.stringify(query && { ...query, group_offset: undefined, group_limit: undefined })
  const requestKey = JSON.stringify([query, refreshVersion])
  const report = settled?.queryKey === queryKey ? settled.report : null
  const loading = query !== null && settled?.requestKey !== requestKey
  const error = settled?.requestKey === requestKey ? settled.error : null
  const currentResults = resultState?.report === report && !loading ? resultState : null
  const results = currentResults?.results ?? report?.results ?? null
  const page = currentResults?.page ?? 0

  useEffect(() => {
    if (query === null) return
    const controller = new AbortController()
    resultController.current?.abort()
    const timer = setTimeout(() => {
      analyticsApi.query(query, controller.signal)
        .then((response: AttackAnalyticsReport) => {
          if (controller.signal.aborted) return
          setSettled({ queryKey, requestKey, report: response, groupOffset: query.group_offset ?? 0, error: null })
        })
        .catch((reason: unknown) => {
          if (controller.signal.aborted) return
          setSettled((previous: ReportState | null) => ({
            queryKey,
            requestKey,
            report: previous?.queryKey === queryKey ? previous.report : null,
            groupOffset: previous?.queryKey === queryKey ? previous.groupOffset : 0,
            error: toApiError(reason).detail,
          }))
        })
    }, ANALYTICS_DEBOUNCE_MS)
    return () => {
      clearTimeout(timer)
      controller.abort()
      resultController.current?.abort()
    }
  }, [query, queryKey, requestKey])

  function reload(): void {
    resultController.current?.abort()
    setResultState(null)
    setRefreshVersion((version: number) => version + 1)
  }

  async function loadResults(cursor: string | null, requestedPage: number): Promise<void> {
    if (!report || !results || loading) return
    resultController.current?.abort()
    const controller = new AbortController()
    resultController.current = controller
    const requestState: ResultsState = {
      report, results, page, requestedPage, cursor, loading: true, error: null,
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
    reportGroupOffset: settled?.queryKey === queryKey ? settled.groupOffset : 0,
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
