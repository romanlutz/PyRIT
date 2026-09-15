import { useEffect, useMemo, useState } from 'react'

import { analyticsApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type { AttackAnalyticsDimension, AttackAnalyticsFacets, AttackAnalyticsFilters } from '@/types'
import { ANALYTICS_DEBOUNCE_MS } from '@/utils/attackAnalytics'

interface FacetState {
  readonly key: string
  readonly data: AttackAnalyticsFacets | null
  readonly error: string | null
}

/** Mounted only for the opened control; closing it discards its unselected options. */
export function useAttackAnalyticsFacet(
  dimension: AttackAnalyticsDimension,
  filters: AttackAnalyticsFilters,
  refreshVersion: number,
) {
  const [search, setSearch] = useState('')
  const [paging, setPaging] = useState({ scope: '', offset: 0 })
  const [retryVersion, setRetryVersion] = useState(0)
  const [settled, setSettled] = useState<FacetState | null>(null)
  const scope = JSON.stringify([dimension, filters, search, refreshVersion])
  const offset = paging.scope === scope ? paging.offset : 0
  const query = useMemo(
    () => ({ dimension, filters, search, offset, limit: 50 }),
    [dimension, filters, search, offset],
  )
  const key = JSON.stringify([query, refreshVersion, retryVersion])
  const current = settled?.key === key ? settled : null

  useEffect(() => {
    const controller = new AbortController()
    const timer = setTimeout(() => {
      analyticsApi.facets(query, controller.signal)
        .then((data: AttackAnalyticsFacets) => {
          if (!controller.signal.aborted) setSettled({ key, data, error: null })
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted) setSettled({ key, data: null, error: toApiError(reason).detail })
        })
    }, ANALYTICS_DEBOUNCE_MS)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [query, key])

  return {
    search,
    offset,
    items: current?.data?.items ?? [],
    loading: current === null,
    error: current?.error ?? null,
    hasMore: current?.data?.has_more ?? false,
    setSearch: (value: string): void => { setSearch(value) },
    firstPage: (): void => { setPaging({ scope, offset: 0 }) },
    nextPage: (): void => {
      if (current?.data?.has_more && current.data.next_offset !== null) setPaging({ scope, offset: current.data.next_offset })
    },
    retry: (): void => { setRetryVersion((version: number) => version + 1) },
  }
}
