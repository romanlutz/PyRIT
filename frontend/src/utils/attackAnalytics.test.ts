import type { AttackAnalyticsFilter, AttackAnalyticsViewState } from '@/types'

import {
  DEFAULT_ANALYTICS_VIEW, analyticsDimensionKey, analyticsOptionLabel,
  analyticsQuery, analyticsSuccessCountsLabel, analyticsValueKey, analyticsViewError, analyticsViewFromSearchParams,
  analyticsViewToSearchParams, appendAnalyticsDrilldown, formatAnalyticsPercent,
} from './attackAnalytics'

describe('attack analytics view state', () => {
  const converters: AttackAnalyticsFilter = {
    dimension: { name: 'converter_type', converter_direction: 'response' },
    values: [{ kind: 'value', value: 'ConverterA' }, { kind: 'value', value: 'ConverterB' }],
    match_mode: 'any',
  }

  it('should start with every saved result, grouped by operation and default heatmap dimensions', () => {
    const restored = analyticsViewFromSearchParams(new URLSearchParams())
    expect(restored).toEqual({ view: DEFAULT_ANALYTICS_VIEW, error: null })
    expect(analyticsQuery(restored.view)).toEqual({
      filters: DEFAULT_ANALYTICS_VIEW.filters,
      group_by: { name: 'operation' }, compare_by: null,
      group_limit: 15, group_offset: 0, axis_limit: 20, result_limit: 25,
    })
    expect(analyticsQuery({ ...restored.view, chart: 'heatmap' })).toMatchObject({
      group_by: { name: 'targeted_harm_category' }, compare_by: { name: 'attack_type' },
    })
  })

  it('should keep the same server query when serializing defaults or changing only presentation', () => {
    const restored = analyticsViewFromSearchParams(analyticsViewToSearchParams(DEFAULT_ANALYTICS_VIEW)).view
    expect(analyticsQuery(restored)).toEqual(analyticsQuery(DEFAULT_ANALYTICS_VIEW))
    expect(analyticsQuery({ ...restored, chart: 'success-rate' }, 15)).toEqual(analyticsQuery(restored, 15))
    const heatmap: AttackAnalyticsViewState = { ...restored, chart: 'heatmap' }
    expect(analyticsQuery({ ...heatmap, heatmapMetric: 'total_results' })).toEqual(analyticsQuery(heatmap))
  })

  it('should preserve repeated AND predicates, converter direction, typed absences and time zones in URLs', () => {
    const extra: AttackAnalyticsFilter = {
      ...converters,
      values: [{ kind: 'value', value: 'ConverterA' }],
      match_mode: 'all',
    }
    const state: AttackAnalyticsViewState = {
      ...DEFAULT_ANALYTICS_VIEW,
      chart: 'heatmap',
      heatmapMetric: 'total_results',
      groupBy: { name: 'label', label_key: 'team.region' },
      heatmapColumn: converters.dimension,
      filters: {
        dimensions: [converters, extra, {
          dimension: { name: 'operation' },
          values: [{ kind: 'missing', value: null }, { kind: 'value', value: 'Unknown' }],
          match_mode: 'any',
        }],
        outcomes: ['success', 'error'],
        updated_after: '2026-01-01T00:00:00-07:00',
        updated_before: '2026-09-14T21:00:00Z',
      },
    }
    expect(analyticsViewFromSearchParams(analyticsViewToSearchParams(state))).toEqual({ view: state, error: null })
  })

  it('should append drill-down constraints instead of replacing an existing ANY converter condition', () => {
    const filters = {
      dimensions: [converters], outcomes: ['success', 'failure'] as const,
      updated_after: '2026-01-01T00:00:00Z', updated_before: '2026-02-01T00:00:00Z',
    }
    const extra: AttackAnalyticsFilter = { ...converters, values: [{ kind: 'value', value: 'ConverterC' }] }
    const actual = appendAnalyticsDrilldown({ ...filters, outcomes: [...filters.outcomes] }, [extra])
    expect(actual).toEqual({ ...filters, dimensions: [converters, extra] })
    expect(filters.dimensions).toEqual([converters])
    expect(appendAnalyticsDrilldown(actual, [], 'success').outcomes).toEqual(['success'])
  })

  it('should keep missing, no-converter, empty-string and literal Unknown keys distinct', () => {
    expect(new Set([
      analyticsValueKey({ kind: 'missing', value: null }),
      analyticsValueKey({ kind: 'no_converters', value: null }),
      analyticsValueKey({ kind: 'value', value: '' }),
      analyticsValueKey({ kind: 'value', value: 'Unknown' }),
    ]).size).toBe(4)
    expect(analyticsOptionLabel({ key: { kind: 'missing', value: null }, label: 'Unknown' })).toBe('Unknown (missing metadata)')
    expect(analyticsOptionLabel({ key: { kind: 'value', value: 'Unknown' }, label: 'Unknown' })).toBe('Unknown')
    expect(analyticsDimensionKey({ name: 'operation' })).toBe(analyticsDimensionKey({ name: 'operation', converter_direction: 'request' }))
  })

  it('should format an unavailable rate differently from zero without deriving a metric', () => {
    expect(formatAnalyticsPercent(null)).toBe('Unavailable')
    expect(formatAnalyticsPercent(0)).toBe('0%')
    expect(formatAnalyticsPercent(0.625)).toBe('62.5%')
  })

  it('labels the numerator as success and the denominator as decided', () => {
    expect(analyticsSuccessCountsLabel({ successes: 3, total_decided: 3 })).toBe('3 success / 3 decided')
    expect(analyticsSuccessCountsLabel({ successes: 0, total_decided: 0 })).toBe('0 success / 0 decided')
  })

  it.each([
    ['local timestamp', { filters: { dimensions: [], outcomes: [], updated_after: '2026-01-01T10:00' } }],
    ['reversed range', { filters: { dimensions: [], outcomes: [], updated_after: '2026-02-01T10:00Z', updated_before: '2026-01-01T10:00Z' } }],
    ['unknown outcome', { filters: { dimensions: [], outcomes: ['unscored'] } }],
    ['unknown dimension', { groupBy: { name: 'scores' } }],
    ['reserved label key', { groupBy: { name: 'label', label_key: 'operation' } }],
    ['missing label key', { groupBy: { name: 'label' } }],
    ['invalid label key', { groupBy: { name: 'label', label_key: 'a/b' } }],
    ['direction on scalar', { groupBy: { name: 'operation', converter_direction: 'response' } }],
    ['same heatmap axes', { heatmapColumn: { name: 'targeted_harm_category' } }],
    ['unknown chart', { chart: 'time-trend' }],
    ['unknown metric', { heatmapMetric: 'average-score' }],
    ['too many filters', { filters: { outcomes: [], dimensions: Array.from({ length: 17 }, () => converters) } }],
    ['empty predicate', { filters: { outcomes: [], dimensions: [{ ...converters, values: [] }] } }],
    ['all matching on a scalar', { filters: { outcomes: [], dimensions: [{ ...converters, dimension: { name: 'operation' }, match_mode: 'all' }] } }],
    ['no-converters on a scalar', { filters: { outcomes: [], dimensions: [{ dimension: { name: 'operation' }, values: [{ kind: 'no_converters', value: null }] }] } }],
    ['untyped value', { filters: { outcomes: [], dimensions: [{ ...converters, values: ['Unknown'] }] } }],
  ])('should reject %s without silently loading a broader cohort', (_name: string, invalid: object) => {
    const result = analyticsViewFromSearchParams(new URLSearchParams({
      analytics: JSON.stringify({ ...DEFAULT_ANALYTICS_VIEW, ...invalid }),
    }))
    expect(result.error).toMatch(/Cannot restore/)
  })

  it('should report malformed JSON and enforce limits on UI-generated predicates', () => {
    expect(analyticsViewFromSearchParams(new URLSearchParams({ analytics: '{' })).error).toBeTruthy()
    expect(analyticsViewError({
      ...DEFAULT_ANALYTICS_VIEW,
      filters: {
        dimensions: [{ ...converters, values: Array.from({ length: 101 }, () => ({ kind: 'value', value: 'A' })) }],
        outcomes: [],
      },
    })).toMatch(/100 values/)
  })
})
