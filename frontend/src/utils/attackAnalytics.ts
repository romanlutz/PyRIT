import type {
  AttackAnalyticsDimension,
  AttackAnalyticsFilter,
  AttackAnalyticsFilters,
  AttackAnalyticsOption,
  AttackAnalyticsQuery,
  AttackAnalyticsStatistics,
  AttackAnalyticsValue,
  AttackAnalyticsViewState,
  AttackOutcome,
} from '@/types'

export const ANALYTICS_PAGE_SIZE = 25
export const ANALYTICS_GROUP_LIMIT = 15
export const ANALYTICS_GROUP_PAGE_SIZE = 50
export const ANALYTICS_DEBOUNCE_MS = 250
export const ANALYTICS_MAX_PREDICATES = 16
export const ANALYTICS_MAX_VALUES = 100
export const ANALYTICS_OUTCOMES: AttackOutcome[] = ['success', 'failure', 'error', 'undetermined']

export const ANALYTICS_ASR_NOTE =
  '* Outcome filter active. ASR uses only the selected outcomes and may be 100% when only successes are selected.'
export const ANALYTICS_ASR_DEFINITION =
  'Attack success rate is attacker successes among decided results. Errors and undetermined results are excluded.'

export const DEFAULT_ANALYTICS_VIEW: AttackAnalyticsViewState = {
  filters: { dimensions: [], outcomes: [] },
  groupBy: { name: 'operation' },
  heatmapRow: { name: 'targeted_harm_category' },
  heatmapColumn: { name: 'attack_type' },
  chart: 'outcomes',
  heatmapMetric: 'success_rate',
}

export const ANALYTICS_DIMENSIONS: Array<{ value: string; label: string; dimension: AttackAnalyticsDimension }> = [
  { value: 'operation', label: 'Operation', dimension: { name: 'operation' } },
  { value: 'operator', label: 'Operator', dimension: { name: 'operator' } },
  { value: 'objective_target', label: 'Objective target', dimension: { name: 'objective_target' } },
  { value: 'model', label: 'Model', dimension: { name: 'model' } },
  { value: 'targeted_harm_category', label: 'Targeted harm category', dimension: { name: 'targeted_harm_category' } },
  { value: 'attack_type', label: 'Attack type', dimension: { name: 'attack_type' } },
  { value: 'request_converters', label: 'Request converter', dimension: { name: 'converter_type', converter_direction: 'request' } },
  { value: 'response_converters', label: 'Response converter', dimension: { name: 'converter_type', converter_direction: 'response' } },
  { value: 'scenario', label: 'Scenario run', dimension: { name: 'scenario' } },
]

const PERCENT_FORMAT = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 })
const COUNT_FORMAT = new Intl.NumberFormat()
const LABEL_KEY_PATTERN = /^[A-Za-z0-9_.-]{1,128}$/
const AWARE_TIMESTAMP_PATTERN = /T.*(?:Z|[+-]\d{2}:\d{2})$/i

export function analyticsDimensionKey(dimension: AttackAnalyticsDimension): string {
  return JSON.stringify([dimension.name, dimension.label_key ?? null, dimension.converter_direction ?? 'request'])
}

export function analyticsValueKey(value: AttackAnalyticsValue): string {
  return JSON.stringify([value.kind, value.value])
}

export function analyticsDimensionLabel(dimension: AttackAnalyticsDimension): string {
  if (dimension.name === 'label') return `Label: ${dimension.label_key}`
  return ANALYTICS_DIMENSIONS.find(
    (option: typeof ANALYTICS_DIMENSIONS[number]) =>
      analyticsDimensionKey(option.dimension) === analyticsDimensionKey(dimension),
  )?.label ?? dimension.name
}

export function analyticsValueLabel(value: AttackAnalyticsValue): string {
  if (value.kind === 'missing') return '(Missing metadata)'
  if (value.kind === 'no_converters') return '(No converters)'
  return value.value || '(Empty value)'
}

export function analyticsOptionLabel(option: AttackAnalyticsOption): string {
  if (option.key.kind === 'missing') return `${option.label} (missing metadata)`
  if (option.key.kind === 'no_converters') return `${option.label} (empty pipeline)`
  return option.label || '(Empty value)'
}

export function formatAnalyticsPercent(value: number | null): string {
  return value === null ? 'Unavailable' : PERCENT_FORMAT.format(value)
}

export function formatAnalyticsCount(value: number): string {
  return COUNT_FORMAT.format(value)
}

export function formatAnalyticsTime(value: string): string {
  return new Date(value).toLocaleString()
}

export function analyticsStatisticsLabel(statistics: AttackAnalyticsStatistics, marked: boolean): string {
  return `${formatAnalyticsCount(statistics.total_results)} results; ` +
    `ASR${marked ? '*' : ''} ${formatAnalyticsPercent(statistics.success_rate)}; ` +
    `${statistics.successes} successes / ${statistics.total_decided} decided; ` +
    `${statistics.failures} failures; ${statistics.errors} errors; ${statistics.undetermined} undetermined`
}

export function hasAnalyticsFilters(filters: AttackAnalyticsFilters): boolean {
  return filters.dimensions.length > 0 || filters.outcomes.length > 0 ||
    Boolean(filters.updated_after) || Boolean(filters.updated_before)
}

/** Drill-down predicates are additional AND conditions, even on the same dimension. */
export function appendAnalyticsDrilldown(
  filters: AttackAnalyticsFilters,
  predicates: AttackAnalyticsFilter[],
  outcome?: AttackOutcome,
): AttackAnalyticsFilters {
  return {
    ...filters,
    dimensions: [...filters.dimensions, ...predicates],
    outcomes: outcome ? [outcome] : filters.outcomes,
  }
}

export function analyticsQuery(
  view: AttackAnalyticsViewState,
  groupOffset = 0,
  allGroups = false,
): AttackAnalyticsQuery {
  return {
    filters: view.filters,
    group_by: view.chart === 'heatmap' ? view.heatmapRow : view.groupBy,
    compare_by: view.chart === 'heatmap' ? view.heatmapColumn : null,
    group_limit: allGroups ? ANALYTICS_GROUP_PAGE_SIZE : ANALYTICS_GROUP_LIMIT,
    group_offset: groupOffset,
    axis_limit: 20,
    result_limit: ANALYTICS_PAGE_SIZE,
  }
}

export function isAnalyticsLabelKey(value: string): boolean {
  return LABEL_KEY_PATTERN.test(value) && value !== 'operation' && value !== 'operator'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseDimension(value: unknown): AttackAnalyticsDimension {
  if (!isRecord(value)) throw new Error('A dimension is missing.')
  if (value.converter_direction !== undefined &&
    value.converter_direction !== 'request' && value.converter_direction !== 'response') {
    throw new Error('Unknown converter direction.')
  }
  if (value.name === 'label') {
    if (typeof value.label_key !== 'string' || !isAnalyticsLabelKey(value.label_key)) {
      throw new Error('Label keys must use 1-128 letters, numbers, underscores, periods or hyphens. Use dedicated operation/operator filters.')
    }
    if (value.converter_direction === 'response') throw new Error('Converter direction applies only to converters.')
    return { name: 'label', label_key: value.label_key }
  }
  if (value.label_key != null) throw new Error('Only custom labels accept a label key.')
  if (value.name === 'converter_type') {
    return { name: 'converter_type', converter_direction: value.converter_direction ?? 'request' }
  }
  if (value.converter_direction === 'response') throw new Error('Converter direction applies only to converters.')
  switch (value.name) {
    case 'operation':
    case 'operator':
    case 'targeted_harm_category':
    case 'attack_type':
    case 'objective_target':
    case 'model':
    case 'scenario':
      return { name: value.name }
    default:
      throw new Error('Unknown analytics dimension.')
  }
}

function parseValue(value: unknown): AttackAnalyticsValue {
  if (!isRecord(value)) throw new Error('Invalid filter value.')
  if (value.kind === 'value' && typeof value.value === 'string' && value.value.length <= 4096) {
    return { kind: 'value', value: value.value }
  }
  if ((value.kind === 'missing' || value.kind === 'no_converters') && value.value === null) {
    return { kind: value.kind, value: null }
  }
  throw new Error('Filter values must use a typed value, missing metadata, or no-converters key.')
}

function parseFilter(value: unknown): AttackAnalyticsFilter {
  if (!isRecord(value) || !Array.isArray(value.values) ||
    value.values.length === 0 || value.values.length > ANALYTICS_MAX_VALUES) {
    throw new Error('Each filter needs between 1 and 100 values.')
  }
  const dimension = parseDimension(value.dimension)
  const values = value.values.map(parseValue)
  const matchMode = value.match_mode ?? 'any'
  if (matchMode !== 'any' && matchMode !== 'all') throw new Error('Unknown filter match mode.')
  if (dimension.name !== 'converter_type' &&
    (matchMode === 'all' || values.some((item: AttackAnalyticsValue) => item.kind === 'no_converters'))) {
    throw new Error('ALL matching and no-converters values apply only to converter filters.')
  }
  if ((dimension.name === 'operation' || dimension.name === 'operator') &&
    values.some((item: AttackAnalyticsValue) => item.value !== null && item.value.length > 128)) {
    throw new Error('Operation and operator values must not exceed 128 characters.')
  }
  return { dimension, values, match_mode: matchMode }
}

function parseTimestamp(value: unknown): string | null {
  if (value == null) return null
  if (typeof value !== 'string' || !AWARE_TIMESTAMP_PATTERN.test(value) || !Number.isFinite(Date.parse(value))) {
    throw new Error('Last updated bounds must be valid timestamps with a time zone.')
  }
  return value
}

function parseFilters(value: unknown): AttackAnalyticsFilters {
  if (!isRecord(value) || !Array.isArray(value.dimensions) ||
    value.dimensions.length > ANALYTICS_MAX_PREDICATES || !Array.isArray(value.outcomes) || value.outcomes.length > 4) {
    throw new Error('Filters allow at most 16 predicates and four outcomes.')
  }
  const dimensions = value.dimensions.map(parseFilter)
  const outcomes: AttackOutcome[] = value.outcomes.map((outcome: unknown) => {
    if (outcome === 'success' || outcome === 'failure' || outcome === 'error' || outcome === 'undetermined') {
      return outcome
    }
    throw new Error('Unknown attack outcome.')
  })
  let valueCount = 0
  for (const predicate of dimensions) valueCount += predicate.values.length
  if (valueCount > 500) throw new Error('Select at most 500 dimension values.')
  const after = parseTimestamp(value.updated_after)
  const before = parseTimestamp(value.updated_before)
  if (after && before && Date.parse(after) >= Date.parse(before)) {
    throw new Error('Last updated after must be earlier than last updated before.')
  }
  return { dimensions, outcomes: [...new Set(outcomes)], updated_after: after, updated_before: before }
}

function parseView(value: unknown): AttackAnalyticsViewState {
  if (!isRecord(value)) throw new Error('Invalid analytics view.')
  const chart = value.chart
  const heatmapMetric = value.heatmapMetric
  if (chart !== 'outcomes' && chart !== 'success-rate' && chart !== 'heatmap') {
    throw new Error('Unknown analytics chart.')
  }
  if (heatmapMetric !== 'success_rate' && heatmapMetric !== 'total_results') {
    throw new Error('Unknown heatmap metric.')
  }
  const heatmapRow = parseDimension(value.heatmapRow)
  const heatmapColumn = parseDimension(value.heatmapColumn)
  if (analyticsDimensionKey(heatmapRow) === analyticsDimensionKey(heatmapColumn)) {
    throw new Error('Choose different heatmap row and column dimensions.')
  }
  return {
    filters: parseFilters(value.filters),
    groupBy: parseDimension(value.groupBy),
    heatmapRow,
    heatmapColumn,
    chart,
    heatmapMetric,
  }
}

export function analyticsViewError(view: AttackAnalyticsViewState): string | null {
  try {
    parseView(view)
    return null
  } catch (error: unknown) {
    return error instanceof Error ? error.message : 'Invalid analytics view.'
  }
}

export function analyticsViewToSearchParams(view: AttackAnalyticsViewState): URLSearchParams {
  return new URLSearchParams({ analytics: JSON.stringify(view) })
}

/** Reject invalid links visibly rather than silently broadening the requested cohort. */
export function analyticsViewFromSearchParams(
  params: URLSearchParams,
): { view: AttackAnalyticsViewState; error: string | null } {
  const encoded = params.get('analytics')
  if (encoded === null) return { view: DEFAULT_ANALYTICS_VIEW, error: null }
  try {
    const value: unknown = JSON.parse(encoded)
    return { view: parseView(value), error: null }
  } catch (error: unknown) {
    return {
      view: DEFAULT_ANALYTICS_VIEW,
      error: `Cannot restore this analytics link. ${error instanceof Error ? error.message : 'Invalid view.'}`,
    }
  }
}
