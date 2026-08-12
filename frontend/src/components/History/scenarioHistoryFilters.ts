import type { ScenarioRunState } from '@/types'

export interface ScenarioHistoryFilters {
  scenarioNames: string[]
  statuses: ScenarioRunState[]
  operator: string[]
  operation: string[]
  otherLabels: string[]
  labelSearchText: string
}

export const DEFAULT_SCENARIO_HISTORY_FILTERS: ScenarioHistoryFilters = {
  scenarioNames: [],
  statuses: [],
  operator: [],
  operation: [],
  otherLabels: [],
  labelSearchText: '',
}

const RUN_STATES = new Set<ScenarioRunState>([
  'CREATED',
  'IN_PROGRESS',
  'COMPLETED',
  'FAILED',
  'CANCELLED',
])

export function scenarioHistoryFiltersFromSearchParams(
  params: URLSearchParams,
): ScenarioHistoryFilters {
  const statuses = params
    .getAll('status')
    .filter((status): status is ScenarioRunState => RUN_STATES.has(status as ScenarioRunState))
  return {
    scenarioNames: params.getAll('scenario'),
    statuses,
    operator: params.getAll('operator'),
    operation: params.getAll('operation'),
    otherLabels: params.getAll('label'),
    labelSearchText: params.get('labelSearch') ?? '',
  }
}

export function scenarioHistoryFiltersToSearchParams(
  filters: ScenarioHistoryFilters,
): URLSearchParams {
  const params = new URLSearchParams()
  for (const scenarioName of filters.scenarioNames) params.append('scenario', scenarioName)
  for (const status of filters.statuses) params.append('status', status)
  for (const operator of filters.operator) params.append('operator', operator)
  for (const operation of filters.operation) params.append('operation', operation)
  for (const label of filters.otherLabels) params.append('label', label)
  if (filters.labelSearchText) params.set('labelSearch', filters.labelSearchText)
  return params
}
