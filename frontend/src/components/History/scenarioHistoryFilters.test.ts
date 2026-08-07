import {
  DEFAULT_SCENARIO_HISTORY_FILTERS,
  scenarioHistoryFiltersFromSearchParams,
  scenarioHistoryFiltersToSearchParams,
} from './scenarioHistoryFilters'

describe('scenario history URL filters', () => {
  it('round-trips repeated filters and label search text', () => {
    const filters = {
      scenarioNames: ['red.team', 'benchmark'],
      statuses: ['IN_PROGRESS', 'FAILED'] as const,
      operator: ['alice', 'bob'],
      operation: ['nightly'],
      otherLabels: ['team:security', 'team:safety'],
      labelSearchText: 'team',
    }

    const params = scenarioHistoryFiltersToSearchParams({
      ...filters,
      statuses: [...filters.statuses],
    })

    expect(params.getAll('scenario')).toEqual(['red.team', 'benchmark'])
    expect(params.getAll('status')).toEqual(['IN_PROGRESS', 'FAILED'])
    expect(scenarioHistoryFiltersFromSearchParams(params)).toEqual({
      ...filters,
      statuses: [...filters.statuses],
    })
  })

  it('ignores invalid run states without dropping valid filters', () => {
    const params = new URLSearchParams('status=COMPLETED&status=UNKNOWN&operator=alice')

    expect(scenarioHistoryFiltersFromSearchParams(params)).toEqual({
      ...DEFAULT_SCENARIO_HISTORY_FILTERS,
      statuses: ['COMPLETED'],
      operator: ['alice'],
    })
  })
})
