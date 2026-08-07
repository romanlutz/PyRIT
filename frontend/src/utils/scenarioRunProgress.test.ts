import type {
  ScenarioProgressResult,
  ScenarioRunPlan,
  ScenarioRunProgress,
} from '@/types'

import {
  INITIAL_SCENARIO_RUN_PROGRESS_STATE,
  getAtomicGroupRollups,
  getElapsedMilliseconds,
  getEtaMilliseconds,
  getOverallProgress,
  getSeedGroupRollups,
  getTechniqueRollups,
  scenarioRunProgressReducer,
  type ScenarioRunProgressState,
} from './scenarioRunProgress'

const PLAN: ScenarioRunPlan = {
  version: 1,
  scenario_registry_name: 'test.scenario',
  atomic_groups: [
    {
      id: 'group-a',
      atomic_attack_name: 'attack-a',
      display_group: 'Technique A',
      technique_eval_hash: 'eval-a',
      seed_group_ids: ['seed-1', 'seed-2'],
    },
    {
      id: 'group-b',
      atomic_attack_name: 'attack-b',
      display_group: 'Technique B',
      technique_eval_hash: 'eval-b',
      seed_group_ids: ['seed-1'],
    },
  ],
  seed_groups: [
    { id: 'seed-1', objective_sha256: 'sha-1', objective: 'First objective' },
    { id: 'seed-2', objective_sha256: 'sha-2', objective: 'Second objective' },
  ],
}

function makeResult(
  id: string,
  atomicGroupId: string,
  seedGroupId: string,
  outcome: ScenarioProgressResult['outcome'],
  minute: number,
  overrides: Partial<ScenarioProgressResult> = {},
): ScenarioProgressResult {
  return {
    attack_result_id: id,
    atomic_group_id: atomicGroupId,
    atomic_attack_name: atomicGroupId === 'group-a' ? 'attack-a' : 'attack-b',
    seed_group_id: seedGroupId,
    outcome,
    execution_time_ms: 1_000,
    timestamp: `2026-01-01T00:${String(minute).padStart(2, '0')}:00Z`,
    total_retries: 0,
    retries: [],
    ...overrides,
  }
}

function makePage(overrides: Partial<ScenarioRunProgress> = {}): ScenarioRunProgress {
  return {
    run: {
      scenario_result_id: 'run-1',
      scenario_name: 'TestScenario',
      scenario_registry_name: 'test.scenario',
      scenario_version: 1,
      status: 'IN_PROGRESS',
      created_at: '2026-01-01T00:00:00Z',
    },
    plan: PLAN,
    reset: false,
    active_atomic_group_ids: [],
    results: [],
    next_cursor: 'cursor-1',
    has_more: false,
    plan_complete: true,
    ...overrides,
  }
}

function readyState(results: ScenarioProgressResult[]): ScenarioRunProgressState {
  return scenarioRunProgressReducer(INITIAL_SCENARIO_RUN_PROGRESS_STATE, {
    type: 'apply-page',
    page: makePage({ results }),
    fresh: true,
  })
}

describe('scenarioRunProgressReducer', () => {
  it('merges duplicated pages idempotently by attack result id', () => {
    const result = makeResult('attempt-1', 'group-a', 'seed-1', 'success', 1)
    const first = readyState([result])
    const duplicate = scenarioRunProgressReducer(first, {
      type: 'apply-page',
      page: makePage({ plan: null, results: [result], next_cursor: 'cursor-1' }),
      fresh: false,
    })

    expect(duplicate.results).toEqual([result])
    expect(duplicate.cursor).toBe('cursor-1')
  })

  it('atomically resets prior results when the server requests reset', () => {
    const first = readyState([makeResult('old', 'group-a', 'seed-1', 'success', 1)])
    const replacement = makeResult('new', 'group-b', 'seed-1', 'failure', 2)
    const reset = scenarioRunProgressReducer(first, {
      type: 'apply-page',
      page: makePage({ reset: true, results: [replacement], next_cursor: 'cursor-2' }),
      fresh: false,
    })

    expect(reset.results).toEqual([replacement])
    expect(reset.cursor).toBe('cursor-2')
  })

  it('retains last-good data and marks it stale after a transient failure', () => {
    const first = readyState([makeResult('attempt-1', 'group-a', 'seed-1', 'success', 1)])
    const failed = scenarioRunProgressReducer(first, {
      type: 'request-failed',
      message: 'Network unavailable',
      notFound: false,
    })

    expect(failed.results).toHaveLength(1)
    expect(failed.loadStatus).toBe('ready')
    expect(failed.stale).toBe(true)
    expect(failed.error).toBe('Network unavailable')
  })
})

describe('scenario run progress calculations', () => {
  it('counts executable units once across multiple attempts and completes from the latest non-error outcome', () => {
    const state = readyState([
      makeResult('error-1', 'group-a', 'seed-1', 'error', 1),
      makeResult('failure-1', 'group-a', 'seed-1', 'failure', 2),
      makeResult('success-1', 'group-a', 'seed-1', 'success', 3),
      makeResult('error-2', 'group-a', 'seed-1', 'error', 4),
    ])

    expect(getOverallProgress(state)).toEqual({ completed: 1, planned: 3, percent: 33 })
    expect(getTechniqueRollups(state)[0]).toMatchObject({
      completed: 1,
      planned: 2,
      succeeded: 1,
      evaluated: 1,
      errors: 2,
      retries: 3,
    })
  })

  it('keeps an error-only unit attempted but incomplete', () => {
    const state = readyState([
      makeResult('error-1', 'group-a', 'seed-1', 'error', 1, { total_retries: 2 }),
    ])

    expect(getOverallProgress(state).completed).toBe(0)
    expect(getAtomicGroupRollups(state)[0]).toMatchObject({
      completed: 0,
      errors: 1,
      retries: 2,
      status: 'Pending',
    })
  })

  it('does not infer a planned total or percentage for legacy runs', () => {
    const state = {
      ...readyState([makeResult('attempt-1', 'group-a', 'seed-1', 'success', 1)]),
      planComplete: false,
    }

    expect(getOverallProgress(state)).toEqual({ completed: 1, planned: null, percent: null })
    expect(getEtaMilliseconds(state, Date.parse('2026-01-01T00:10:00Z'))).toBeNull()
  })

  it('calculates technique and seed rollups across techniques', () => {
    const state = readyState([
      makeResult('a-1', 'group-a', 'seed-1', 'success', 1),
      makeResult('a-2', 'group-a', 'seed-2', 'failure', 2),
      makeResult('b-1', 'group-b', 'seed-1', 'failure', 3),
    ])

    expect(getTechniqueRollups(state)).toEqual([
      expect.objectContaining({
        displayGroup: 'Technique A',
        completed: 2,
        planned: 2,
        succeeded: 1,
        evaluated: 2,
        successPercent: 50,
      }),
      expect.objectContaining({
        displayGroup: 'Technique B',
        completed: 1,
        planned: 1,
        succeeded: 0,
        evaluated: 1,
        successPercent: 0,
      }),
    ])
    expect(getSeedGroupRollups(state)[0]).toMatchObject({
      id: 'seed-1',
      completed: 2,
      planned: 2,
      succeeded: 1,
      evaluated: 2,
      successPercent: 50,
    })
  })

  it('sorts atomic states and lets active IDs win while a run is nonterminal', () => {
    const state = {
      ...readyState([
        makeResult('a-1', 'group-a', 'seed-1', 'success', 1),
        makeResult('a-2', 'group-a', 'seed-2', 'failure', 2),
      ]),
      activeAtomicGroupIds: ['group-a'],
    }

    expect(getAtomicGroupRollups(state).map((group) => [group.id, group.status])).toEqual([
      ['group-a', 'Running'],
      ['group-b', 'Pending'],
    ])
  })

  it('marks unfinished groups incomplete in terminal runs', () => {
    const state = {
      ...readyState([makeResult('a-1', 'group-a', 'seed-1', 'success', 1)]),
      run: { ...makePage().run, status: 'FAILED' as const, completed_at: '2026-01-01T00:05:00Z' },
    }

    expect(getAtomicGroupRollups(state).map((group) => [group.id, group.status])).toEqual([
      ['group-a', 'Incomplete'],
      ['group-b', 'Incomplete'],
    ])
  })

  it('uses now for active elapsed time and completed_at for terminal elapsed time', () => {
    const active = makePage().run
    expect(getElapsedMilliseconds(active, Date.parse('2026-01-01T00:05:00Z'))).toBe(300_000)

    const terminal = {
      ...active,
      status: 'COMPLETED' as const,
      completed_at: '2026-01-01T00:03:00Z',
    }
    expect(getElapsedMilliseconds(terminal, Date.parse('2026-01-01T00:05:00Z'))).toBe(180_000)
  })

  it('calculates ETA from observed wall-clock completion rate and hides unsafe estimates', () => {
    const state = readyState([makeResult('a-1', 'group-a', 'seed-1', 'success', 1)])
    expect(getEtaMilliseconds(state, Date.parse('2026-01-01T00:02:00Z'))).toBe(240_000)

    expect(getEtaMilliseconds(
      { ...state, results: [] },
      Date.parse('2026-01-01T00:02:00Z'),
    )).toBeNull()
    const run = state.run
    expect(run).not.toBeNull()
    if (run) {
      expect(getEtaMilliseconds(
        { ...state, run: { ...run, status: 'COMPLETED' } },
        Date.parse('2026-01-01T00:02:00Z'),
      )).toBeNull()
    }
  })
})
