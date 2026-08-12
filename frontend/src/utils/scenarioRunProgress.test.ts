import type {
  ScenarioProgressResult,
  ScenarioRunPlan,
  ScenarioRunProgress,
} from '@/types'

import {
  INITIAL_SCENARIO_RUN_PROGRESS_STATE,
  getAttackGroupRollups,
  getAtomicGroupRollups,
  getAttemptAccounting,
  getAttemptPresentations,
  getAttemptRollups,
  getElapsedMilliseconds,
  getEtaMilliseconds,
  getOverallProgress,
  getSeedGroupRollups,
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
      group_kind: 'attack',
    },
    {
      id: 'group-b',
      atomic_attack_name: 'attack-b',
      display_group: 'Technique B',
      technique_eval_hash: 'eval-b',
      seed_group_ids: ['seed-1'],
      group_kind: 'attack',
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
      started_at: '2026-01-01T00:00:00Z',
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

  it('does not double-count overload evidence during cancellation catch-up', () => {
    const overload = {
      component_role: 'objective_target',
      count: 1,
      rate_limit_count: 1,
      server_error_count: 0,
      status_codes: [429],
      latest_timestamp: '2026-01-01T00:01:00Z',
    }
    const first = scenarioRunProgressReducer(INITIAL_SCENARIO_RUN_PROGRESS_STATE, {
      type: 'apply-page',
      page: makePage({ run: { ...makePage().run, overload_summaries: [overload] } }),
      fresh: true,
    })
    const cancelled = scenarioRunProgressReducer(first, {
      type: 'apply-run-summary',
      run: {
        scenario_result_id: 'run-1',
        scenario_name: 'TestScenario',
        scenario_registry_name: 'test.scenario',
        scenario_version: 1,
        status: 'CANCELLED',
        created_at: '2026-01-01T00:00:00Z',
        started_at: '2026-01-01T00:00:30Z',
        updated_at: '2026-01-01T00:02:00Z',
        techniques_used: [],
        total_attacks: 2,
        completed_attacks: 2,
        objective_achieved_rate: 0,
        failed_attacks: [],
        attack_retries: [],
        total_retries: 2,
        labels: {},
        overload_summaries: [{ ...overload, count: 2, rate_limit_count: 2 }],
      },
    })
    const caughtUp = scenarioRunProgressReducer(cancelled, {
      type: 'apply-page',
      page: makePage({
        plan: null,
        run: {
          ...makePage().run,
          status: 'CANCELLED',
          overload_summaries: [{
            ...overload,
            latest_timestamp: '2026-01-01T00:02:00Z',
          }],
        },
      }),
      fresh: false,
    })
    const olderDelta = scenarioRunProgressReducer(caughtUp, {
      type: 'apply-page',
      page: makePage({
        plan: null,
        run: {
          ...makePage().run,
          status: 'CANCELLED',
          overload_summaries: [{
            ...overload,
            latest_timestamp: '2026-01-01T00:00:30Z',
          }],
        },
      }),
      fresh: false,
    })

    expect(cancelled.overloadSummaries[0].count).toBe(1)
    expect(cancelled.run?.started_at).toBe('2026-01-01T00:00:30Z')
    expect(caughtUp.overloadSummaries[0].count).toBe(2)
    expect(olderDelta.overloadSummaries[0].latest_timestamp).toBe('2026-01-01T00:02:00Z')
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

  it('reports an initial failure and returns to loading when retried', () => {
    const failed = scenarioRunProgressReducer(INITIAL_SCENARIO_RUN_PROGRESS_STATE, {
      type: 'request-failed',
      message: 'Backend unavailable',
      notFound: false,
    })
    const retried = scenarioRunProgressReducer(failed, { type: 'retry' })

    expect(failed.loadStatus).toBe('error')
    expect(failed.stale).toBe(false)
    expect(retried.loadStatus).toBe('loading')
    expect(retried.error).toBeNull()
  })

  it('drops a cached plan when the server resets without a replacement', () => {
    const reset = scenarioRunProgressReducer(readyState([
      makeResult('attempt-1', 'group-a', 'seed-1', 'success', 1),
    ]), {
      type: 'apply-page',
      page: makePage({
        plan: null,
        reset: true,
        results: [],
      }),
      fresh: false,
    })

    expect(reset.plan).toBeNull()
    expect(reset.results).toEqual([])
  })
})

describe('scenario run progress calculations', () => {
  it('counts progress units once and includes repeated logical attempts as retry work', () => {
    const state = readyState([
      makeResult('error-1', 'group-a', 'seed-1', 'error', 1),
      makeResult('failure-1', 'group-a', 'seed-1', 'failure', 2),
      makeResult('success-1', 'group-a', 'seed-1', 'success', 3),
      makeResult('error-2', 'group-a', 'seed-1', 'error', 4),
    ])

    expect(getOverallProgress(state)).toEqual({ completed: 1, planned: 3, percent: 33 })
    expect(getAttackGroupRollups(state)[0]).toMatchObject({
      completed: 1,
      planned: 2,
      succeeded: 1,
      evaluated: 1,
      errors: 2,
      retries: 3,
      persistedAttempts: 4,
      attackAttempts: 4,
    })
    expect(getAttemptAccounting(state).retries).toBe(3)
    expect(getAttemptRollups(state)[0].retries).toBe(3)
  })

  it('attributes cross-label Adaptive retries without undercounting the logical unit', () => {
    const state = readyState([
      makeResult('adaptive-1', 'group-a', 'seed-1', 'failure', 1, {
        result_kind: 'adaptive_technique',
        technique_name: 'Technique Alpha',
      }),
      makeResult('adaptive-2', 'group-a', 'seed-1', 'success', 2, {
        result_kind: 'adaptive_technique',
        technique_name: 'Technique Beta',
      }),
    ])
    const rollups = getAttemptRollups(state)

    expect(rollups).toEqual(expect.arrayContaining([
      expect.objectContaining({ label: 'Technique Alpha', retries: 0 }),
      expect.objectContaining({ label: 'Technique Beta', retries: 1 }),
    ]))
    expect(rollups.reduce((total, rollup) => total + rollup.retries, 0)).toBe(1)
    expect(getAttemptAccounting(state).retries).toBe(1)
    expect(getAttackGroupRollups(state)[0].retries).toBe(1)
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

  it('keeps legacy groups without typed semantics unknown rather than guessing they are attacks', () => {
    const state = readyState([makeResult('legacy-attempt', 'group-a', 'seed-1', 'success', 1)])
    const legacyPlan = state.plan
      ? {
          ...state.plan,
          atomic_groups: state.plan.atomic_groups.map(({ group_kind: _groupKind, ...group }) => group),
        }
      : null

    expect(getAttemptPresentations({ ...state, plan: legacyPlan }).get('legacy-attempt')).toMatchObject({
      role: 'unknown',
      label: 'Additional persisted result',
    })
  })

  it('calculates attack-group and objective rollups across progress groups', () => {
    const state = readyState([
      makeResult('a-1', 'group-a', 'seed-1', 'success', 1),
      makeResult('a-2', 'group-a', 'seed-2', 'failure', 2),
      makeResult('b-1', 'group-b', 'seed-1', 'failure', 3),
    ])

    expect(getAttackGroupRollups(state)).toEqual([
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
      persistedAttempts: 2,
      attackAttempts: 2,
    })
  })

  it('sorts legacy seed groups by ID when objective text is unavailable', () => {
    const state = {
      ...readyState([
        makeResult('z-attempt', 'group-a', 'seed-z', 'success', 1),
        makeResult('a-attempt', 'group-b', 'seed-a', 'failure', 2),
      ]),
      plan: null,
      planComplete: false,
    }

    expect(getSeedGroupRollups(state).map(({ id, objective }) => ({ id, objective }))).toEqual([
      { id: 'seed-a', objective: null },
      { id: 'seed-z', objective: null },
    ])
  })

  it('presents inferred and explicit attempt roles with safe fallback labels', () => {
    const state = {
      ...readyState([
        makeResult('orchestration', 'group-a', 'seed-1', 'failure', 1),
        makeResult('adaptive-child', 'group-a', 'seed-2', 'success', 2, {
          result_kind: 'adaptive_technique',
        }),
        makeResult('adaptive-parent', 'group-a', 'seed-2', 'failure', 3, {
          result_kind: 'aggregate_parent',
        }),
        makeResult('aggregate-parent', 'group-b', 'seed-1', 'failure', 4, {
          result_kind: 'aggregate_parent',
        }),
        makeResult('fallback-attack', 'group-b', 'seed-2', 'error', 5, {
          atomic_attack_name: '',
        }),
      ]),
      plan: {
        ...PLAN,
        atomic_groups: [
          { ...PLAN.atomic_groups[0], group_kind: 'adaptive' as const },
          {
            ...PLAN.atomic_groups[1],
            atomic_attack_name: '',
            display_group: '',
            group_kind: 'attack' as const,
          },
        ],
      },
    }
    const presentations = getAttemptPresentations(state)

    expect(presentations.get('orchestration')).toMatchObject({
      role: 'adaptive_orchestration',
      label: 'Adaptive orchestration',
    })
    expect(presentations.get('adaptive-child')).toMatchObject({
      role: 'adaptive_technique',
      label: 'Adaptive technique',
    })
    expect(presentations.get('adaptive-parent')?.label).toBe('Adaptive aggregate parent')
    expect(presentations.get('aggregate-parent')?.label).toBe('Aggregate parent')
    expect(presentations.get('fallback-attack')).toMatchObject({
      role: 'attack',
      label: 'Attack',
    })
    expect(getAttemptRollups(state)).toEqual(expect.arrayContaining([
      expect.objectContaining({ role: 'attack', succeeded: 0, errors: 1 }),
    ]))
  })

  it('extends plan metadata for persisted seeds and groups absent from the plan', () => {
    const state = readyState([
      makeResult('new-seed', 'group-a', 'seed-extra', 'success', 1),
      makeResult('new-group', 'group-new', 'seed-1', 'failure', 2, {
        atomic_attack_name: '',
      }),
    ])
    const rollups = getAtomicGroupRollups(state)

    expect(rollups.find((group) => group.id === 'group-a')?.planned).toBe(3)
    expect(rollups.find((group) => group.id === 'group-new')?.displayGroup).toBe(
      'Persisted attack group',
    )
  })

  it('separates eight progress units, twelve persisted results, and zero retries', () => {
    const objectiveIds = ['seed-1', 'seed-2', 'seed-3', 'seed-4']
    const plan: ScenarioRunPlan = {
      version: 1,
      scenario_registry_name: 'adaptive.text',
      seed_groups: objectiveIds.map((id, index) => ({
        id,
        objective_sha256: `sha-${index + 1}`,
        objective: `Objective ${index + 1}`,
      })),
      atomic_groups: [
        {
          id: 'baseline',
          atomic_attack_name: 'baseline',
          display_group: 'Direct baseline',
          technique_eval_hash: 'baseline-eval',
          seed_group_ids: objectiveIds,
          group_kind: 'direct_baseline',
        },
        ...objectiveIds.map((seedId, index) => ({
          id: `adaptive-${index + 1}`,
          atomic_attack_name: 'adaptive',
          display_group: index < 1 ? 'Fairness' : 'Harassment',
          technique_eval_hash: `adaptive-eval-${index + 1}`,
          seed_group_ids: [seedId],
          group_kind: 'adaptive' as const,
        })),
      ],
    }
    const results = objectiveIds.flatMap((seedId, index) => [
      makeResult(`baseline-${index}`, 'baseline', seedId, 'success', index * 3, {
        atomic_attack_name: 'baseline',
      }),
      makeResult(`technique-${index}`, `adaptive-${index + 1}`, seedId, 'success', index * 3 + 1, {
        technique_name: index === 0 ? 'Fairness technique' : 'Harassment technique',
        attempt_index: 1,
      }),
      makeResult(`envelope-${index}`, `adaptive-${index + 1}`, seedId, 'success', index * 3 + 2, {
        total_retries: 7,
        result_kind: 'aggregate_parent',
      }),
    ])
    const state = scenarioRunProgressReducer(INITIAL_SCENARIO_RUN_PROGRESS_STATE, {
      type: 'apply-page',
      page: makePage({
        plan,
        results,
        run: {
          ...makePage().run,
          scenario_registry_name: 'adaptive.text',
          status: 'COMPLETED',
          completed_at: '2026-01-01T00:15:00Z',
        },
      }),
      fresh: true,
    })

    expect(getOverallProgress(state)).toEqual({ completed: 8, planned: 8, percent: 100 })
    expect(getAttemptAccounting(state)).toMatchObject({
      objectiveCount: 4,
      persistedAttempts: 12,
      attackAttempts: 8,
      aggregateParentRecords: 4,
      adaptiveAggregateParentRecords: 4,
      uniformTargetAttacksPerObjective: 2,
      completedProgressUnits: 8,
      plannedProgressUnits: 8,
      retries: 0,
    })
    expect(Array.from(getAttemptAccounting(state).uniformTargetRoleCounts?.entries() ?? [])).toEqual([
      ['direct_baseline', 1],
      ['adaptive_technique', 1],
    ])
    expect(getSeedGroupRollups(state)[0]).toMatchObject({
      completed: 2,
      planned: 2,
      persistedAttempts: 3,
      attackAttempts: 2,
      retries: 0,
    })
    expect(getAttemptRollups(state)).toEqual([
      expect.objectContaining({ role: 'direct_baseline', persistedAttempts: 4, retries: 0 }),
      expect.objectContaining({ role: 'adaptive_technique', label: 'Fairness technique', persistedAttempts: 1 }),
      expect.objectContaining({ role: 'adaptive_technique', label: 'Harassment technique', persistedAttempts: 3 }),
      expect.objectContaining({ role: 'aggregate_parent', persistedAttempts: 4, retries: 0 }),
    ])
  })

  it('recognizes legacy Adaptive envelopes from typed sibling technique results', () => {
    const state = readyState([
      makeResult('adaptive-child', 'group-a', 'seed-1', 'failure', 1, {
        result_kind: 'adaptive_technique',
        technique_name: 'many_shot',
        attempt_index: 1,
      }),
      makeResult('adaptive-envelope', 'group-a', 'seed-1', 'failure', 2, {
        result_kind: 'aggregate_parent',
      }),
      makeResult('unrelated-envelope', 'group-a', 'seed-2', 'failure', 3, {
        result_kind: 'aggregate_parent',
      }),
    ])

    expect(getAttemptAccounting(state)).toMatchObject({
      attackAttempts: 1,
      aggregateParentRecords: 2,
      adaptiveAggregateParentRecords: 1,
      retries: 0,
    })
  })

  it('counts only Adaptive child attempts in retry and error accounting', () => {
    const state = readyState([
      makeResult('adaptive-child-error', 'group-a', 'seed-1', 'error', 1, {
        total_retries: 2,
        result_kind: 'adaptive_technique',
        technique_name: 'first technique',
        attempt_index: 1,
      }),
      makeResult('adaptive-child-success', 'group-a', 'seed-1', 'success', 2, {
        result_kind: 'adaptive_technique',
        technique_name: 'second technique',
        attempt_index: 2,
      }),
      makeResult('adaptive-envelope', 'group-a', 'seed-1', 'error', 3, {
        total_retries: 7,
        result_kind: 'aggregate_parent',
      }),
    ])

    expect(getOverallProgress(state).completed).toBe(1)
    expect(getAttemptAccounting(state)).toMatchObject({
      persistedAttempts: 3,
      attackAttempts: 2,
      aggregateParentRecords: 1,
      retries: 3,
    })
    expect(getAttackGroupRollups(state).find((group) => group.id === 'Technique A')).toMatchObject({
      completed: 1,
      succeeded: 1,
      errors: 1,
      persistedAttempts: 3,
      attackAttempts: 2,
    })
    expect(getSeedGroupRollups(state).find((seed) => seed.id === 'seed-1')).toMatchObject({
      completed: 1,
      succeeded: 1,
      errors: 1,
      persistedAttempts: 3,
      attackAttempts: 2,
    })
    expect(getAttemptRollups(state)).toEqual(expect.arrayContaining([
      expect.objectContaining({
        role: 'adaptive_technique',
        label: 'first technique',
        persistedAttempts: 1,
        errors: 1,
        retries: 2,
      }),
      expect.objectContaining({
        role: 'adaptive_technique',
        label: 'second technique',
        persistedAttempts: 1,
        errors: 0,
        retries: 1,
      }),
      expect.objectContaining({
        role: 'aggregate_parent',
        persistedAttempts: 1,
        retries: 0,
      }),
    ]))
  })

  it('does not let a later successful aggregate envelope complete a target-facing error', () => {
    const state = readyState([
      makeResult('target-error', 'group-a', 'seed-1', 'error', 1, {
        result_kind: 'attack',
      }),
      makeResult('aggregate-success', 'group-a', 'seed-1', 'success', 2, {
        result_kind: 'aggregate_parent',
      }),
    ])

    expect(getOverallProgress(state)).toEqual({ completed: 0, planned: 3, percent: 0 })
    expect(getAttackGroupRollups(state).find((group) => group.id === 'Technique A')).toMatchObject({
      completed: 0,
      succeeded: 0,
      errors: 1,
      persistedAttempts: 2,
      attackAttempts: 1,
    })
    expect(getSeedGroupRollups(state).find((seed) => seed.id === 'seed-1')).toMatchObject({
      completed: 0,
      succeeded: 0,
      errors: 1,
      persistedAttempts: 2,
      attackAttempts: 1,
    })
    expect(getAttemptAccounting(state)).toMatchObject({
      persistedAttempts: 2,
      attackAttempts: 1,
      aggregateParentRecords: 1,
      completedProgressUnits: 0,
    })
  })

  it('omits uniform target arithmetic when no target-facing attacks exist', () => {
    const state = readyState([
      makeResult('aggregate-only', 'group-a', 'seed-1', 'failure', 1, {
        result_kind: 'aggregate_parent',
      }),
    ])

    expect(getAttemptAccounting(state)).toMatchObject({
      attackAttempts: 0,
      uniformTargetAttacksPerObjective: null,
      uniformTargetRoleCounts: null,
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

  it('uses execution start for elapsed time and excludes a long queue delay from active ETA', () => {
    const active = {
      ...makePage().run,
      created_at: '2026-01-01T00:00:00Z',
      started_at: '2026-01-01T01:00:00Z',
    }
    expect(getElapsedMilliseconds(active, Date.parse('2026-01-01T01:05:00Z'))).toBe(300_000)

    const terminal = {
      ...active,
      status: 'COMPLETED' as const,
      completed_at: '2026-01-01T01:03:00Z',
    }
    expect(getElapsedMilliseconds(terminal, Date.parse('2026-01-01T01:05:00Z'))).toBe(180_000)

    const state = readyState([makeResult('a-1', 'group-a', 'seed-1', 'success', 1)])
    state.run = active
    expect(getEtaMilliseconds(state, Date.parse('2026-01-01T01:02:00Z'))).toBe(240_000)
    expect(getElapsedMilliseconds(
      { ...active, created_at: 'not-a-timestamp' },
      Date.parse('2026-01-01T00:05:00Z'),
    )).toBe(0)
  })

  it('does not count queue wait as elapsed time or fabricate a queued ETA', () => {
    const queued = {
      ...makePage().run,
      status: 'QUEUED' as const,
      created_at: '2026-01-01T00:00:00Z',
      started_at: null,
    }
    const now = Date.parse('2026-01-01T01:00:00Z')

    expect(getElapsedMilliseconds(queued, now)).toBe(0)

    const state = readyState([makeResult('a-1', 'group-a', 'seed-1', 'success', 1)])
    state.run = queued
    expect(getEtaMilliseconds(state, now)).toBeNull()

    expect(getElapsedMilliseconds(
      { ...queued, status: 'CANCELLED', completed_at: '2026-01-01T00:30:00Z' },
      now,
    )).toBe(0)
  })

  it('calculates ETA from observed wall-clock completion rate and hides unsafe estimates', () => {
    const state = readyState([makeResult('a-1', 'group-a', 'seed-1', 'success', 1)])
    expect(getEtaMilliseconds(state, Date.parse('2026-01-01T00:02:00Z'))).toBe(240_000)
    expect(getEtaMilliseconds(state, Date.parse(makePage().run.created_at))).toBeNull()
    expect(getEtaMilliseconds(state, Number.MAX_VALUE)).toBeNull()

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
