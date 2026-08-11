import type {
  ScenarioProgressHeader,
  ScenarioProgressResult,
  ScenarioRunPlan,
  ScenarioRunPlanAtomicGroup,
  ScenarioRunState,
  ScenarioRunSummary,
} from '@/types'

export type ScenarioRunLoadStatus = 'loading' | 'ready' | 'not-found' | 'error'
export type AtomicGroupStatus = 'Running' | 'Pending' | 'Incomplete' | 'Completed'

export interface ScenarioRunProgressState {
  readonly loadStatus: ScenarioRunLoadStatus
  readonly run: ScenarioProgressHeader | null
  readonly plan: ScenarioRunPlan | null
  readonly planComplete: boolean
  readonly activeAtomicGroupIds: string[]
  readonly results: ScenarioProgressResult[]
  readonly cursor: string | null
  readonly hasMore: boolean
  readonly error: string | null
  readonly stale: boolean
}

export type ScenarioRunProgressAction =
  | { readonly type: 'apply-page'; readonly page: import('@/types').ScenarioRunProgress; readonly fresh: boolean }
  | { readonly type: 'request-failed'; readonly message: string; readonly notFound: boolean }
  | { readonly type: 'retry' }
  | { readonly type: 'apply-run-summary'; readonly run: ScenarioRunSummary }

export interface OverallProgress {
  readonly completed: number
  readonly planned: number | null
  readonly percent: number | null
}

export interface Rollup {
  readonly completed: number
  readonly planned: number
  readonly succeeded: number
  readonly evaluated: number
  readonly successPercent: number | null
  readonly errors: number
  readonly retries: number
}

export interface TechniqueRollup extends Rollup {
  readonly id: string
  readonly displayGroup: string
  readonly atomicAttackNames: string[]
}

export interface SeedGroupRollup extends Rollup {
  readonly id: string
  readonly objective: string | null
}

export interface AtomicGroupRollup extends Rollup {
  readonly id: string
  readonly atomicAttackName: string
  readonly displayGroup: string
  readonly status: AtomicGroupStatus
}

interface UnitAttempts {
  readonly atomicGroupId: string
  readonly seedGroupId: string
  readonly attempts: ScenarioProgressResult[]
  readonly latestAttempt: ScenarioProgressResult
  readonly latestNonError: ScenarioProgressResult | null
}

const TERMINAL_STATES: ReadonlySet<ScenarioRunState> = new Set(['COMPLETED', 'FAILED', 'CANCELLED'])
const ATOMIC_STATUS_ORDER: Record<AtomicGroupStatus, number> = {
  Running: 0,
  Pending: 1,
  Incomplete: 2,
  Completed: 3,
}

export const INITIAL_SCENARIO_RUN_PROGRESS_STATE: ScenarioRunProgressState = {
  loadStatus: 'loading',
  run: null,
  plan: null,
  planComplete: false,
  activeAtomicGroupIds: [],
  results: [],
  cursor: null,
  hasMore: false,
  error: null,
  stale: false,
}

export function isTerminalRunState(status: ScenarioRunState): boolean {
  return TERMINAL_STATES.has(status)
}

export function scenarioRunProgressReducer(
  state: ScenarioRunProgressState,
  action: ScenarioRunProgressAction,
): ScenarioRunProgressState {
  if (action.type === 'request-failed') {
    const hasGoodData = state.run !== null
    return {
      ...state,
      loadStatus: action.notFound && !hasGoodData ? 'not-found' : hasGoodData ? 'ready' : 'error',
      error: action.message,
      stale: hasGoodData,
      hasMore: false,
    }
  }

  if (action.type === 'retry') {
    return {
      ...state,
      loadStatus: state.run ? 'ready' : 'loading',
      error: null,
      stale: false,
    }
  }

  if (action.type === 'apply-run-summary') {
    return {
      ...state,
      loadStatus: 'ready',
      run: {
        scenario_result_id: action.run.scenario_result_id,
        scenario_name: action.run.scenario_name,
        scenario_registry_name: action.run.scenario_registry_name,
        scenario_version: action.run.scenario_version,
        status: action.run.status,
        created_at: action.run.created_at,
        completed_at: action.run.completed_at,
      },
      activeAtomicGroupIds: [],
      error: null,
      stale: false,
      hasMore: false,
    }
  }

  const shouldReset = action.fresh || action.page.reset || action.page.plan !== null
  const resultsById = new Map<string, ScenarioProgressResult>()
  if (!shouldReset) {
    for (const result of state.results) {
      resultsById.set(result.attack_result_id, result)
    }
  }
  for (const result of action.page.results) {
    resultsById.set(result.attack_result_id, result)
  }

  const results = [...resultsById.values()].sort(compareAttempts)
  return {
    loadStatus: 'ready',
    run: action.page.run,
    plan: action.page.plan ?? (shouldReset ? null : state.plan),
    planComplete: action.page.plan_complete,
    activeAtomicGroupIds: [...new Set(action.page.active_atomic_group_ids)],
    results,
    cursor: action.page.next_cursor ?? state.cursor,
    hasMore: action.page.has_more,
    error: null,
    stale: false,
  }
}

export function getOverallProgress(state: ScenarioRunProgressState): OverallProgress {
  const units = buildUnitAttempts(state.results)
  const completed = [...units.values()].filter((unit) => unit.latestNonError !== null).length
  if (!state.planComplete || !state.plan) {
    return { completed, planned: null, percent: null }
  }

  const planned = state.plan.atomic_groups.reduce(
    (total, group) => total + new Set(group.seed_group_ids).size,
    0,
  )
  const plannedKeys = buildPlannedUnitKeys(state.plan.atomic_groups)
  const plannedCompleted = [...units.entries()].filter(
    ([key, unit]) => plannedKeys.has(key) && unit.latestNonError !== null,
  ).length
  return {
    completed: plannedCompleted,
    planned,
    percent: planned > 0 ? boundedPercent(plannedCompleted, planned) : 0,
  }
}

export function getElapsedMilliseconds(
  run: ScenarioProgressHeader,
  nowMilliseconds: number,
): number {
  const created = Date.parse(run.created_at)
  const terminalEnd = run.completed_at ? Date.parse(run.completed_at) : Number.NaN
  const end = isTerminalRunState(run.status) && Number.isFinite(terminalEnd)
    ? terminalEnd
    : nowMilliseconds
  if (!Number.isFinite(created) || !Number.isFinite(end)) {
    return 0
  }
  return Math.max(0, end - created)
}

export function getEtaMilliseconds(
  state: ScenarioRunProgressState,
  nowMilliseconds: number,
): number | null {
  if (!state.run || !state.planComplete || isTerminalRunState(state.run.status)) {
    return null
  }
  const progress = getOverallProgress(state)
  if (progress.planned === null || progress.planned <= 0 || progress.completed <= 0) {
    return null
  }
  const remaining = Math.max(0, progress.planned - progress.completed)
  if (remaining === 0) {
    return 0
  }
  const elapsed = getElapsedMilliseconds(state.run, nowMilliseconds)
  if (elapsed <= 0) {
    return null
  }
  const estimate = (elapsed / progress.completed) * remaining
  return Number.isFinite(estimate) && estimate >= 0 ? estimate : null
}

export function getTechniqueRollups(state: ScenarioRunProgressState): TechniqueRollup[] {
  const groupMetadata = buildGroupMetadata(state)
  const units = buildUnitAttempts(state.results)
  const rollups = new Map<string, TechniqueRollup>()

  for (const group of groupMetadata.values()) {
    const existing = rollups.get(group.display_group)
    const base = existing ?? {
      id: group.display_group,
      displayGroup: group.display_group,
      atomicAttackNames: [],
      completed: 0,
      planned: 0,
      succeeded: 0,
      evaluated: 0,
      successPercent: null,
      errors: 0,
      retries: 0,
    }
    const groupRollup = aggregateGroup(group.id, group.seed_group_ids, units)
    rollups.set(group.display_group, {
      ...base,
      atomicAttackNames: [...new Set([...base.atomicAttackNames, group.atomic_attack_name])],
      completed: base.completed + groupRollup.completed,
      planned: base.planned + groupRollup.planned,
      succeeded: base.succeeded + groupRollup.succeeded,
      evaluated: base.evaluated + groupRollup.evaluated,
      successPercent: null,
      errors: base.errors + groupRollup.errors,
      retries: base.retries + groupRollup.retries,
    })
  }

  return [...rollups.values()]
    .map((rollup) => ({
      ...rollup,
      successPercent: rollup.evaluated > 0 ? boundedPercent(rollup.succeeded, rollup.evaluated) : null,
    }))
    .sort((left, right) => left.displayGroup.localeCompare(right.displayGroup))
}

export function getSeedGroupRollups(state: ScenarioRunProgressState): SeedGroupRollup[] {
  const groups = buildGroupMetadata(state)
  const units = buildUnitAttempts(state.results)
  const objectives = new Map(state.plan?.seed_groups.map((seed) => [seed.id, seed.objective]) ?? [])
  const seedIds = new Set<string>(objectives.keys())
  for (const group of groups.values()) {
    for (const seedId of group.seed_group_ids) {
      seedIds.add(seedId)
    }
  }

  return [...seedIds].map((seedId) => {
    const relevantGroups = [...groups.values()].filter((group) => group.seed_group_ids.includes(seedId))
    const relevantUnits = relevantGroups
      .map((group) => units.get(unitKey(group.id, seedId)))
      .filter((unit): unit is UnitAttempts => unit !== undefined)
    const rollup = aggregateUnits(relevantUnits, relevantGroups.length)
    return { id: seedId, objective: objectives.get(seedId) ?? null, ...rollup }
  }).sort((left, right) => {
    const leftLabel = left.objective ?? left.id
    const rightLabel = right.objective ?? right.id
    return leftLabel.localeCompare(rightLabel)
  })
}

export function getAtomicGroupRollups(state: ScenarioRunProgressState): AtomicGroupRollup[] {
  const groups = buildGroupMetadata(state)
  const units = buildUnitAttempts(state.results)
  const terminal = state.run ? isTerminalRunState(state.run.status) : false
  const activeIds = new Set(state.activeAtomicGroupIds)

  return [...groups.values()].map((group) => {
    const rollup = aggregateGroup(group.id, group.seed_group_ids, units)
    let status: AtomicGroupStatus
    if (!terminal && activeIds.has(group.id)) {
      status = 'Running'
    } else if (rollup.completed >= rollup.planned && rollup.planned > 0) {
      status = 'Completed'
    } else if (terminal) {
      status = 'Incomplete'
    } else {
      status = 'Pending'
    }
    return {
      id: group.id,
      atomicAttackName: group.atomic_attack_name,
      displayGroup: group.display_group,
      status,
      ...rollup,
    }
  }).sort((left, right) => {
    const statusDifference = ATOMIC_STATUS_ORDER[left.status] - ATOMIC_STATUS_ORDER[right.status]
    if (statusDifference !== 0) {
      return statusDifference
    }
    return left.displayGroup.localeCompare(right.displayGroup)
      || left.atomicAttackName.localeCompare(right.atomicAttackName)
  })
}

function buildGroupMetadata(state: ScenarioRunProgressState): Map<string, ScenarioRunPlanAtomicGroup> {
  const groups = new Map<string, ScenarioRunPlanAtomicGroup>()
  for (const group of state.plan?.atomic_groups ?? []) {
    groups.set(group.id, { ...group, seed_group_ids: [...new Set(group.seed_group_ids)] })
  }
  for (const result of state.results) {
    const existing = groups.get(result.atomic_group_id)
    if (existing) {
      if (!existing.seed_group_ids.includes(result.seed_group_id)) {
        groups.set(existing.id, {
          ...existing,
          seed_group_ids: [...existing.seed_group_ids, result.seed_group_id],
        })
      }
      continue
    }
    groups.set(result.atomic_group_id, {
      id: result.atomic_group_id,
      atomic_attack_name: result.atomic_attack_name,
      display_group: result.atomic_attack_name || 'Persisted attack group',
      technique_eval_hash: '',
      seed_group_ids: [result.seed_group_id],
    })
  }
  return groups
}

function buildUnitAttempts(results: ScenarioProgressResult[]): Map<string, UnitAttempts> {
  const grouped = new Map<string, ScenarioProgressResult[]>()
  for (const result of results) {
    const key = unitKey(result.atomic_group_id, result.seed_group_id)
    const attempts = grouped.get(key) ?? []
    attempts.push(result)
    grouped.set(key, attempts)
  }

  const units = new Map<string, UnitAttempts>()
  for (const [key, unsortedAttempts] of grouped) {
    const attempts = [...unsortedAttempts].sort(compareAttempts)
    const latestAttempt = attempts[attempts.length - 1]
    let latestNonError: ScenarioProgressResult | null = null
    for (const attempt of attempts) {
      if (attempt.outcome !== 'error') {
        latestNonError = attempt
      }
    }
    units.set(key, {
      atomicGroupId: latestAttempt.atomic_group_id,
      seedGroupId: latestAttempt.seed_group_id,
      attempts,
      latestAttempt,
      latestNonError,
    })
  }
  return units
}

function aggregateGroup(
  atomicGroupId: string,
  seedGroupIds: string[],
  units: Map<string, UnitAttempts>,
): Rollup {
  const relevantUnits = [...new Set(seedGroupIds)]
    .map((seedGroupId) => units.get(unitKey(atomicGroupId, seedGroupId)))
    .filter((unit): unit is UnitAttempts => unit !== undefined)
  return aggregateUnits(relevantUnits, new Set(seedGroupIds).size)
}

function aggregateUnits(units: UnitAttempts[], planned: number): Rollup {
  let completed = 0
  let succeeded = 0
  let errors = 0
  let retries = 0
  for (const unit of units) {
    if (unit.latestNonError) {
      completed += 1
      if (unit.latestNonError.outcome === 'success') {
        succeeded += 1
      }
    }
    errors += unit.attempts.filter((attempt) => attempt.outcome === 'error').length
    retries += Math.max(0, unit.attempts.length - 1)
    retries += unit.attempts.reduce((total, attempt) => total + Math.max(0, attempt.total_retries), 0)
  }
  return {
    completed,
    planned,
    succeeded,
    evaluated: completed,
    successPercent: completed > 0 ? boundedPercent(succeeded, completed) : null,
    errors,
    retries,
  }
}

function buildPlannedUnitKeys(groups: ScenarioRunPlanAtomicGroup[]): Set<string> {
  const keys = new Set<string>()
  for (const group of groups) {
    for (const seedGroupId of group.seed_group_ids) {
      keys.add(unitKey(group.id, seedGroupId))
    }
  }
  return keys
}

function unitKey(atomicGroupId: string, seedGroupId: string): string {
  return `${atomicGroupId}\u0000${seedGroupId}`
}

function compareAttempts(left: ScenarioProgressResult, right: ScenarioProgressResult): number {
  const timestampDifference = Date.parse(left.timestamp) - Date.parse(right.timestamp)
  if (Number.isFinite(timestampDifference) && timestampDifference !== 0) {
    return timestampDifference
  }
  return left.attack_result_id.localeCompare(right.attack_result_id)
}

function boundedPercent(numerator: number, denominator: number): number {
  if (denominator <= 0) {
    return 0
  }
  return Math.min(100, Math.max(0, Math.round((numerator / denominator) * 100)))
}
