import type {
  ScenarioProgressHeader,
  ScenarioProgressResult,
  ScenarioOverloadSummary,
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
  readonly overloadSummaries: ScenarioOverloadSummary[]
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
  readonly persistedAttempts: number
  readonly attackAttempts: number
}

export interface AttackGroupRollup extends Rollup {
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

export type ScenarioAttemptRole = NonNullable<ScenarioProgressResult['result_kind']>

export interface AttemptPresentation {
  readonly role: ScenarioAttemptRole
  readonly label: string
  readonly techniqueName: string | null
}

export interface AttemptRollup {
  readonly id: string
  readonly role: ScenarioAttemptRole
  readonly label: string
  readonly persistedAttempts: number
  readonly succeeded: number
  readonly errors: number
  readonly retries: number
}

export interface AttemptAccounting {
  readonly objectiveCount: number
  readonly persistedAttempts: number
  readonly attackAttempts: number
  readonly aggregateParentRecords: number
  readonly adaptiveAggregateParentRecords: number
  readonly uniformTargetAttacksPerObjective: number | null
  readonly uniformTargetRoleCounts: ReadonlyMap<ScenarioAttemptRole, number> | null
  readonly completedProgressUnits: number
  readonly plannedProgressUnits: number | null
  readonly retries: number
}

export function isTargetAttackRole(role: ScenarioAttemptRole): boolean {
  return role === 'attack' || role === 'direct_baseline' || role === 'adaptive_technique'
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
  overloadSummaries: [],
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
        started_at: action.run.started_at,
        completed_at: action.run.completed_at,
        pyrit_version: action.run.pyrit_version,
        target: action.run.target,
        techniques_used: action.run.techniques_used,
        datasets_used: action.run.datasets_used ?? [],
        scenario_parameters: action.run.scenario_parameters ?? {},
        labels: action.run.labels,
        queue_position: action.run.queue_position,
        active_scenario_result_id: action.run.active_scenario_result_id,
        overload_summaries: action.run.overload_summaries ?? [],
      },
      activeAtomicGroupIds: [],
      error: null,
      stale: false,
      hasMore: false,
      overloadSummaries: state.overloadSummaries,
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
  const overloadSummaries = mergeOverloadSummaries(
    shouldReset ? [] : state.overloadSummaries,
    action.page.run.overload_summaries ?? [],
  )
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
    overloadSummaries,
  }
}

function mergeOverloadSummaries(
  existing: ScenarioOverloadSummary[],
  incoming: ScenarioOverloadSummary[],
): ScenarioOverloadSummary[] {
  const byRole = new Map(existing.map((summary) => [summary.component_role, summary]))
  for (const summary of incoming) {
    const previous = byRole.get(summary.component_role)
    byRole.set(summary.component_role, previous ? {
      component_role: summary.component_role,
      count: previous.count + summary.count,
      rate_limit_count: previous.rate_limit_count + summary.rate_limit_count,
      server_error_count: previous.server_error_count + summary.server_error_count,
      status_codes: [...new Set([...previous.status_codes, ...summary.status_codes])].sort((left, right) => left - right),
      latest_timestamp: Date.parse(summary.latest_timestamp) >= Date.parse(previous.latest_timestamp)
        ? summary.latest_timestamp
        : previous.latest_timestamp,
    } : summary)
  }
  return [...byRole.values()].sort(
    (left, right) => Date.parse(right.latest_timestamp) - Date.parse(left.latest_timestamp),
  )
}

export function getOverallProgress(state: ScenarioRunProgressState): OverallProgress {
  const presentations = getAttemptPresentations(state)
  const units = buildUnitAttempts(state.results, presentations)
  return calculateOverallProgress(state, units)
}

function calculateOverallProgress(
  state: ScenarioRunProgressState,
  units: ReadonlyMap<string, UnitAttempts>,
): OverallProgress {
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
  if (!run.started_at) {
    return 0
  }
  const started = Date.parse(run.started_at)
  const terminalEnd = run.completed_at ? Date.parse(run.completed_at) : Number.NaN
  const end = isTerminalRunState(run.status) && Number.isFinite(terminalEnd)
    ? terminalEnd
    : nowMilliseconds
  if (!Number.isFinite(started) || !Number.isFinite(end)) {
    return 0
  }
  return Math.max(0, end - started)
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

export function getAttackGroupRollups(state: ScenarioRunProgressState): AttackGroupRollup[] {
  const groupMetadata = buildGroupMetadata(state)
  const presentations = getAttemptPresentations(state)
  const units = buildUnitAttempts(state.results, presentations)
  const rollups = new Map<string, AttackGroupRollup>()

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
      persistedAttempts: 0,
      attackAttempts: 0,
    }
    const groupRollup = aggregateGroup(group.id, group.seed_group_ids, units, presentations)
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
      persistedAttempts: base.persistedAttempts + groupRollup.persistedAttempts,
      attackAttempts: base.attackAttempts + groupRollup.attackAttempts,
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
  const presentations = getAttemptPresentations(state)
  const units = buildUnitAttempts(state.results, presentations)
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
    const rollup = aggregateUnits(relevantUnits, relevantGroups.length, presentations)
    return { id: seedId, objective: objectives.get(seedId) ?? null, ...rollup }
  }).sort((left, right) => {
    const leftLabel = left.objective ?? left.id
    const rightLabel = right.objective ?? right.id
    return leftLabel.localeCompare(rightLabel)
  })
}

export function getAtomicGroupRollups(state: ScenarioRunProgressState): AtomicGroupRollup[] {
  const groups = buildGroupMetadata(state)
  const presentations = getAttemptPresentations(state)
  const units = buildUnitAttempts(state.results, presentations)
  const terminal = state.run ? isTerminalRunState(state.run.status) : false
  const activeIds = new Set(state.activeAtomicGroupIds)

  return [...groups.values()].map((group) => {
    const rollup = aggregateGroup(group.id, group.seed_group_ids, units, presentations)
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

export function getAttemptPresentations(state: ScenarioRunProgressState): Map<string, AttemptPresentation> {
  const groups = buildGroupMetadata(state)
  const adaptiveUnits = new Set(
    state.results
      .filter((result) => result.result_kind === 'adaptive_technique' || Boolean(result.technique_name))
      .map((result) => unitKey(result.atomic_group_id, result.seed_group_id)),
  )
  const presentations = new Map<string, AttemptPresentation>()
  for (const result of state.results) {
    const group = groups.get(result.atomic_group_id)
    let role = result.result_kind ?? 'unknown'
    const techniqueName = result.technique_name?.trim() || null
    if (role === 'unknown') {
      if (techniqueName) {
        role = 'adaptive_technique'
      } else if (group?.group_kind === 'direct_baseline' || group?.atomic_attack_name === 'baseline') {
        role = 'direct_baseline'
      } else if (
        group?.group_kind === 'adaptive'
        || adaptiveUnits.has(unitKey(result.atomic_group_id, result.seed_group_id))
      ) {
        role = 'adaptive_orchestration'
      } else if (group?.group_kind === 'attack') {
        role = 'attack'
      }
    }
    const label = role === 'direct_baseline'
      ? 'Direct baseline'
      : role === 'adaptive_technique'
        ? techniqueName ?? 'Adaptive technique'
        : role === 'adaptive_orchestration'
          ? 'Adaptive orchestration'
          : role === 'aggregate_parent'
            ? adaptiveUnits.has(unitKey(result.atomic_group_id, result.seed_group_id))
              ? 'Adaptive aggregate parent'
              : 'Aggregate parent'
          : role === 'attack'
            ? group?.display_group || result.atomic_attack_name || 'Attack'
            : 'Additional persisted result'
    presentations.set(result.attack_result_id, { role, label, techniqueName })
  }
  return presentations
}

export function getAttemptRollups(state: ScenarioRunProgressState): AttemptRollup[] {
  const presentations = getAttemptPresentations(state)
  const rollups = new Map<string, AttemptRollup>()
  const targetAttemptsByUnit = new Map<string, number>()
  for (const result of state.results) {
    const presentation = presentations.get(result.attack_result_id)
    if (!presentation) {
      continue
    }
    const id = `${presentation.role}\u0000${presentation.label}`
    const existing = rollups.get(id) ?? {
      id,
      role: presentation.role,
      label: presentation.label,
      persistedAttempts: 0,
      succeeded: 0,
      errors: 0,
      retries: 0,
    }
    rollups.set(id, {
      ...existing,
      persistedAttempts: existing.persistedAttempts + 1,
      succeeded: existing.succeeded + (result.outcome === 'success' ? 1 : 0),
      errors: existing.errors + (result.outcome === 'error' ? 1 : 0),
      retries: existing.retries + (
        isTargetAttackRole(presentation.role)
          ? Math.max(0, result.total_retries) + registerAdditionalAttempt(
            targetAttemptsByUnit,
            unitKey(result.atomic_group_id, result.seed_group_id),
          )
          : 0
      ),
    })
  }
  const roleOrder: Record<ScenarioAttemptRole, number> = {
    direct_baseline: 0,
    adaptive_technique: 1,
    attack: 2,
    adaptive_orchestration: 3,
    aggregate_parent: 4,
    unknown: 5,
  }
  return [...rollups.values()].sort(
    (left, right) => roleOrder[left.role] - roleOrder[right.role] || left.label.localeCompare(right.label),
  )
}

export function getAttemptAccounting(state: ScenarioRunProgressState): AttemptAccounting {
  const presentations = getAttemptPresentations(state)
  const modernAdaptiveGroupIds = new Set(
    state.plan?.atomic_groups
      .filter((group) => group.group_kind === 'adaptive')
      .map((group) => group.id) ?? [],
  )
  const legacyAdaptiveResultKeys = new Set<string>()
  for (const result of state.results) {
    const role = presentations.get(result.attack_result_id)?.role ?? 'unknown'
    if (role === 'adaptive_technique' || role === 'adaptive_orchestration') {
      legacyAdaptiveResultKeys.add(`${result.atomic_group_id}\0${result.seed_group_id}`)
    }
  }
  const objectiveIds = new Set(state.plan?.seed_groups.map((seed) => seed.id) ?? [])
  for (const result of state.results) {
    objectiveIds.add(result.seed_group_id)
  }
  const attemptsByObjective = new Map<string, ScenarioProgressResult[]>()
  for (const objectiveId of objectiveIds) {
    attemptsByObjective.set(objectiveId, [])
  }
  for (const result of state.results) {
    attemptsByObjective.get(result.seed_group_id)?.push(result)
  }
  const targetAttemptsByObjective = [...attemptsByObjective.values()].map((attempts) => (
    attempts.filter((attempt) => {
      const role = presentations.get(attempt.attack_result_id)?.role ?? 'unknown'
      return isTargetAttackRole(role)
    })
  ))
  const observedCounts = targetAttemptsByObjective.map((attempts) => attempts.length)
  const uniformTargetAttacksPerObjective = observedCounts.length > 0
    && observedCounts[0] > 0
    && observedCounts.every((count) => count === observedCounts[0])
    ? observedCounts[0]
    : null
  const roleCounts = targetAttemptsByObjective.map((attempts) => {
    const counts = new Map<ScenarioAttemptRole, number>()
    for (const attempt of attempts) {
      const role = presentations.get(attempt.attack_result_id)?.role ?? 'unknown'
      counts.set(role, (counts.get(role) ?? 0) + 1)
    }
    return counts
  })
  const firstRoleSignature = roleCounts[0] ? roleCountSignature(roleCounts[0]) : null
  const uniformTargetRoleCounts = firstRoleSignature !== null
    && roleCounts[0].size > 0
    && roleCounts.every((counts) => roleCountSignature(counts) === firstRoleSignature)
    ? roleCounts[0]
    : null
  const units = buildUnitAttempts(state.results, presentations)
  const overall = calculateOverallProgress(state, units)
  return {
    objectiveCount: objectiveIds.size,
    persistedAttempts: state.results.length,
    attackAttempts: state.results.filter((result) => {
      const role = presentations.get(result.attack_result_id)?.role ?? 'unknown'
      return isTargetAttackRole(role)
    }).length,
    aggregateParentRecords: state.results.filter((result) => (
      presentations.get(result.attack_result_id)?.role === 'adaptive_orchestration'
      || presentations.get(result.attack_result_id)?.role === 'aggregate_parent'
    )).length,
    adaptiveAggregateParentRecords: state.results.filter((result) => {
      const presentation = presentations.get(result.attack_result_id)
      return presentation?.role === 'adaptive_orchestration'
        || (
          presentation?.role === 'aggregate_parent'
          && (
            modernAdaptiveGroupIds.has(result.atomic_group_id)
            || legacyAdaptiveResultKeys.has(`${result.atomic_group_id}\0${result.seed_group_id}`)
          )
        )
    }).length,
    uniformTargetAttacksPerObjective,
    uniformTargetRoleCounts,
    completedProgressUnits: overall.completed,
    plannedProgressUnits: overall.planned,
    retries: [...units.values()].reduce(
      (total, unit) => total + unitRetryWork(unit, presentations),
      0,
    ),
  }
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

function buildUnitAttempts(
  results: ScenarioProgressResult[],
  presentations: ReadonlyMap<string, AttemptPresentation>,
): Map<string, UnitAttempts> {
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
      const role = presentations.get(attempt.attack_result_id)?.role ?? 'unknown'
      if (isTargetAttackRole(role) && attempt.outcome !== 'error') {
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
  presentations: Map<string, AttemptPresentation>,
): Rollup {
  const relevantUnits = [...new Set(seedGroupIds)]
    .map((seedGroupId) => units.get(unitKey(atomicGroupId, seedGroupId)))
    .filter((unit): unit is UnitAttempts => unit !== undefined)
  return aggregateUnits(relevantUnits, new Set(seedGroupIds).size, presentations)
}

function aggregateUnits(
  units: UnitAttempts[],
  planned: number,
  presentations: Map<string, AttemptPresentation>,
): Rollup {
  let completed = 0
  let succeeded = 0
  let errors = 0
  let retries = 0
  let persistedAttempts = 0
  let attackAttempts = 0
  for (const unit of units) {
    if (unit.latestNonError) {
      completed += 1
      if (unit.latestNonError.outcome === 'success') {
        succeeded += 1
      }
    }
    errors += unit.attempts.filter((attempt) => {
      const role = presentations.get(attempt.attack_result_id)?.role ?? 'unknown'
      return isTargetAttackRole(role) && attempt.outcome === 'error'
    }).length
    retries += unitRetryWork(unit, presentations)
    persistedAttempts += unit.attempts.length
    attackAttempts += unit.attempts.filter((attempt) => {
      const role = presentations.get(attempt.attack_result_id)?.role ?? 'unknown'
      return isTargetAttackRole(role)
    }).length
  }

  return {
    completed,
    planned,
    succeeded,
    evaluated: completed,
    successPercent: completed > 0 ? boundedPercent(succeeded, completed) : null,
    errors,
    retries,
    persistedAttempts,
    attackAttempts,
  }
}

function unitRetryWork(
  unit: UnitAttempts,
  presentations: ReadonlyMap<string, AttemptPresentation>,
): number {
  const targetAttempts = unit.attempts.filter((attempt) => {
    const role = presentations.get(attempt.attack_result_id)?.role ?? 'unknown'
    return isTargetAttackRole(role)
  })
  const innerRetries = targetAttempts.reduce(
    (total, attempt) => total + Math.max(0, attempt.total_retries),
    0,
  )
  return innerRetries + Math.max(0, targetAttempts.length - 1)
}

function registerAdditionalAttempt(counts: Map<string, number>, key: string): number {
  const previousCount = counts.get(key) ?? 0
  counts.set(key, previousCount + 1)
  return previousCount > 0 ? 1 : 0
}

function roleCountSignature(counts: ReadonlyMap<ScenarioAttemptRole, number>): string {
  return [...counts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([role, count]) => `${role}:${count}`)
    .join('|')
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
