import { useEffect, useMemo, useRef, useState } from 'react'

import {
  Badge,
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  MessageBar,
  MessageBarActions,
  MessageBarBody,
  mergeClasses,
  ProgressBar,
  Skeleton,
  SkeletonItem,
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableHeaderCell,
  TableRow,
  Text,
} from '@fluentui/react-components'
import {
  ArrowLeftRegular,
  ArrowSyncRegular,
  CheckmarkCircleRegular,
  DismissCircleRegular,
  ErrorCircleRegular,
  EyeRegular,
  StopRegular,
} from '@fluentui/react-icons'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'

import { useScenarioRunProgress } from '@/hooks/useScenarioRunProgress'
import { useScenarioQueue } from '@/hooks/useScenarioQueue'
import { scenariosApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type {
  ScenarioProgressResult,
  ScenarioRunState,
} from '@/types'
import {
  attackRoutePath,
  routerPathParamValue,
} from '@/utils/routeParams'
import {
  ATTACK_OUTCOME_BADGE_COLORS,
  formatAttackOutcome,
} from '@/utils/attackOutcome'
import {
  getAttemptAccounting,
  getAttemptPresentations,
  getAttemptRollups,
  getAtomicGroupRollups,
  getElapsedMilliseconds,
  getEtaMilliseconds,
  getOverallProgress,
  getSeedGroupRollups,
  isTargetAttackRole,
  isTerminalRunState,
  type AttemptAccounting,
  type ScenarioAttemptRole,
} from '@/utils/scenarioRunProgress'

import { useScenarioRunPageStyles } from './ScenarioRunPage.styles'
import ScenarioQueue from './ScenarioQueue'

const CLOCK_REFRESH_INTERVAL_MS = 1_000
const OBJECTIVE_PREVIEW_LENGTH = 96
const INTERACTIVE_ELEMENT_SELECTOR = 'a, button, input, select, textarea, [role="button"], [role="link"]'

const RUN_BADGE_COLORS: Record<ScenarioRunState, 'informative' | 'brand' | 'success' | 'danger' | 'warning'> = {
  CREATED: 'informative',
  QUEUED: 'informative',
  IN_PROGRESS: 'brand',
  COMPLETED: 'success',
  FAILED: 'danger',
  CANCELLED: 'warning',
}

export default function ScenarioRunPage() {
  const { scenarioResultId: encodedId } = useParams<{ scenarioResultId: string }>()
  return <ScenarioRunPageContent key={encodedId} scenarioResultId={routerPathParamValue(encodedId)} />
}

interface ScenarioRunPageContentProps {
  readonly scenarioResultId: string
}

function ScenarioRunPageContent({ scenarioResultId }: ScenarioRunPageContentProps) {
  const styles = useScenarioRunPageStyles()
  const location = useLocation()
  const navigate = useNavigate()
  const { state, retry, applyRunSummary } = useScenarioRunProgress(scenarioResultId)
  const queue = useScenarioQueue()
  const [nowMilliseconds, setNowMilliseconds] = useState(() => Date.now())
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [cancelError, setCancelError] = useState<string | null>(null)
  const [selectedAttempt, setSelectedAttempt] = useState<ScenarioProgressResult | null>(null)
  const detailsTriggerRef = useRef<HTMLButtonElement | null>(null)
  const navigationState = location.state as {
    fromScenarioHistory?: boolean
    scenarioHistorySearch?: string
    scenarioName?: string
  } | null
  const backPath = navigationState?.fromScenarioHistory
    ? `/scenario-history${navigationState.scenarioHistorySearch ?? ''}`
    : navigationState?.scenarioName
      ? `/scenarios/${encodeURIComponent(navigationState.scenarioName)}`
      : '/scenario-history'
  const backLabel = navigationState?.scenarioName && !navigationState.fromScenarioHistory
    ? 'Back to scenario'
    : 'Back to scenario history'

  const overall = useMemo(() => getOverallProgress(state), [state])
  const attemptRollups = useMemo(() => getAttemptRollups(state), [state])
  const attemptPresentations = useMemo(() => getAttemptPresentations(state), [state])
  const attemptAccounting = useMemo(() => getAttemptAccounting(state), [state])
  const seedGroups = useMemo(() => getSeedGroupRollups(state), [state])
  const atomicGroups = useMemo(() => getAtomicGroupRollups(state), [state])
  const seedObjectives = useMemo(
    () => new Map(state.plan?.seed_groups.map((seed) => [seed.id, seed.objective]) ?? []),
    [state.plan],
  )
  const atomicGroupNames = useMemo(
    () => new Map(atomicGroups.map((group) => [group.id, group.displayGroup])),
    [atomicGroups],
  )

  useEffect(() => {
    if (!state.run || isTerminalRunState(state.run.status)) {
      return
    }
    const timer = setInterval(() => setNowMilliseconds(Date.now()), CLOCK_REFRESH_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [state.run])

  const closeAttemptDetails = (): void => {
    setSelectedAttempt(null)
    requestAnimationFrame(() => detailsTriggerRef.current?.focus())
  }

  const openAttemptDetails = (
    attempt: ScenarioProgressResult,
    trigger: HTMLButtonElement,
  ): void => {
    detailsTriggerRef.current = trigger
    setSelectedAttempt(attempt)
  }

  const handleCancel = async (): Promise<void> => {
    setCancelling(true)
    setCancelError(null)
    try {
      const run = await scenariosApi.cancelRun(scenarioResultId)
      applyRunSummary(run)
      setCancelDialogOpen(false)
    } catch (error: unknown) {
      setCancelError(toApiError(error).detail)
    } finally {
      setCancelling(false)
    }
  }

  if (state.loadStatus === 'loading' && !state.run) {
    return (
      <main className={styles.root} data-testid="scenario-run-page">
        <div className={styles.content}>
          <Link to={backPath} className={styles.backLink}>
            <ArrowLeftRegular /> {backLabel}
          </Link>
          <div className={styles.centeredState} aria-label="Loading scenario run">
            <Skeleton className={styles.loadingBlock}>
              <SkeletonItem size={32} />
              <br />
              <SkeletonItem size={16} />
              <br />
              <SkeletonItem size={96} />
            </Skeleton>
            <Text>Loading scenario run...</Text>
          </div>
        </div>
      </main>
    )
  }

  if (state.loadStatus === 'not-found' && !state.run) {
    return (
      <main className={styles.root} data-testid="scenario-run-page">
        <div className={styles.content}>
          <Link to={backPath} className={styles.backLink}>
            <ArrowLeftRegular /> {backLabel}
          </Link>
          <div className={styles.centeredState}>
            <ErrorCircleRegular fontSize={32} />
            <Text as="h1" size={600} weight="semibold">Scenario run not found</Text>
            <Text>{state.error}</Text>
            <Button className={styles.touchTarget} icon={<ArrowSyncRegular />} onClick={retry}>
              Retry
            </Button>
          </div>
        </div>
      </main>
    )
  }

  if (state.loadStatus === 'error' && !state.run) {
    return (
      <main className={styles.root} data-testid="scenario-run-page">
        <div className={styles.content}>
          <Link to={backPath} className={styles.backLink}>
            <ArrowLeftRegular /> {backLabel}
          </Link>
          <div className={styles.centeredState}>
            <ErrorCircleRegular fontSize={32} />
            <Text as="h1" size={600} weight="semibold">Unable to load scenario run</Text>
            <Text>{state.error}</Text>
            <Button appearance="primary" className={styles.touchTarget} icon={<ArrowSyncRegular />} onClick={retry}>
              Retry
            </Button>
          </div>
        </div>
      </main>
    )
  }

  if (!state.run) {
    return null
  }

  const run = state.run
  const queued = run.status === 'QUEUED'
  const canCancel = run.status === 'CREATED' || queued || run.status === 'IN_PROGRESS'
  const elapsed = getElapsedMilliseconds(run, nowMilliseconds)
  const eta = getEtaMilliseconds(state, nowMilliseconds)
  const progressText = queued
    ? `Queued${run.queue_position ? ` · Position ${run.queue_position}` : ''}`
    : overall.planned === null
    ? `${overall.completed} known completed progress units; planned total unavailable`
    : `${overall.completed} of ${overall.planned} progress units completed`
  const terminal = isTerminalRunState(run.status)
  const targetFacingResults = state.results.filter((result) => (
    isTargetAttackRole(attemptPresentations.get(result.attack_result_id)?.role ?? 'unknown')
  ))
  const supportingResults = state.results.filter((result) => (
    !isTargetAttackRole(attemptPresentations.get(result.attack_result_id)?.role ?? 'unknown')
  ))
  const targetFacingRollups = attemptRollups.filter((rollup) => isTargetAttackRole(rollup.role))
  const supportingResultsLabel = supportingResults.every((result) => {
    const role = attemptPresentations.get(result.attack_result_id)?.role ?? 'unknown'
    return role === 'adaptive_orchestration' || role === 'aggregate_parent'
  })
    ? 'Orchestration results'
    : 'Orchestration and unclassified results'
  const attemptAccountingSection = state.results.length > 0 ? (
    <ObservedAttemptAccounting accounting={attemptAccounting} />
  ) : null

  return (
    <main className={styles.root} data-testid="scenario-run-page">
      <div className={styles.content}>
        <Link to={backPath} className={styles.backLink}>
          <ArrowLeftRegular /> {backLabel}
        </Link>

        <header className={styles.header}>
          <div className={styles.headerIdentity}>
            <div className={styles.titleRow}>
              <Text as="h1" size={700} weight="semibold">
                {run.scenario_registry_name ?? run.scenario_name}
              </Text>
              <Badge
                appearance="filled"
                color={RUN_BADGE_COLORS[run.status]}
                icon={statusIcon(run.status)}
                data-testid="run-state-badge"
              >
                {formatRunState(run.status)}
              </Badge>
            </div>
            {run.scenario_registry_name && run.scenario_registry_name !== run.scenario_name && (
              <Text size={300}>{run.scenario_name}</Text>
            )}
            <Text size={200} className={styles.runId}>
              Run ID: <code>{run.scenario_result_id}</code>
            </Text>
          </div>
          {canCancel && (
            <div className={styles.headerActions}>
              <Button
                appearance="secondary"
                className={mergeClasses(styles.touchTarget, styles.cancelButton, styles.wideButton)}
                icon={<StopRegular />}
                onClick={() => {
                  setCancelError(null)
                  setCancelDialogOpen(true)
                }}
              >
                Cancel run
              </Button>
            </div>
          )}
        </header>

        <div className={styles.metadata} aria-label="Run metadata">
          <div className={styles.metadataItem}>
            <Text size={200} className={styles.metadataLabel}>Scenario version</Text>
            <Text weight="semibold">{run.scenario_version}</Text>
          </div>
          <div className={styles.metadataItem}>
            <Text size={200} className={styles.metadataLabel}>Created</Text>
            <Text weight="semibold">{formatTimestamp(run.created_at)}</Text>
          </div>
          <div className={styles.metadataItem}>
            <Text size={200} className={styles.metadataLabel}>Completed</Text>
            <Text weight="semibold">{run.completed_at ? formatTimestamp(run.completed_at) : 'Not yet'}</Text>
          </div>
          {run.target && (
            <div className={styles.metadataItem}>
              <Text size={200} className={styles.metadataLabel}>Target</Text>
              <Text weight="semibold">{run.target.model_name ?? run.target.target_type}</Text>
              <Text size={200} className={styles.sectionHint}>{run.target.target_type}</Text>
            </div>
          )}
          {run.pyrit_version && (
            <div className={styles.metadataItem}>
              <Text size={200} className={styles.metadataLabel}>PyRIT version</Text>
              <Text weight="semibold">{run.pyrit_version}</Text>
            </div>
          )}
          {queued && (
            <div className={styles.metadataItem}>
              <Text size={200} className={styles.metadataLabel}>Waiting position</Text>
              <Text weight="semibold">{run.queue_position ?? 'Updating'}</Text>
            </div>
          )}
        </div>

        <section className={styles.section} aria-labelledby="run-configuration-heading">
          <div className={styles.sectionHeading}>
            <Text as="h2" id="run-configuration-heading" size={500} weight="semibold">
              Run configuration
            </Text>
            <Text className={styles.sectionHint}>Persisted, secret-free settings for this run.</Text>
          </div>
          <div className={styles.summaryGrid}>
            <ConfigurationItem
              label="Configured techniques"
              value={(run.techniques_used?.length ?? 0) > 0 ? run.techniques_used?.join(', ') ?? '' : 'Unavailable'}
            />
            <ConfigurationItem
              label="Datasets"
              value={(run.datasets_used?.length ?? 0) > 0 ? run.datasets_used?.join(', ') ?? '' : 'Unavailable'}
            />
            <ConfigurationItem
              label="Scenario parameters"
              value={formatConfiguration(run.scenario_parameters ?? {})}
            />
            <ConfigurationItem
              label="Labels"
              value={formatConfiguration(run.labels ?? {})}
            />
            {run.target?.endpoint && <ConfigurationItem label="Target endpoint" value={run.target.endpoint} />}
            {run.target?.identifier_hash && (
              <ConfigurationItem label="Target identifier" value={run.target.identifier_hash} />
            )}
          </div>
        </section>

        {state.stale && (
          <MessageBar intent="warning">
            <MessageBarBody>
              Live updates paused. Showing the last successfully loaded progress. {state.error}
            </MessageBarBody>
            <MessageBarActions>
              <Button className={styles.touchTarget} icon={<ArrowSyncRegular />} onClick={retry}>
                Retry
              </Button>
            </MessageBarActions>
          </MessageBar>
        )}

        <ScenarioQueue
          snapshot={queue.snapshot}
          loading={queue.loading}
          stale={queue.stale}
          error={queue.error}
          currentScenarioResultId={scenarioResultId}
        />

        {(state.overloadSummaries?.length ?? 0) > 0 && (
          <MessageBar intent="warning" data-testid="scenario-overload-warning">
            <MessageBarBody>
              Recent target overload detected: {formatOverloadSummaries(state.overloadSummaries ?? [])}.
              {' '}PyRIT is retrying these requests without adaptive throttling; concurrency is not automatically reduced yet.
            </MessageBarBody>
          </MessageBar>
        )}

        {run.status === 'FAILED' && (
          <MessageBar intent="error">
            <MessageBarBody>
              This run ended before all planned progress units completed. Persisted result records remain available below.
            </MessageBarBody>
          </MessageBar>
        )}

        {!state.planComplete && (
          <MessageBar intent="info">
            <MessageBarBody>
              This legacy run has no complete persisted progress plan. Known groups and attempts are shown, but planned totals and ETA are unavailable.
            </MessageBarBody>
          </MessageBar>
        )}

        {terminal && attemptAccountingSection}

        <section className={styles.section} aria-labelledby="overall-progress-heading">
          <div className={styles.sectionHeading}>
            <Text as="h2" id="overall-progress-heading" size={500} weight="semibold">
              Overall progress
            </Text>
            <Text className={styles.sectionHint}>{progressText}</Text>
          </div>
          {queued ? (
            <div className={styles.progressSurface} data-testid="queued-run-progress">
              <div className={styles.progressPrimary}>
                <Text size={500} weight="semibold">
                  Queued{run.queue_position ? ` · Position ${run.queue_position}` : ''}
                </Text>
                <Text className={styles.sectionHint}>
                  {run.active_scenario_result_id
                    ? `Waiting for active run ${run.active_scenario_result_id} to finish.`
                    : 'Waiting for the scheduler to start this run.'}
                </Text>
              </div>
              <div className={styles.metric}>
                <Text size={200} className={styles.metricLabel}>Execution progress</Text>
                <Text weight="semibold">Not started</Text>
              </div>
              <div className={styles.metric}>
                <Text size={200} className={styles.metricLabel}>Estimated remaining</Text>
                <Text weight="semibold">Available after start</Text>
              </div>
            </div>
          ) : (
          <div className={styles.progressSurface}>
            <div className={styles.progressPrimary}>
              <div className={styles.progressText}>
                <Text weight="semibold">{progressText}</Text>
                {overall.percent !== null && <Text weight="semibold">{overall.percent}%</Text>}
              </div>
              {overall.percent !== null ? (
                <ProgressBar
                  value={overall.percent / 100}
                  aria-label="Overall scenario run progress"
                  aria-valuetext={progressText}
                />
              ) : (
                <Text className={styles.sectionHint}>Progress percentage unavailable</Text>
              )}
            </div>
            <div className={styles.metric}>
              <Text size={200} className={styles.metricLabel}>Elapsed</Text>
              <Text size={500} weight="semibold" className={styles.metricValue}>
                {formatDuration(elapsed)}
              </Text>
            </div>
            <div className={styles.metric}>
              <Text size={200} className={styles.metricLabel}>Estimated remaining</Text>
              <Text size={500} weight="semibold" className={styles.metricValue}>
                {eta === null ? 'Unavailable' : formatDuration(eta)}
              </Text>
            </div>
          </div>
          )}
          <span className={styles.liveStatus} aria-live="polite">
            {queued ? progressText : isTerminalRunState(run.status) ? `Run ${formatRunState(run.status)}` : ''}
          </span>
        </section>

        {!terminal && attemptAccountingSection}

        <section className={styles.section} aria-labelledby="attempt-summary-heading">
          <div className={styles.sectionHeading}>
            <Text as="h2" id="attempt-summary-heading" size={500} weight="semibold">
              Result summary
            </Text>
            <Text className={styles.sectionHint}>Target-facing attacks grouped by their execution role.</Text>
          </div>
          {targetFacingRollups.length === 0 ? (
            <EmptyState text="Attack roles will appear when the first target-facing result is persisted." />
          ) : (
            <div className={styles.summaryGrid}>
              {targetFacingRollups.map((attempt) => (
                <article key={attempt.id} className={styles.summaryItem}>
                  <div className={styles.summaryTitle}>
                    <Text as="h3" size={400} weight="semibold">{attempt.label}</Text>
                    <Text weight="semibold">
                      {attempt.persistedAttempts} {attempt.persistedAttempts === 1 ? 'result' : 'results'}
                    </Text>
                  </div>
                  <Text size={200} className={styles.sectionHint}>
                    {formatAttemptRole(attempt.role)}
                  </Text>
                  <div className={styles.summaryStats}>
                    <Metric label="Succeeded" value={String(attempt.succeeded)} />
                    <Metric label="Errors" value={String(attempt.errors)} />
                    <Metric label="Retries" value={String(attempt.retries)} />
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className={styles.section} aria-labelledby="atomic-groups-heading">
          <div className={styles.sectionHeading}>
            <Text as="h2" id="atomic-groups-heading" size={500} weight="semibold">
              Atomic attack groups
            </Text>
            <Text className={styles.sectionHint}>Running groups are listed first.</Text>
          </div>
          {atomicGroups.length === 0 ? (
            <EmptyState text="No atomic attack groups have been persisted yet." />
          ) : (
            <div className={styles.tableScroll}>
              <Table size="small" className={styles.table} aria-label="Atomic attack groups">
                <TableHeader>
                  <TableRow>
                    <TableHeaderCell>Status</TableHeaderCell>
                    <TableHeaderCell>Attack group</TableHeaderCell>
                    <TableHeaderCell>Attack</TableHeaderCell>
                    <TableHeaderCell>Progress units</TableHeaderCell>
                    <TableHeaderCell>Persisted result records</TableHeaderCell>
                    <TableHeaderCell>Target-facing attacks</TableHeaderCell>
                    <TableHeaderCell>Success</TableHeaderCell>
                    <TableHeaderCell>Errors</TableHeaderCell>
                    <TableHeaderCell>Retries</TableHeaderCell>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {atomicGroups.map((group) => (
                    <TableRow key={group.id}>
                      <TableCell><AtomicStatusBadge status={group.status} /></TableCell>
                      <TableCell>{group.displayGroup}</TableCell>
                      <TableCell>{group.atomicAttackName || 'Persisted attack'}</TableCell>
                      <TableCell className={styles.nowrap}>
                        {formatCompletion(group.completed, group.planned, state.planComplete)}
                      </TableCell>
                      <TableCell>{group.persistedAttempts}</TableCell>
                      <TableCell>{group.attackAttempts}</TableCell>
                      <TableCell className={styles.nowrap}>
                        {formatSuccess(group.succeeded, group.evaluated, group.successPercent)}
                      </TableCell>
                      <TableCell>{group.errors}</TableCell>
                      <TableCell>{group.retries}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </section>

        <section className={styles.section} aria-labelledby="objectives-heading">
          <div className={styles.sectionHeading}>
            <Text as="h2" id="objectives-heading" size={500} weight="semibold">
              Objectives
            </Text>
            <Text className={styles.sectionHint}>Target-facing attacks and aggregate parent records grouped by objective.</Text>
          </div>
          {seedGroups.length === 0 ? (
            <EmptyState text="No objectives have been persisted yet." />
          ) : (
            <div className={styles.tableScroll}>
              <Table size="small" className={styles.table} aria-label="Objectives">
                <TableHeader>
                  <TableRow>
                    <TableHeaderCell>Objective</TableHeaderCell>
                    <TableHeaderCell>Progress units</TableHeaderCell>
                    <TableHeaderCell>Persisted result records</TableHeaderCell>
                    <TableHeaderCell>Target-facing attacks</TableHeaderCell>
                    <TableHeaderCell>Success</TableHeaderCell>
                    <TableHeaderCell>Errors</TableHeaderCell>
                    <TableHeaderCell>Retries</TableHeaderCell>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {seedGroups.map((seed) => (
                    <TableRow key={seed.id}>
                      <TableCell>
                        <Text className={styles.preview}>{objectivePreview(seed.objective, seed.id)}</Text>
                      </TableCell>
                      <TableCell className={styles.nowrap}>
                        {formatCompletion(seed.completed, seed.planned, state.planComplete)}
                      </TableCell>
                      <TableCell>{seed.persistedAttempts}</TableCell>
                      <TableCell>{seed.attackAttempts}</TableCell>
                      <TableCell className={styles.nowrap}>
                        {formatSuccess(seed.succeeded, seed.evaluated, seed.successPercent)}
                      </TableCell>
                      <TableCell>{seed.errors}</TableCell>
                      <TableCell>{seed.retries}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </section>

        <section className={styles.section} aria-labelledby="attempts-heading">
          <div className={styles.sectionHeading}>
            <Text as="h2" id="attempts-heading" size={500} weight="semibold">
              Target-facing attacks
            </Text>
            <Text className={styles.sectionHint}>{targetFacingResults.length} attack results</Text>
          </div>
          {targetFacingResults.length === 0 ? (
            <EmptyState text="This run has not persisted a target-facing attack yet." />
          ) : (
            <div className={styles.tableScroll}>
              <Table size="small" className={styles.attemptsTable} aria-label="Target-facing attacks">
                <TableHeader>
                  <TableRow>
                    <TableHeaderCell>Result record</TableHeaderCell>
                    <TableHeaderCell>Role</TableHeaderCell>
                    <TableHeaderCell>Technique</TableHeaderCell>
                    <TableHeaderCell>Outcome</TableHeaderCell>
                    <TableHeaderCell>Attack group</TableHeaderCell>
                    <TableHeaderCell>Objective</TableHeaderCell>
                    <TableHeaderCell>Execution</TableHeaderCell>
                    <TableHeaderCell>Retries / error</TableHeaderCell>
                    <TableHeaderCell>Timestamp</TableHeaderCell>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[...targetFacingResults].reverse().map((attempt) => {
                    const presentation = attemptPresentations.get(attempt.attack_result_id)
                    const targetFacing = isTargetAttackRole(presentation?.role ?? 'unknown')
                    const attackDestination = targetFacing
                      ? attackRoutePath(attempt.attack_result_id, scenarioResultId)
                      : null
                    return (
                      <TableRow
                        key={attempt.attack_result_id}
                        className={attackDestination ? styles.clickableAttemptRow : undefined}
                        tabIndex={attackDestination ? 0 : undefined}
                        aria-label={attackDestination ? `Open attack ${attempt.attack_result_id}` : undefined}
                        onClick={attackDestination ? (event) => {
                          if (!shouldIgnoreAttemptRowClick(event)) {
                            navigate(attackDestination)
                          }
                        } : undefined}
                        onKeyDown={attackDestination ? (event) => {
                          if (
                            (event.key === 'Enter' || event.key === ' ')
                            && !hasActivationModifier(event)
                            && !isInteractiveTarget(event.target)
                          ) {
                            event.preventDefault()
                            navigate(attackDestination)
                          }
                        } : undefined}
                      >
                        <TableCell>
                          {attackDestination ? (
                            <Link
                              className={styles.attackLink}
                              to={attackDestination}
                              aria-label={`Open attack ${attempt.attack_result_id}`}
                              onClick={(event) => event.stopPropagation()}
                            >
                              <Text className={styles.preview} title={attempt.attack_result_id}>
                                {attempt.attack_result_id}
                              </Text>
                            </Link>
                          ) : (
                            <Text className={styles.preview} title={attempt.attack_result_id}>
                              {attempt.attack_result_id}
                            </Text>
                          )}
                        </TableCell>
                        <TableCell>{formatAttemptRole(presentation?.role ?? 'unknown')}</TableCell>
                        <TableCell>{presentation?.techniqueName ?? 'Not applicable'}</TableCell>
                        <TableCell>
                          <Badge appearance="tint" color={ATTACK_OUTCOME_BADGE_COLORS[attempt.outcome]}>
                            {formatAttackOutcome(attempt.outcome)}
                          </Badge>
                        </TableCell>
                        <TableCell>{atomicGroupNames.get(attempt.atomic_group_id) ?? attempt.atomic_attack_name}</TableCell>
                        <TableCell>
                          <Button
                            appearance="subtle"
                            className={styles.objectiveButton}
                            icon={<EyeRegular />}
                            aria-label={`View details for result record ${attempt.attack_result_id}`}
                            onClick={(event) => openAttemptDetails(attempt, event.currentTarget)}
                          >
                            <Text className={styles.preview}>
                              {objectivePreview(seedObjectives.get(attempt.seed_group_id) ?? null, attempt.seed_group_id)}
                            </Text>
                          </Button>
                        </TableCell>
                        <TableCell className={styles.nowrap}>{formatDuration(attempt.execution_time_ms)}</TableCell>
                        <TableCell>
                          {attempt.outcome === 'error'
                            ? attempt.error_message ?? attempt.error_type ?? 'Error'
                            : targetFacing
                              ? `${attempt.total_retries} retries`
                              : 'Not applicable'}
                        </TableCell>
                        <TableCell className={styles.nowrap}>{formatTimestamp(attempt.timestamp)}</TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </div>
          )}
          {supportingResults.length > 0 && (
            <details className={styles.supportingResults}>
              <summary>
                {supportingResultsLabel} ({supportingResults.length})
              </summary>
              <Text className={styles.sectionHint}>
                These persisted records summarize orchestration or cannot be classified as target-facing attacks.
                They are not counted as attacks or retries.
              </Text>
              <div className={styles.tableScroll}>
                <Table size="small" className={styles.attemptsTable} aria-label={supportingResultsLabel}>
                  <TableHeader>
                    <TableRow>
                      <TableHeaderCell>Result record</TableHeaderCell>
                      <TableHeaderCell>Role</TableHeaderCell>
                      <TableHeaderCell>Outcome</TableHeaderCell>
                      <TableHeaderCell>Attack group</TableHeaderCell>
                      <TableHeaderCell>Objective</TableHeaderCell>
                      <TableHeaderCell>Execution</TableHeaderCell>
                      <TableHeaderCell>Timestamp</TableHeaderCell>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[...supportingResults].reverse().map((result) => {
                      const presentation = attemptPresentations.get(result.attack_result_id)
                      return (
                        <TableRow key={result.attack_result_id}>
                          <TableCell>
                            <Text className={styles.preview} title={result.attack_result_id}>
                              {result.attack_result_id}
                            </Text>
                          </TableCell>
                          <TableCell>{formatAttemptRole(presentation?.role ?? 'unknown')}</TableCell>
                          <TableCell>
                            <Badge appearance="tint" color={ATTACK_OUTCOME_BADGE_COLORS[result.outcome]}>
                              {formatAttackOutcome(result.outcome)}
                            </Badge>
                          </TableCell>
                          <TableCell>{atomicGroupNames.get(result.atomic_group_id) ?? result.atomic_attack_name}</TableCell>
                          <TableCell>
                            <Button
                              appearance="subtle"
                              className={styles.objectiveButton}
                              icon={<EyeRegular />}
                              aria-label={`View details for result record ${result.attack_result_id}`}
                              onClick={(event) => openAttemptDetails(result, event.currentTarget)}
                            >
                              <Text className={styles.preview}>
                                {objectivePreview(seedObjectives.get(result.seed_group_id) ?? null, result.seed_group_id)}
                              </Text>
                            </Button>
                          </TableCell>
                          <TableCell className={styles.nowrap}>{formatDuration(result.execution_time_ms)}</TableCell>
                          <TableCell className={styles.nowrap}>{formatTimestamp(result.timestamp)}</TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              </div>
            </details>
          )}
        </section>
      </div>

      <Dialog
        open={cancelDialogOpen}
        onOpenChange={(_, data) => {
          if (!cancelling) {
            setCancelDialogOpen(data.open)
          }
        }}
      >
        <DialogSurface>
          <DialogBody>
            <DialogTitle>Cancel this scenario run?</DialogTitle>
            <DialogContent className={styles.dialogContent}>
              <Text>
                {queued
                  ? 'This run will be removed from the queue and will never execute.'
                  : 'In-flight work will be stopped. Attempts already persisted will remain available in this dashboard.'}
              </Text>
              {cancelError && (
                <MessageBar intent="error">
                  <MessageBarBody>{cancelError}</MessageBarBody>
                </MessageBar>
              )}
            </DialogContent>
            <DialogActions>
              <Button disabled={cancelling} onClick={() => setCancelDialogOpen(false)}>Keep running</Button>
              <Button
                appearance="primary"
                className={styles.cancelButton}
                disabled={cancelling}
                icon={<StopRegular />}
                onClick={() => void handleCancel()}
              >
                {cancelling ? 'Cancelling...' : 'Cancel run'}
              </Button>
            </DialogActions>
          </DialogBody>
        </DialogSurface>
      </Dialog>

      <Dialog
        open={selectedAttempt !== null}
        onOpenChange={(_, data) => {
          if (!data.open) {
            closeAttemptDetails()
          }
        }}
      >
        <DialogSurface>
          <DialogBody>
            <DialogTitle>Result record details</DialogTitle>
            {selectedAttempt && (
              <DialogContent className={styles.dialogContent}>
                <div>
                  <Text size={200} className={styles.metadataLabel}>Objective</Text>
                  <Text as="p" className={styles.objective}>
                    {seedObjectives.get(selectedAttempt.seed_group_id) ?? 'Objective text unavailable for this legacy attempt.'}
                  </Text>
                </div>
                <div className={styles.detailGrid}>
                  <Metric
                    label="Role"
                    value={formatAttemptRole(
                      attemptPresentations.get(selectedAttempt.attack_result_id)?.role ?? 'unknown',
                    )}
                  />
                  <Metric
                    label="Technique"
                    value={attemptPresentations.get(selectedAttempt.attack_result_id)?.techniqueName ?? 'Not applicable'}
                  />
                  <Metric label="Attack result ID" value={selectedAttempt.attack_result_id} />
                  <Metric label="Outcome" value={formatAttackOutcome(selectedAttempt.outcome)} />
                  <Metric label="Attack group" value={atomicGroupNames.get(selectedAttempt.atomic_group_id) ?? selectedAttempt.atomic_attack_name} />
                  <Metric label="Atomic attack" value={selectedAttempt.atomic_attack_name || 'Persisted attack'} />
                  <Metric label="Logical seed group" value={selectedAttempt.seed_group_id} />
                  <Metric label="Execution time" value={formatDuration(selectedAttempt.execution_time_ms)} />
                  <Metric
                    label="Retries"
                    value={isTargetAttackRole(
                      attemptPresentations.get(selectedAttempt.attack_result_id)?.role ?? 'unknown',
                    )
                      ? String(selectedAttempt.total_retries)
                      : 'Not applicable'}
                  />
                  <Metric label="Timestamp" value={formatTimestamp(selectedAttempt.timestamp)} />
                </div>
                {selectedAttempt.outcome === 'error' && (
                  <MessageBar intent="error">
                    <MessageBarBody>
                      {selectedAttempt.error_type ? `${selectedAttempt.error_type}: ` : ''}
                      {selectedAttempt.error_message ?? 'No error detail was persisted.'}
                    </MessageBarBody>
                  </MessageBar>
                )}
              </DialogContent>
            )}
            <DialogActions>
              <Button appearance="primary" onClick={closeAttemptDetails}>Close</Button>
            </DialogActions>
          </DialogBody>
        </DialogSurface>
      </Dialog>
    </main>
  )
}

interface MetricProps {
  readonly label: string
  readonly value: string
}

interface ObservedAttemptAccountingProps {
  readonly accounting: AttemptAccounting
}

function ObservedAttemptAccounting({ accounting }: ObservedAttemptAccountingProps) {
  const styles = useScenarioRunPageStyles()
  const progress = accounting.plannedProgressUnits === null
    ? `${accounting.completedProgressUnits} known completed progress units; planned total unavailable`
    : `${accounting.completedProgressUnits}/${accounting.plannedProgressUnits} planned progress units completed`
  const otherAggregateRecords = accounting.aggregateParentRecords - accounting.adaptiveAggregateParentRecords
  const unclassifiedRecords = accounting.persistedAttempts - accounting.attackAttempts - accounting.aggregateParentRecords
  const persistedComponents = accounting.persistedAttempts === accounting.attackAttempts
    ? []
    : [
        `${accounting.attackAttempts} target-facing attack ${accounting.attackAttempts === 1 ? 'result' : 'results'}`,
        ...(accounting.adaptiveAggregateParentRecords > 0
          ? [`${accounting.adaptiveAggregateParentRecords} Adaptive orchestration ${accounting.adaptiveAggregateParentRecords === 1 ? 'summary' : 'summaries'}`]
          : []),
        ...(otherAggregateRecords > 0
          ? [`${otherAggregateRecords} aggregate parent ${otherAggregateRecords === 1 ? 'record' : 'records'}`]
          : []),
        ...(unclassifiedRecords > 0
          ? [`${unclassifiedRecords} unclassified ${unclassifiedRecords === 1 ? 'record' : 'records'}`]
          : []),
      ]
  const persistedBreakdown = persistedComponents.length > 0 ? `: ${persistedComponents.join(' + ')}` : ''
  const uniformEquation = accounting.uniformTargetAttacksPerObjective !== null
    && accounting.objectiveCount * accounting.uniformTargetAttacksPerObjective === accounting.attackAttempts
  const observedAttackLabel = accounting.uniformTargetAttacksPerObjective === 1
    ? 'observed attack each'
    : 'observed attacks each'
  const targetAttackLabel = accounting.attackAttempts === 1
    ? 'target-facing attack'
    : 'target-facing attacks'

  return (
    <section className={styles.section} aria-labelledby="observed-attempts-heading">
      <div className={styles.sectionHeading}>
        <Text as="h2" id="observed-attempts-heading" size={500} weight="semibold">
          Observed execution accounting
        </Text>
        <Text className={styles.sectionHint}>Observed target-facing attacks lead this summary. Progress and storage details follow.</Text>
      </div>
      <div
        className={styles.accountingSurface}
        role="group"
        aria-label={`${uniformEquation ? `${accounting.objectiveCount} ${accounting.objectiveCount === 1 ? 'objective' : 'objectives'} multiplied by ${accounting.uniformTargetAttacksPerObjective} ${observedAttackLabel} equals ` : ''}${accounting.attackAttempts} ${targetAttackLabel}. ${progress}. ${accounting.persistedAttempts} persisted result records${persistedBreakdown}. ${accounting.retries} actual retries.`}
      >
        <div className={styles.accountingEquation} aria-hidden="true">
          {uniformEquation && (
            <>
              <AccountingOperand
                value={String(accounting.objectiveCount)}
                label={accounting.objectiveCount === 1 ? 'objective' : 'objectives'}
              />
              <Text size={500} weight="semibold" className={styles.accountingOperator}>×</Text>
              <AccountingOperand
                value={String(accounting.uniformTargetAttacksPerObjective)}
                label={observedAttackLabel}
              />
              <Text size={500} weight="semibold" className={styles.accountingOperator}>=</Text>
            </>
          )}
          <AccountingOperand
            value={String(accounting.attackAttempts)}
            label={targetAttackLabel}
            result
          />
        </div>
        {accounting.uniformTargetRoleCounts && (
          <Text className={styles.accountingProvenance}>
            Per objective: {formatRoleBreakdown(accounting.uniformTargetRoleCounts)}.
          </Text>
        )}
        <Text className={styles.sectionHint}>
          {progress} · {accounting.persistedAttempts} persisted result {accounting.persistedAttempts === 1 ? 'record' : 'records'}
          {persistedBreakdown} · {accounting.retries} actual {accounting.retries === 1 ? 'retry' : 'retries'}
        </Text>
      </div>
    </section>
  )
}

interface AccountingOperandProps {
  readonly value: string
  readonly label: string
  readonly result?: boolean
}

function AccountingOperand({ value, label, result = false }: AccountingOperandProps) {
  const styles = useScenarioRunPageStyles()
  return (
    <span className={result ? styles.accountingResult : styles.accountingOperand}>
      <Text size={600} weight="semibold" className={styles.metricValue}>{value}</Text>
      <Text size={200}>{label}</Text>
    </span>
  )
}

interface ConfigurationItemProps {
  readonly label: string
  readonly value: string
}

function ConfigurationItem({ label, value }: ConfigurationItemProps) {
  const styles = useScenarioRunPageStyles()
  return (
    <article className={styles.summaryItem}>
      <Text size={200} className={styles.metadataLabel}>{label}</Text>
      <Text className={styles.objective}>{value}</Text>
    </article>
  )
}

function Metric({ label, value }: MetricProps) {
  const styles = useScenarioRunPageStyles()
  return (
    <div className={styles.metric}>
      <Text size={200} className={styles.metricLabel}>{label}</Text>
      <Text weight="semibold" className={styles.metricValue}>{value}</Text>
    </div>
  )
}

interface EmptyStateProps {
  readonly text: string
}

function EmptyState({ text }: EmptyStateProps) {
  const styles = useScenarioRunPageStyles()
  return (
    <div className={styles.emptyState}>
      <Text>{text}</Text>
    </div>
  )
}

interface AtomicStatusBadgeProps {
  readonly status: 'Running' | 'Pending' | 'Incomplete' | 'Completed'
}

function AtomicStatusBadge({ status }: AtomicStatusBadgeProps) {
  const color = status === 'Running'
    ? 'brand'
    : status === 'Completed'
      ? 'success'
      : status === 'Incomplete'
        ? 'warning'
        : 'informative'
  return <Badge appearance="tint" color={color}>{status}</Badge>
}

function formatRunState(status: ScenarioRunState): string {
  return status.toLowerCase().replace('_', ' ').replace(/^\w/, (letter) => letter.toUpperCase())
}

function formatAttemptRole(role: ScenarioAttemptRole): string {
  switch (role) {
    case 'direct_baseline':
      return 'Direct baseline'
    case 'adaptive_technique':
      return 'Adaptive technique'
    case 'adaptive_orchestration':
      return 'Adaptive orchestration'
    case 'aggregate_parent':
      return 'Aggregate parent'
    case 'attack':
      return 'Attack'
    default:
      return 'Unknown / additional'
  }
}

function formatRoleBreakdown(counts: ReadonlyMap<ScenarioAttemptRole, number>): string {
  const order: ScenarioAttemptRole[] = [
    'direct_baseline',
    'adaptive_technique',
    'attack',
    'adaptive_orchestration',
    'aggregate_parent',
    'unknown',
  ]
  return order.flatMap((role) => {
    const count = counts.get(role) ?? 0
    if (count === 0) {
      return []
    }
    const label = role === 'direct_baseline'
      ? count === 1 ? 'direct baseline' : 'direct baseline attacks'
      : role === 'adaptive_technique'
        ? `Adaptive ${count === 1 ? 'technique' : 'techniques'}`
        : role === 'adaptive_orchestration'
          ? `Adaptive orchestration ${count === 1 ? 'result' : 'results'}`
          : role === 'aggregate_parent'
            ? `aggregate parent ${count === 1 ? 'result' : 'results'}`
          : role === 'attack'
            ? count === 1 ? 'attack' : 'attacks'
            : `additional persisted ${count === 1 ? 'result' : 'results'}`
    return [`${count} ${label}`]
  }).join(' + ')
}

function statusIcon(status: ScenarioRunState): React.ReactElement {
  if (status === 'COMPLETED') {
    return <CheckmarkCircleRegular />
  }
  if (status === 'FAILED') {
    return <DismissCircleRegular />
  }
  if (status === 'CANCELLED') {
    return <StopRegular />
  }
  return <ArrowSyncRegular />
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp)
  if (Number.isNaN(date.getTime())) {
    return 'Unavailable'
  }

  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function formatConfiguration(value: Record<string, unknown>): string {
  const entries = Object.entries(value)
  if (entries.length === 0) {
    return 'None'
  }
  return entries
    .map(([key, item]) => `${key}: ${typeof item === 'string' ? item : JSON.stringify(item)}`)
    .join(', ')
}

function formatDuration(milliseconds: number): string {
  if (!Number.isFinite(milliseconds) || milliseconds < 0) {
    return 'Unavailable'
  }
  const totalSeconds = Math.floor(milliseconds / 1_000)
  const hours = Math.floor(totalSeconds / 3_600)
  const minutes = Math.floor((totalSeconds % 3_600) / 60)
  const seconds = totalSeconds % 60
  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`
  }
  return `${seconds}s`
}

function formatSuccess(succeeded: number, evaluated: number, percent: number | null): string {
  return percent === null ? `${succeeded}/${evaluated} —` : `${succeeded}/${evaluated} (${percent}%)`
}

function formatCompletion(completed: number, planned: number, planComplete: boolean): string {
  return planComplete ? `${completed}/${planned}` : `${completed}/total unavailable`
}

function objectivePreview(objective: string | null, fallbackId: string): string {
  if (!objective) {
    return `Objective unavailable (${fallbackId})`
  }

  if (objective.length <= OBJECTIVE_PREVIEW_LENGTH) {
    return objective
  }
  return `${objective.slice(0, OBJECTIVE_PREVIEW_LENGTH - 1)}…`
}

function formatOverloadSummaries(
  summaries: import('@/types').ScenarioOverloadSummary[],
): string {
  return summaries.map((summary) => {
    const codes = summary.status_codes.join('/')
    return `${formatRole(summary.component_role)} (${summary.count} × HTTP ${codes}, latest ${formatTimestamp(summary.latest_timestamp)})`
  }).join('; ')
}

function formatRole(role: string): string {
  return role.replace(/_/g, ' ').replace(/^\w/, (letter: string) => letter.toUpperCase())
}

function shouldIgnoreAttemptRowClick(event: React.MouseEvent<HTMLTableRowElement>): boolean {
  return event.button !== 0
    || hasActivationModifier(event)
    || isInteractiveTarget(event.target)
}

function hasActivationModifier(
  event: Pick<React.MouseEvent, 'altKey' | 'ctrlKey' | 'metaKey' | 'shiftKey'>
    | Pick<React.KeyboardEvent, 'altKey' | 'ctrlKey' | 'metaKey' | 'shiftKey'>,
): boolean {
  return event.altKey || event.ctrlKey || event.metaKey || event.shiftKey
}

function isInteractiveTarget(target: EventTarget): boolean {
  return target instanceof Element && target.closest(INTERACTIVE_ELEMENT_SELECTOR) !== null
}
