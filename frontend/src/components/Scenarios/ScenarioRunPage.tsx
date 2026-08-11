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
import { Link, useNavigate, useParams } from 'react-router'

import { useScenarioRunProgress } from '@/hooks/useScenarioRunProgress'
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
  getAtomicGroupRollups,
  getElapsedMilliseconds,
  getEtaMilliseconds,
  getOverallProgress,
  getSeedGroupRollups,
  getTechniqueRollups,
  isTerminalRunState,
} from '@/utils/scenarioRunProgress'

import { useScenarioRunPageStyles } from './ScenarioRunPage.styles'

const CLOCK_REFRESH_INTERVAL_MS = 1_000
const OBJECTIVE_PREVIEW_LENGTH = 96
const INTERACTIVE_ELEMENT_SELECTOR = 'a, button, input, select, textarea, [role="button"], [role="link"]'

const RUN_BADGE_COLORS: Record<ScenarioRunState, 'informative' | 'brand' | 'success' | 'danger' | 'warning'> = {
  CREATED: 'informative',
  IN_PROGRESS: 'brand',
  COMPLETED: 'success',
  FAILED: 'danger',
  CANCELLED: 'warning',
}

const OUTCOME_BADGE_COLORS: Record<ScenarioProgressResult['outcome'], 'success' | 'danger' | 'warning' | 'informative'> = {
  success: 'success',
  failure: 'danger',
  error: 'warning',
  undetermined: 'informative',
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
  const navigate = useNavigate()
  const { state, retry, applyRunSummary } = useScenarioRunProgress(scenarioResultId)
  const [nowMilliseconds, setNowMilliseconds] = useState(() => Date.now())
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [cancelError, setCancelError] = useState<string | null>(null)
  const [selectedAttempt, setSelectedAttempt] = useState<ScenarioProgressResult | null>(null)
  const detailsTriggerRef = useRef<HTMLButtonElement | null>(null)

  const overall = useMemo(() => getOverallProgress(state), [state])
  const techniques = useMemo(() => getTechniqueRollups(state), [state])
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
          <Link to="/scenarios" className={styles.backLink}>
            <ArrowLeftRegular /> Back to scenarios
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
          <Link to="/scenarios" className={styles.backLink}>
            <ArrowLeftRegular /> Back to scenarios
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
          <Link to="/scenarios" className={styles.backLink}>
            <ArrowLeftRegular /> Back to scenarios
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
  const canCancel = run.status === 'CREATED' || run.status === 'IN_PROGRESS'
  const elapsed = getElapsedMilliseconds(run, nowMilliseconds)
  const eta = getEtaMilliseconds(state, nowMilliseconds)
  const progressText = overall.planned === null
    ? `${overall.completed} known completed units; planned total unavailable`
    : `${overall.completed} of ${overall.planned} executable units completed`

  return (
    <main className={styles.root} data-testid="scenario-run-page">
      <div className={styles.content}>
        <Link to="/scenarios" className={styles.backLink}>
          <ArrowLeftRegular /> Back to scenarios
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
                className={mergeClasses(styles.touchTarget, styles.wideButton)}
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
        </div>

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

        {run.status === 'FAILED' && (
          <MessageBar intent="error">
            <MessageBarBody>
              This run ended before all planned executable units completed. Persisted attempts remain available below.
            </MessageBarBody>
          </MessageBar>
        )}

        {!state.planComplete && (
          <MessageBar intent="info">
            <MessageBarBody>
              This legacy run has no complete persisted execution plan. Known groups and attempts are shown, but planned totals and ETA are unavailable.
            </MessageBarBody>
          </MessageBar>
        )}

        <section className={styles.section} aria-labelledby="overall-progress-heading">
          <div className={styles.sectionHeading}>
            <Text as="h2" id="overall-progress-heading" size={500} weight="semibold">
              Overall progress
            </Text>
            <Text className={styles.sectionHint}>{progressText}</Text>
          </div>
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
          <span className={styles.liveStatus} aria-live="polite">
            {isTerminalRunState(run.status) ? `Run ${formatRunState(run.status)}` : ''}
          </span>
        </section>

        <section className={styles.section} aria-labelledby="technique-summary-heading">
          <div className={styles.sectionHeading}>
            <Text as="h2" id="technique-summary-heading" size={500} weight="semibold">
              Technique summary
            </Text>
            <Text className={styles.sectionHint}>Success is measured over evaluated non-error units.</Text>
          </div>
          {techniques.length === 0 ? (
            <EmptyState text="Technique results will appear when the first attack attempt is persisted." />
          ) : (
            <div className={styles.summaryGrid}>
              {techniques.map((technique) => (
                <article key={technique.id} className={styles.summaryItem}>
                  <div className={styles.summaryTitle}>
                    <Text as="h3" size={400} weight="semibold">{technique.displayGroup}</Text>
                    <Text weight="semibold">{formatSuccess(technique.succeeded, technique.evaluated, technique.successPercent)}</Text>
                  </div>
                  <Text size={200} className={styles.sectionHint}>
                    {technique.atomicAttackNames.join(', ')}
                  </Text>
                  <div className={styles.summaryStats}>
                    <Metric
                      label="Progress"
                      value={formatCompletion(technique.completed, technique.planned, state.planComplete)}
                    />
                    <Metric label="Errors" value={String(technique.errors)} />
                    <Metric label="Retries" value={String(technique.retries)} />
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
                    <TableHeaderCell>Display group</TableHeaderCell>
                    <TableHeaderCell>Attack</TableHeaderCell>
                    <TableHeaderCell>Completed</TableHeaderCell>
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

        <section className={styles.section} aria-labelledby="seed-groups-heading">
          <div className={styles.sectionHeading}>
            <Text as="h2" id="seed-groups-heading" size={500} weight="semibold">
              Logical seed groups
            </Text>
            <Text className={styles.sectionHint}>Aggregated across techniques.</Text>
          </div>
          {seedGroups.length === 0 ? (
            <EmptyState text="No logical seed groups have been persisted yet." />
          ) : (
            <div className={styles.tableScroll}>
              <Table size="small" className={styles.table} aria-label="Logical seed groups">
                <TableHeader>
                  <TableRow>
                    <TableHeaderCell>Objective</TableHeaderCell>
                    <TableHeaderCell>Completed</TableHeaderCell>
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
              Persisted attack attempts
            </Text>
            <Text className={styles.sectionHint}>{state.results.length} attempts</Text>
          </div>
          {state.results.length === 0 ? (
            <EmptyState text="This run has not persisted an attack attempt yet." />
          ) : (
            <div className={styles.tableScroll}>
              <Table size="small" className={styles.attemptsTable} aria-label="Persisted attack attempts">
                <TableHeader>
                  <TableRow>
                    <TableHeaderCell>Attack</TableHeaderCell>
                    <TableHeaderCell>Outcome</TableHeaderCell>
                    <TableHeaderCell>Group</TableHeaderCell>
                    <TableHeaderCell>Seed</TableHeaderCell>
                    <TableHeaderCell>Objective</TableHeaderCell>
                    <TableHeaderCell>Execution</TableHeaderCell>
                    <TableHeaderCell>Retries / error</TableHeaderCell>
                    <TableHeaderCell>Timestamp</TableHeaderCell>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[...state.results].reverse().map((attempt) => {
                    const attackDestination = attackRoutePath(
                      attempt.attack_result_id,
                      scenarioResultId,
                    )
                    return (
                      <TableRow
                        key={attempt.attack_result_id}
                        className={styles.clickableAttemptRow}
                        tabIndex={0}
                        aria-label={`Open attack ${attempt.attack_result_id}`}
                        onClick={(event) => {
                          if (!shouldIgnoreAttemptRowClick(event)) {
                            navigate(attackDestination)
                          }
                        }}
                        onKeyDown={(event) => {
                          if (
                            (event.key === 'Enter' || event.key === ' ')
                            && !hasActivationModifier(event)
                            && !isInteractiveTarget(event.target)
                          ) {
                            event.preventDefault()
                            navigate(attackDestination)
                          }
                        }}
                      >
                        <TableCell>
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
                        </TableCell>
                        <TableCell>
                          <Badge appearance="tint" color={OUTCOME_BADGE_COLORS[attempt.outcome]}>
                            {formatOutcome(attempt.outcome)}
                          </Badge>
                        </TableCell>
                        <TableCell>{atomicGroupNames.get(attempt.atomic_group_id) ?? attempt.atomic_attack_name}</TableCell>
                        <TableCell><Text className={styles.preview}>{attempt.seed_group_id}</Text></TableCell>
                        <TableCell>
                          <Button
                            appearance="subtle"
                            className={styles.objectiveButton}
                            icon={<EyeRegular />}
                            aria-label={`View details for attack attempt ${attempt.attack_result_id}`}
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
                            : `${attempt.total_retries} retries`}
                        </TableCell>
                        <TableCell className={styles.nowrap}>{formatTimestamp(attempt.timestamp)}</TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </div>
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
                In-flight work will be stopped. Attempts already persisted will remain available in this dashboard.
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
            <DialogTitle>Attack attempt details</DialogTitle>
            {selectedAttempt && (
              <DialogContent className={styles.dialogContent}>
                <div>
                  <Text size={200} className={styles.metadataLabel}>Objective</Text>
                  <Text as="p" className={styles.objective}>
                    {seedObjectives.get(selectedAttempt.seed_group_id) ?? 'Objective text unavailable for this legacy attempt.'}
                  </Text>
                </div>
                <div className={styles.detailGrid}>
                  <Metric label="Attack result ID" value={selectedAttempt.attack_result_id} />
                  <Metric label="Outcome" value={formatOutcome(selectedAttempt.outcome)} />
                  <Metric label="Display group" value={atomicGroupNames.get(selectedAttempt.atomic_group_id) ?? selectedAttempt.atomic_attack_name} />
                  <Metric label="Atomic attack" value={selectedAttempt.atomic_attack_name || 'Persisted attack'} />
                  <Metric label="Logical seed group" value={selectedAttempt.seed_group_id} />
                  <Metric label="Execution time" value={formatDuration(selectedAttempt.execution_time_ms)} />
                  <Metric label="Retries" value={String(selectedAttempt.total_retries)} />
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

function formatOutcome(outcome: ScenarioProgressResult['outcome']): string {
  return outcome.replace(/^\w/, (letter) => letter.toUpperCase())
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
