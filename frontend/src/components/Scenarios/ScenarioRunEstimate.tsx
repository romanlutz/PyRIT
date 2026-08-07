import { Badge, Spinner, Text } from '@fluentui/react-components'

import type {
  ScenarioRunEstimate,
  ScenarioRunEstimateComponent,
  ScenarioRunEstimateState,
} from '@/types'

import { useScenarioRunEstimateStyles } from './ScenarioRunEstimate.styles'

interface ScenarioRunEstimateSummaryProps {
  state: ScenarioRunEstimateState
  primaryCount?: number
  unitLabels?: ScenarioRunEstimateUnitLabels
  supportingText?: string
}

interface ScenarioRunEstimateDetailsProps {
  state: ScenarioRunEstimateState
  idPrefix?: string
  primaryCount?: number
  unitLabels?: ScenarioRunEstimateUnitLabels
  supportingText?: string
}

interface ScenarioRunEstimateUnitLabels {
  singular: string
  plural: string
}

const DEFAULT_UNIT_LABELS: ScenarioRunEstimateUnitLabels = {
  singular: 'planned attack',
  plural: 'planned attacks',
}

function stateEstimate(state: ScenarioRunEstimateState): ScenarioRunEstimate | undefined {
  switch (state.status) {
    case 'available':
    case 'conditional':
    case 'refreshing':
    case 'stale':
      return state.estimate
    default:
      return undefined
  }
}

function scopeLabel(state: ScenarioRunEstimateState): string {
  const scope = state.status === 'loading' || state.status === 'unavailable'
    ? state.scope
    : state.estimate.scope
  return scope === 'default' ? 'Default configuration' : 'Current configuration'
}

function statusLabel(state: ScenarioRunEstimateState): string | null {
  switch (state.status) {
    case 'loading':
      return 'Loading estimate'
    case 'available':
      return null
    case 'conditional':
      return null
    case 'refreshing':
      return 'Updating estimate'
    case 'stale':
      return 'Previous estimate'
    case 'unavailable':
      return 'Estimate unavailable'
  }
}

function statusColor(state: ScenarioRunEstimateState): 'brand' | 'warning' | 'subtle' {
  switch (state.status) {
    case 'available':
    case 'refreshing':
      return 'brand'
    case 'conditional':
    case 'stale':
      return 'warning'
    default:
      return 'subtle'
  }
}

function formatEstimateValue(value: number): string {
  return value.toLocaleString()
}

function unitLabel(value: number, labels: ScenarioRunEstimateUnitLabels): string {
  return value === 1 ? labels.singular : labels.plural
}

function formatEstimateSummary(
  estimate: ScenarioRunEstimate,
  labels: ScenarioRunEstimateUnitLabels,
  primaryCount?: number,
): string {
  if (primaryCount !== undefined) {
    return `${formatEstimateValue(primaryCount)} ${unitLabel(primaryCount, labels)}`
  }
  if (estimate.total !== null) {
    return `${formatEstimateValue(estimate.total)} ${unitLabel(estimate.total, labels)}`
  }
  if (estimate.minimum !== null && estimate.maximum !== null) {
    return estimate.minimum === estimate.maximum
      ? `${formatEstimateValue(estimate.minimum)} ${unitLabel(estimate.minimum, labels)}`
      : `${formatEstimateValue(estimate.minimum)}–${formatEstimateValue(estimate.maximum)} ${labels.plural}`
  }
  if (estimate.maximum !== null) {
    return `Up to ${formatEstimateValue(estimate.maximum)} ${unitLabel(estimate.maximum, labels)}`
  }
  if (estimate.minimum !== null) {
    return `At least ${formatEstimateValue(estimate.minimum)} ${unitLabel(estimate.minimum, labels)}`
  }
  return estimate.scope === 'default'
    ? 'Select targets to calculate'
    : 'Run size is confirmed at launch.'
}

function formatComponentFormula(component: ScenarioRunEstimateComponent): string {
  if (component.factors.length === 0) {
    return `${component.label}: ${formatEstimateValue(component.count)}`
  }
  const factors = component.factors
    .map((factor) => `${formatEstimateValue(factor.count)} ${factor.label}`)
    .join(' × ')
  return `${component.label}: ${factors} = ${formatEstimateValue(component.count)}`
}

function formatCalculation(estimate: ScenarioRunEstimate): string {
  const components = estimate.components.length > 0
    ? estimate.components.map(formatComponentFormula).join(' + ')
    : 'Component breakdown unavailable'
  let total = 'exact total unavailable'
  if (estimate.total !== null) {
    total = `planned total = ${formatEstimateValue(estimate.total)}`
  } else if (estimate.minimum !== null && estimate.maximum !== null) {
    total = estimate.minimum === estimate.maximum
      ? `planned total = ${formatEstimateValue(estimate.minimum)}`
      : `planned range = ${formatEstimateValue(estimate.minimum)}–${formatEstimateValue(estimate.maximum)}`
  } else if (estimate.maximum !== null) {
    total = `planned maximum = ${formatEstimateValue(estimate.maximum)}`
  } else if (estimate.minimum !== null) {
    total = `planned minimum = ${formatEstimateValue(estimate.minimum)}`
  }
  return `${components}; ${total}`
}

export function ScenarioRunEstimateSummary({
  state,
  primaryCount,
  unitLabels = DEFAULT_UNIT_LABELS,
  supportingText,
}: ScenarioRunEstimateSummaryProps) {
  const styles = useScenarioRunEstimateStyles()
  const estimate = stateEstimate(state)
  const label = statusLabel(state)

  return (
    <div className={styles.summary} aria-live="polite">
      <div className={styles.summaryHeader}>
        {label && <Badge appearance="tint" color={statusColor(state)}>{label}</Badge>}
        {estimate && (
          <Text className={styles.total} weight="semibold">
            {formatEstimateSummary(estimate, unitLabels, primaryCount)}
          </Text>
        )}
      </div>
      {estimate && supportingText && (
        <Text size={200} className={styles.muted}>{supportingText}</Text>
      )}
    </div>
  )
}

function EstimateComponents({
  estimate,
  idPrefix,
}: {
  estimate: ScenarioRunEstimate
  idPrefix: string
}) {
  const styles = useScenarioRunEstimateStyles()
  const headingId = `${idPrefix}-components`

  return (
    <section className={styles.detailGroup} aria-labelledby={headingId}>
      <Text as="h4" id={headingId} weight="semibold">
        Planned components
      </Text>
      {estimate.components.length === 0 ? (
        <Text size={200} className={styles.muted}>
          A component breakdown isn’t available.
        </Text>
      ) : (
        <ol className={styles.componentList}>
          {estimate.components.map((component) => (
            <li className={styles.component} key={component.id}>
              <div className={styles.componentHeader}>
                <Text weight="semibold">{component.label}</Text>
                <div className={styles.componentCount}>
                  {component.isBaseline && (
                    <Badge appearance="tint" color="informative">Baseline</Badge>
                  )}
                  <Text weight="semibold">{formatEstimateValue(component.count)}</Text>
                </div>
              </div>
              {component.factors.length > 0 && (
                <ul className={styles.factorList} aria-label={`${component.label} factors`}>
                  {component.factors.map((factor) => (
                    <li key={factor.id}>
                      <Text size={200}>
                        × {formatEstimateValue(factor.count)} {factor.label}
                      </Text>
                    </li>
                  ))}
                </ul>
              )}
              {component.note && (
                <Text size={200} className={styles.muted}>{component.note}</Text>
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}

function EstimateDatasets({
  estimate,
  idPrefix,
}: {
  estimate: ScenarioRunEstimate
  idPrefix: string
}) {
  const styles = useScenarioRunEstimateStyles()
  const headingId = `${idPrefix}-datasets`

  return (
    <section className={styles.detailGroup} aria-labelledby={headingId}>
      <Text as="h4" id={headingId} weight="semibold">
        Dataset populations
      </Text>
      {estimate.datasets.length === 0 ? (
        <Text size={200} className={styles.muted}>
          Dataset population details aren’t available.
        </Text>
      ) : (
        <div className={styles.datasetList}>
          {estimate.datasets.map((dataset) => (
            <article className={styles.dataset} key={dataset.id}>
              <div className={styles.datasetHeader}>
                <Text weight="semibold">{dataset.name}</Text>
                <Badge appearance="tint" color="informative">{dataset.kind}</Badge>
              </div>
              <dl className={styles.countList}>
                <div className={styles.countRow}>
                  <dt>Logical seed groups</dt>
                  <dd>{formatEstimateValue(dataset.logicalSeedGroupCount)}</dd>
                </div>
                <div className={styles.countRow}>
                  <dt>Selected seed groups</dt>
                  <dd>{formatEstimateValue(dataset.selectedSeedGroupCount)}</dd>
                </div>
              </dl>
              {dataset.configuredCaps.length > 0 && (
                <div className={styles.capGroup}>
                  <Text size={200} weight="semibold">Configured caps</Text>
                  <ul className={styles.capList}>
                    {dataset.configuredCaps.map((cap) => (
                      <li key={cap.id}>
                        <Text size={200}>
                          {cap.label}: {formatEstimateValue(cap.count)}
                          {' '}({cap.configuredOn}{cap.datasetName ? `: ${cap.datasetName}` : ''})
                        </Text>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {dataset.selectionNote && (
                <Text size={200} className={styles.muted}>{dataset.selectionNote}</Text>
              )}
            </article>
          ))}
        </div>
      )}
    </section>
  )
}

export function ScenarioRunEstimateDetails({
  state,
  idPrefix = 'scenario-run-estimate',
  primaryCount,
  unitLabels,
  supportingText,
}: ScenarioRunEstimateDetailsProps) {
  const styles = useScenarioRunEstimateStyles()

  if (state.status === 'loading') {
    return (
      <div className={styles.details} aria-live="polite">
        <Spinner size="tiny" label="Calculating planned attacks..." />
        <Text size={200} className={styles.muted}>{scopeLabel(state)}</Text>
      </div>
    )
  }

  if (state.status === 'unavailable') {
    return (
      <div className={styles.details} aria-live="polite">
        <ScenarioRunEstimateSummary
          state={state}
          primaryCount={primaryCount}
          unitLabels={unitLabels}
          supportingText={supportingText}
        />
        <Text>{state.label}</Text>
        {state.note && <Text size={200} className={styles.muted}>{state.note}</Text>}
      </div>
    )
  }

  const { estimate } = state
  return (
    <div className={styles.details} aria-live="polite">
      <ScenarioRunEstimateSummary
        state={state}
        primaryCount={primaryCount}
        unitLabels={unitLabels}
        supportingText={supportingText}
      />
      {state.status === 'refreshing' && (
        <Text size={200} className={styles.muted}>{state.label}</Text>
      )}
      {state.status === 'stale' && (
        <div className={styles.staleNotice} role="status">
          <Text weight="semibold">{state.label}</Text>
          <Text size={200}>{state.error}</Text>
        </div>
      )}
      <EstimateComponents estimate={estimate} idPrefix={idPrefix} />
      <EstimateDatasets estimate={estimate} idPrefix={idPrefix} />
      <section className={styles.detailGroup} aria-labelledby={`${idPrefix}-formula`}>
        <Text as="h4" id={`${idPrefix}-formula`} weight="semibold">
          How this count is calculated
        </Text>
        <code className={styles.formula}>{formatCalculation(estimate)}</code>
      </section>
      <section className={styles.detailGroup} aria-labelledby={`${idPrefix}-notes`}>
        <Text as="h4" id={`${idPrefix}-notes`} weight="semibold">
          Estimate notes
        </Text>
        <Text size={200} className={styles.muted}>
          {estimate.note ?? 'No additional calculation note is available.'}
        </Text>
        <Text size={200} className={styles.muted}>
          Retries are {estimate.retriesIncluded ? 'included' : 'not included'}.
          {' '}Estimate schema v{estimate.version}.
        </Text>
      </section>
    </div>
  )
}
