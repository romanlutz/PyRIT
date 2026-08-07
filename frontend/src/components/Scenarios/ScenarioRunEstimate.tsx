import { Badge, Spinner, Text } from '@fluentui/react-components'

import type {
  ScenarioRunEstimateState,
  ScenarioRunEstimateTerm,
} from '@/types'

import { useScenarioRunEstimateStyles } from './ScenarioRunEstimate.styles'

interface ScenarioRunEstimateSummaryProps {
  state: ScenarioRunEstimateState
}

interface ScenarioRunEstimateDetailsProps {
  state: ScenarioRunEstimateState
}

function scopeLabel(state: ScenarioRunEstimateState): string {
  if (state.status === 'available' || state.status === 'conditional') {
    return state.estimate.scope === 'default' ? 'Default configuration' : 'Current configuration'
  }
  return state.scope === 'default' ? 'Default configuration' : 'Current configuration'
}

function statusLabel(state: ScenarioRunEstimateState): string {
  switch (state.status) {
    case 'loading':
      return 'Loading estimate'
    case 'available':
      return 'Backend estimate'
    case 'conditional':
      return 'Conditional estimate'
    case 'unavailable':
      return 'Estimate unavailable'
  }
}

function statusColor(state: ScenarioRunEstimateState): 'brand' | 'warning' | 'subtle' {
  switch (state.status) {
    case 'available':
      return 'brand'
    case 'conditional':
      return 'warning'
    default:
      return 'subtle'
  }
}

function formatEstimateValue(value: number): string {
  return value.toLocaleString()
}

function EstimateTerms({
  heading,
  terms,
  factor,
}: {
  heading: string
  terms: ScenarioRunEstimateTerm[]
  factor: boolean
}) {
  const styles = useScenarioRunEstimateStyles()

  if (terms.length === 0) {
    return null
  }

  return (
    <div className={styles.termGroup}>
      <Text weight="semibold">{heading}</Text>
      <dl className={styles.termList}>
        {terms.map((term) => (
          <div className={styles.termRow} key={term.id}>
            <dt className={styles.termLabel}>{term.label}</dt>
            <dd className={styles.termValue}>
              {factor ? '× ' : ''}{formatEstimateValue(term.value)}
            </dd>
            {term.detail && (
              <dd className={styles.termDetail}>{term.detail}</dd>
            )}
          </div>
        ))}
      </dl>
    </div>
  )
}

export function ScenarioRunEstimateSummary({ state }: ScenarioRunEstimateSummaryProps) {
  const styles = useScenarioRunEstimateStyles()
  const total = state.status === 'available' || state.status === 'conditional'
    ? state.estimate.total
    : undefined

  return (
    <div className={styles.summary} aria-live="polite">
      <div className={styles.summaryHeader}>
        <Badge appearance="tint" color={statusColor(state)}>{statusLabel(state)}</Badge>
        {total !== undefined && (
          <Text className={styles.total} weight="semibold">
            {formatEstimateValue(total)} attacks
          </Text>
        )}
      </div>
      <Text size={200} className={styles.muted}>{scopeLabel(state)}</Text>
    </div>
  )
}

export function ScenarioRunEstimateDetails({ state }: ScenarioRunEstimateDetailsProps) {
  const styles = useScenarioRunEstimateStyles()

  if (state.status === 'loading') {
    return (
      <div className={styles.details} aria-live="polite">
        <Spinner size="tiny" label="Loading backend run estimate..." />
        <Text size={200} className={styles.muted}>{scopeLabel(state)}</Text>
      </div>
    )
  }

  if (state.status === 'unavailable') {
    return (
      <div className={styles.details} aria-live="polite">
        <ScenarioRunEstimateSummary state={state} />
        <Text>{state.label}</Text>
        {state.caveat && <Text size={200} className={styles.caveat}>{state.caveat}</Text>}
      </div>
    )
  }

  const { estimate } = state
  return (
    <div className={styles.details} aria-live="polite">
      <ScenarioRunEstimateSummary state={state} />
      <EstimateTerms
        heading="Additive components"
        terms={estimate.additiveComponents}
        factor={false}
      />
      <EstimateTerms
        heading="Multiplicative factors"
        terms={estimate.multiplicativeFactors}
        factor
      />
      <div className={styles.termGroup}>
        <Text weight="semibold">Backend formula</Text>
        <code className={styles.formula}>
          {estimate.formula ?? 'No formula supplied by the backend.'}
        </code>
      </div>
      <div className={styles.termGroup}>
        <Text weight="semibold">Caveat</Text>
        <Text size={200} className={styles.caveat}>
          {estimate.caveat ?? 'No additional caveat supplied by the backend.'}
        </Text>
      </div>
    </div>
  )
}
