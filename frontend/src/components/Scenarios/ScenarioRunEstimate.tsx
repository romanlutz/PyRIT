import { Badge, Spinner, Text } from '@fluentui/react-components'

import type {
  ScenarioRunEstimate,
  ScenarioRunEstimateComponent,
  ScenarioRunEstimateFactor,
  ScenarioRunEstimateState,
} from '@/types'

import { useScenarioRunEstimateStyles } from './ScenarioRunEstimate.styles'

interface ScenarioRunEstimateSummaryProps {
  state: ScenarioRunEstimateState
}

interface ScenarioRunEstimateDetailsProps {
  state: ScenarioRunEstimateState
  idPrefix?: string
}

interface CalculationOperand {
  id: string
  value: string
  label: string
  result?: boolean
}

interface CalculationOperator {
  id: string
  symbol: '(' | ')' | '×' | '+' | '='
}

type CalculationPart =
  | { kind: 'operand'; operand: CalculationOperand }
  | { kind: 'operator'; operator: CalculationOperator }

interface RunCalculation {
  parts: CalculationPart[]
  accessibleLabel: string
  context?: string
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

function formatCount(value: number): string {
  return value.toLocaleString()
}

function countLabel(value: number, singular: string, plural: string): string {
  return `${formatCount(value)} ${value === 1 ? singular : plural}`
}

function formatEstimateSummary(estimate: ScenarioRunEstimate): string {
  if (estimate.adaptiveDetails) {
    const { objectiveCount, techniqueAttemptCountUpperBound } = estimate.adaptiveDetails
    return `${countLabel(objectiveCount, 'objective', 'objectives')} · up to ${countLabel(
      techniqueAttemptCountUpperBound,
      'technique attempt',
      'technique attempts',
    )}`
  }
  if (estimate.total !== null) {
    return countLabel(estimate.total, 'planned attack', 'planned attacks')
  }
  if (estimate.minimum !== null && estimate.maximum !== null) {
    return estimate.minimum === estimate.maximum
      ? countLabel(estimate.minimum, 'planned attack', 'planned attacks')
      : `${formatCount(estimate.minimum)}–${formatCount(estimate.maximum)} planned attacks`
  }
  if (estimate.maximum !== null) {
    return `Up to ${countLabel(estimate.maximum, 'planned attack', 'planned attacks')}`
  }
  if (estimate.minimum !== null) {
    return `At least ${countLabel(estimate.minimum, 'planned attack', 'planned attacks')}`
  }
  return estimate.scope === 'default'
    ? 'Select targets to calculate'
    : 'Run size is confirmed at launch.'
}

function operand(id: string, value: string, label: string, result = false): CalculationPart {
  return { kind: 'operand', operand: { id, value, label, result } }
}

function operator(id: string, symbol: CalculationOperator['symbol']): CalculationPart {
  return { kind: 'operator', operator: { id, symbol } }
}

function humanizeLabel(label: string): string {
  const words = label.replace(/_/g, ' ').trim()
  return words.length > 0 ? `${words[0].toUpperCase()}${words.slice(1)}` : label
}

function semanticFactorLabel(factor: ScenarioRunEstimateFactor): string {
  const label = factor.label.toLowerCase()
  if (label.includes('seed group') || label === 'objectives') {
    return factor.count === 1 ? 'objective' : 'objectives'
  }
  if (label.includes('technique')) {
    return factor.count === 1 ? 'technique' : 'techniques'
  }
  if (factor.count === 1 && label.endsWith('s')) {
    return label.slice(0, -1)
  }
  return label
}

function factorPriority(factor: ScenarioRunEstimateFactor): number {
  const label = semanticFactorLabel(factor)
  if (label === 'technique' || label === 'techniques') return 0
  if (label === 'objective' || label === 'objectives') return 1
  return 2
}

function objectiveFactor(component: ScenarioRunEstimateComponent): ScenarioRunEstimateFactor | undefined {
  return component.factors.find((factor) => {
    const label = semanticFactorLabel(factor)
    return label === 'objective' || label === 'objectives'
  })
}

function resultOperand(estimate: ScenarioRunEstimate): CalculationOperand {
  if (estimate.total !== null) {
    return {
      id: 'result',
      value: formatCount(estimate.total),
      label: estimate.total === 1 ? 'planned attack' : 'planned attacks',
      result: true,
    }
  }
  if (estimate.minimum !== null && estimate.maximum !== null) {
    return {
      id: 'result',
      value: estimate.minimum === estimate.maximum
        ? formatCount(estimate.minimum)
        : `${formatCount(estimate.minimum)}–${formatCount(estimate.maximum)}`,
      label: estimate.minimum === 1 && estimate.maximum === 1 ? 'planned attack' : 'planned attacks',
      result: true,
    }
  }
  if (estimate.maximum !== null) {
    return {
      id: 'result',
      value: `up to ${formatCount(estimate.maximum)}`,
      label: estimate.maximum === 1 ? 'planned attack' : 'planned attacks',
      result: true,
    }
  }
  if (estimate.minimum !== null) {
    return {
      id: 'result',
      value: `at least ${formatCount(estimate.minimum)}`,
      label: estimate.minimum === 1 ? 'planned attack' : 'planned attacks',
      result: true,
    }
  }
  return { id: 'result', value: 'Exact total', label: 'unavailable', result: true }
}

function adaptiveCalculation(estimate: ScenarioRunEstimate): RunCalculation {
  const details = estimate.adaptiveDetails
  if (!details) {
    throw new Error('Adaptive calculation requires adaptive details.')
  }
  const objectiveLabel = details.objectiveCount === 1 ? 'objective' : 'objectives'
  const techniqueLabel = details.techniquesPerObjectiveUpperBound === 1
    ? 'technique per objective'
    : 'techniques per objective'
  const attemptLabel = details.techniqueAttemptCountUpperBound === 1
    ? 'technique attempt'
    : 'technique attempts'
  const poolLimit = details.maxAttemptsPerObjective < details.candidateTechniqueCount
    ? ` of ${formatCount(details.candidateTechniqueCount)} candidates`
    : ''
  return {
    parts: [
      operand('adaptive-objectives', formatCount(details.objectiveCount), objectiveLabel),
      operator('adaptive-multiply', '×'),
      operand(
        'adaptive-techniques',
        `up to ${formatCount(details.techniquesPerObjectiveUpperBound)}`,
        `${techniqueLabel}${poolLimit}`,
      ),
      operator('adaptive-equals', '='),
      operand(
        'adaptive-result',
        `up to ${formatCount(details.techniqueAttemptCountUpperBound)}`,
        attemptLabel,
        true,
      ),
    ],
    accessibleLabel: `${countLabel(details.objectiveCount, 'objective', 'objectives')} multiplied by up to ${
      countLabel(details.techniquesPerObjectiveUpperBound, 'technique per objective', 'techniques per objective')
    } equals up to ${
      countLabel(details.techniqueAttemptCountUpperBound, 'technique attempt', 'technique attempts')
    }.`,
    context: `Progress tracks ${countLabel(details.objectiveCount, 'objective', 'objectives')}. Each objective stops after the first successful technique, and compatibility may reduce how many candidates it can try.`,
  }
}

function homogeneousTechniqueCalculation(
  components: ScenarioRunEstimateComponent[],
): CalculationPart[] | null {
  if (components.length < 2 || components.some((component) => component.condition !== null)) {
    return null
  }
  const objectiveCounts = components.map((component) => objectiveFactor(component)?.count)
  if (objectiveCounts.some((count) => count === undefined)) {
    return null
  }
  const firstCount = objectiveCounts[0]
  if (!objectiveCounts.every((count) => count === firstCount)) {
    return null
  }
  return [
    operand(
      'technique-count',
      formatCount(components.length),
      components.length === 1 ? 'technique' : 'techniques',
    ),
    operator('technique-multiply', '×'),
    operand(
      'objective-count',
      formatCount(firstCount ?? 0),
      firstCount === 1 ? 'objective' : 'objectives',
    ),
  ]
}

function componentTerms(components: ScenarioRunEstimateComponent[]): CalculationPart[] {
  const homogeneous = homogeneousTechniqueCalculation(components)
  if (homogeneous) {
    return homogeneous
  }
  if (components.length === 1 && components[0].condition === null && components[0].factors.length > 0) {
    return [...components[0].factors]
      .sort((left, right) => factorPriority(left) - factorPriority(right))
      .flatMap((factor, index) => [
        ...(index > 0 ? [operator(`factor-${index}-multiply`, '×')] : []),
        operand(`factor-${factor.id}`, formatCount(factor.count), semanticFactorLabel(factor)),
      ])
  }
  return components.flatMap((component, index) => {
    const objectiveCount = objectiveFactor(component)?.count
    const value = formatCount(objectiveCount ?? component.count)
    const unit = objectiveCount === undefined
      ? component.count === 1 ? 'planned attack' : 'planned attacks'
      : objectiveCount === 1 ? 'objective' : 'objectives'
    const condition = component.condition ? ' · if supported' : ''
    return [
      ...(index > 0 ? [operator(`component-${index}-plus`, '+')] : []),
      operand(
        `component-${component.id}`,
        value,
        `${unit} · ${humanizeLabel(component.label)}${condition}`,
      ),
    ]
  })
}

function ordinaryCalculation(estimate: ScenarioRunEstimate): RunCalculation {
  const baselineCount = estimate.components
    .filter((component) => component.isBaseline)
    .reduce((sum, component) => sum + component.count, 0)
  const attackComponents = estimate.components.filter((component) => !component.isBaseline)
  const resultOnly = baselineCount === 0
    && attackComponents.length === 1
    && attackComponents[0].factors.length === 0
    && attackComponents[0].condition === null
  const attackParts = resultOnly ? [] : componentTerms(attackComponents)
  const hasMultiplication = attackParts.some(
    (part) => part.kind === 'operator' && part.operator.symbol === '×',
  )
  const parts: CalculationPart[] = []
  if (baselineCount > 0 && hasMultiplication) {
    parts.push(operator('attack-open', '('))
  }
  parts.push(...attackParts)
  if (baselineCount > 0 && hasMultiplication) {
    parts.push(operator('attack-close', ')'))
  }
  if (baselineCount > 0) {
    if (attackParts.length > 0) {
      parts.push(operator('baseline-plus', '+'))
    }
    parts.push(operand(
      'baseline',
      formatCount(baselineCount),
      baselineCount === 1 ? 'direct baseline send' : 'direct baseline sends',
    ))
  }
  const result = resultOperand(estimate)
  if (parts.length > 0) {
    parts.push(operator('total-equals', '='))
  }
  parts.push({ kind: 'operand', operand: result })

  const visibleExpression = parts.map((part) => part.kind === 'operator'
    ? part.operator.symbol
    : `${part.operand.value} ${part.operand.label}`).join(' ')
  return {
    parts,
    accessibleLabel: `${visibleExpression
      .replace(/×/g, 'multiplied by')
      .replace(/\+/g, 'plus')
      .replace(/=/g, 'equals')}.`,
    context: estimate.total === null && estimate.minimum === null && estimate.maximum === null
      ? formatEstimateSummary(estimate)
      : undefined,
  }
}

function runCalculation(estimate: ScenarioRunEstimate): RunCalculation {
  return estimate.adaptiveDetails ? adaptiveCalculation(estimate) : ordinaryCalculation(estimate)
}

export function ScenarioRunEstimateSummary({ state }: ScenarioRunEstimateSummaryProps) {
  const styles = useScenarioRunEstimateStyles()
  const estimate = stateEstimate(state)
  const label = statusLabel(state)

  return (
    <div className={styles.summary} aria-live="polite">
      <div className={styles.summaryHeader}>
        {label && <Badge appearance="tint" color={statusColor(state)}>{label}</Badge>}
        {estimate && (
          <Text className={styles.total} weight="semibold">
            {formatEstimateSummary(estimate)}
          </Text>
        )}
      </div>
    </div>
  )
}

function RunCalculationView({
  estimate,
  idPrefix,
}: {
  estimate: ScenarioRunEstimate
  idPrefix: string
}) {
  const styles = useScenarioRunEstimateStyles()
  const calculation = runCalculation(estimate)
  const headingId = `${idPrefix}-calculation`

  return (
    <section className={styles.calculationSection} aria-labelledby={headingId}>
      <Text as="h4" id={headingId} weight="semibold">Run calculation</Text>
      <div
        className={styles.equation}
        role="group"
        aria-label={calculation.accessibleLabel}
        data-testid="run-calculation"
      >
        {calculation.parts.map((part) => part.kind === 'operator' ? (
          <Text
            aria-hidden="true"
            className={styles.operator}
            key={part.operator.id}
            weight="semibold"
          >
            {part.operator.symbol}
          </Text>
        ) : (
          <span
            aria-hidden="true"
            className={part.operand.result ? styles.resultOperand : styles.operand}
            key={part.operand.id}
          >
            <Text className={styles.operandValue} weight="semibold">{part.operand.value}</Text>
            <Text size={200}>{part.operand.label}</Text>
          </span>
        ))}
      </div>
      {calculation.context && (
        <Text size={200} className={styles.calculationContext}>{calculation.context}</Text>
      )}
    </section>
  )
}

function EstimateSources({ estimate }: { estimate: ScenarioRunEstimate }) {
  const styles = useScenarioRunEstimateStyles()
  if (estimate.datasets.length === 0) {
    return null
  }

  return (
    <div className={styles.sources} role="group" aria-label="Objective sources">
      {estimate.datasets.map((dataset) => (
        <div className={styles.source} key={dataset.id}>
          <Text size={200}>
            {countLabel(dataset.selectedSeedGroupCount, 'objective', 'objectives')} from {dataset.name}
            {dataset.logicalSeedGroupCount !== dataset.selectedSeedGroupCount
              ? ` · ${formatCount(dataset.logicalSeedGroupCount)} available`
              : ''}
          </Text>
          {dataset.configuredCaps.length > 0 && (
            <Text size={200} className={styles.muted}>
              {dataset.configuredCaps.map((cap) => (
                `${cap.label}: ${formatCount(cap.count)}`
              )).join(' · ')}
            </Text>
          )}
        </div>
      ))}
    </div>
  )
}

export function ScenarioRunEstimateDetails({
  state,
  idPrefix = 'scenario-run-estimate',
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
        <ScenarioRunEstimateSummary state={state} />
        <Text>{state.label}</Text>
        {state.note && <Text size={200} className={styles.muted}>{state.note}</Text>}
      </div>
    )
  }

  const { estimate } = state
  return (
    <div className={styles.details} aria-live="polite">
      <RunCalculationView estimate={estimate} idPrefix={idPrefix} />
      {state.status === 'refreshing' && (
        <Text size={200} className={styles.muted}>{state.label}</Text>
      )}
      {state.status === 'stale' && (
        <div className={styles.staleNotice} role="status">
          <Text weight="semibold">Previous estimate</Text>
          <Text size={200}>{state.label}</Text>
          <Text size={200}>{state.error}</Text>
        </div>
      )}
      <EstimateSources estimate={estimate} />
      <Text size={200} className={styles.muted}>
        Retries are {estimate.retriesIncluded ? 'included' : 'not included'}.
      </Text>
    </div>
  )
}
