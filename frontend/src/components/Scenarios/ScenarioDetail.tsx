import { useEffect, useRef, useState } from 'react'

import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Button,
  Checkbox,
  Field,
  Input,
  MessageBar,
  MessageBarBody,
  Select,
  Spinner,
  SpinButton,
  Text,
} from '@fluentui/react-components'
import { ArrowLeftRegular, ArrowSyncRegular, SettingsRegular } from '@fluentui/react-icons'
import { Link, useNavigate, useParams } from 'react-router-dom'

import ParameterField from '@/components/Parameters/ParameterField'
import { buildParametersFromForm, getInitialFormValues, type ParameterFormValue } from '@/components/Parameters/parameterForm'
import type { ViewName } from '@/components/Sidebar/Navigation'
import { scenariosApi, targetsApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type { RegisteredScenario, RunScenarioRequest, TargetInstance } from '@/types'
import { fetchAllPages } from '@/utils/fetchAllPages'
import { routerPathParamValue } from '@/utils/routeParams'

import { useScenarioDetailStyles } from './ScenarioDetail.styles'

/** Items requested per target page while paging through the full list. */
const TARGET_PAGE_SIZE = 200

/**
 * Common/opaque parameters every scenario declares via
 * `Scenario._common_scenario_parameters` — the launch form already exposes a
 * purpose-built control for each of these (target, techniques, datasets,
 * labels, concurrency, retries, baseline), and `technique_converters` has no
 * UI at all. They're hidden from the dynamic scenario-specific parameter list.
 */
const COMMON_SCENARIO_PARAMETER_NAMES = new Set([
  'objective_target',
  'scenario_techniques',
  'technique_converters',
  'dataset_config',
  'memory_labels',
  'max_concurrency',
  'max_retries',
  'include_baseline',
])

const MIN_MAX_CONCURRENCY = 1
const MAX_MAX_CONCURRENCY = 100
const MIN_MAX_RETRIES = 0
const MAX_MAX_RETRIES = 20
const DEFAULT_MAX_CONCURRENCY = 10
const DEFAULT_MAX_RETRIES = 0

/** Resolves a Fluent `SpinButton` change event to a numeric value, preferring the parsed `value` over the raw `displayValue`. */
function resolveSpinButtonValue(data: { value?: number | null; displayValue?: string }, previous: number): number {
  if (typeof data.value === 'number') {
    return data.value
  }
  const parsed = data.displayValue !== undefined ? Number(data.displayValue) : NaN
  return Number.isFinite(parsed) ? parsed : previous
}

type LoadStatus = 'loading' | 'success' | 'not-found' | 'error'

/** Options rendered for technique selection: aggregates first, then concrete techniques, deduped by exact name. */
function uniqueTechniqueOptions(scenario: RegisteredScenario): { aggregates: string[]; concrete: string[] } {
  const seen = new Set<string>()
  const aggregates: string[] = []
  for (const name of scenario.aggregate_techniques) {
    if (!seen.has(name)) {
      seen.add(name)
      aggregates.push(name)
    }
  }
  const concrete: string[] = []
  for (const name of scenario.all_techniques) {
    if (!seen.has(name)) {
      seen.add(name)
      concrete.push(name)
    }
  }
  return { aggregates, concrete }
}

interface ScenarioDetailProps {
  activeTarget: TargetInstance | null
  labels: Record<string, string>
  onNavigate: (view: ViewName) => void
}

export default function ScenarioDetail(props: ScenarioDetailProps) {
  const { scenarioName: encodedScenarioName } = useParams<{ scenarioName: string }>()
  // Keying on the raw URL param forces a full remount (and state reset to the
  // initial "loading" values) whenever the route navigates from one scenario
  // detail page directly to another, without needing to reset state from
  // inside an effect.
  return <ScenarioDetailContent key={encodedScenarioName} encodedScenarioName={encodedScenarioName} {...props} />
}

interface ScenarioDetailContentProps extends ScenarioDetailProps {
  encodedScenarioName: string | undefined
}

function ScenarioDetailContent({
  encodedScenarioName,
  activeTarget,
  labels,
  onNavigate,
}: ScenarioDetailContentProps) {
  const styles = useScenarioDetailStyles()
  const decodedScenarioName = routerPathParamValue(encodedScenarioName)

  const [scenario, setScenario] = useState<RegisteredScenario | null>(null)
  const [scenarioStatus, setScenarioStatus] = useState<LoadStatus>('loading')
  const [scenarioError, setScenarioError] = useState<string | null>(null)
  const [targets, setTargets] = useState<TargetInstance[] | null>(null)
  const [targetsError, setTargetsError] = useState<string | null>(null)
  const [refetchCount, setRefetchCount] = useState(0)

  useEffect(() => {
    let cancelled = false
    scenariosApi
      .getScenario(decodedScenarioName)
      .then((data) => {
        if (cancelled) return
        setScenario(data)
        setScenarioStatus('success')
        setScenarioError(null)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        const apiError = toApiError(err)
        setScenario(null)
        setScenarioStatus(apiError.status === 404 ? 'not-found' : 'error')
        setScenarioError(apiError.status === 404 ? null : apiError.detail)
      })
    return () => {
      cancelled = true
    }
  }, [decodedScenarioName, refetchCount])

  useEffect(() => {
    let cancelled = false
    fetchAllPages(
      (cursor) => targetsApi.listTargets(TARGET_PAGE_SIZE, cursor),
      undefined,
      (target) => target.target_registry_name,
    )
      .then((items) => {
        if (cancelled) return
        setTargets(items)
        setTargetsError(null)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setTargets([])
        setTargetsError(toApiError(err).detail)
      })
    return () => {
      cancelled = true
    }
  }, [refetchCount])

  const handleRetry = (): void => {
    setScenarioStatus('loading')
    setScenarioError(null)
    setTargets(null)
    setTargetsError(null)
    setRefetchCount((count) => count + 1)
  }

  if (scenarioStatus === 'loading' || targets === null) {
    return (
      <div className={styles.root} data-testid="scenario-detail">
        <div className={styles.centeredState}>
          <Spinner label="Loading scenario..." />
        </div>
      </div>
    )
  }

  if (scenarioStatus === 'not-found') {
    return (
      <div className={styles.root} data-testid="scenario-detail">
        <Link to="/scenarios" className={styles.backLink}>
          <ArrowLeftRegular /> Back to scenarios
        </Link>
        <div className={styles.centeredState} data-testid="scenario-not-found">
          <Text size={400}>Scenario &quot;{decodedScenarioName}&quot; was not found</Text>
          <Text size={200}>It may have been renamed or is no longer registered.</Text>
        </div>
      </div>
    )
  }

  if (scenarioStatus === 'error' || targetsError) {
    return (
      <div className={styles.root} data-testid="scenario-detail">
        <Link to="/scenarios" className={styles.backLink}>
          <ArrowLeftRegular /> Back to scenarios
        </Link>
        <div className={styles.centeredState} data-testid="scenario-error">
          <MessageBar intent="error">
            <MessageBarBody>{scenarioError ?? targetsError}</MessageBarBody>
          </MessageBar>
          <Button
            className={styles.touchTarget}
            appearance="primary"
            icon={<ArrowSyncRegular />}
            onClick={handleRetry}
            data-testid="retry-btn"
          >
            Retry
          </Button>
        </div>
      </div>
    )
  }

  // scenarioStatus === 'success' from here on; TS can't narrow `scenario` from
  // `scenarioStatus`, so this is asserted defensively — both are set together.
  if (!scenario) {
    return null
  }

  if (targets.length === 0) {
    return (
      <div className={styles.root} data-testid="scenario-detail">
        <Link to="/scenarios" className={styles.backLink}>
          <ArrowLeftRegular /> Back to scenarios
        </Link>
        <div className={styles.centeredState} data-testid="no-targets-state">
          <Text size={400}>No targets configured</Text>
          <Text size={200}>Configure a target before launching a scenario.</Text>
          <Button
            className={styles.touchTarget}
            appearance="primary"
            icon={<SettingsRegular />}
            onClick={() => onNavigate('config')}
          >
            Configure target
          </Button>
        </div>
      </div>
    )
  }

  return (
    <ScenarioLaunchForm
      key={scenario.scenario_name}
      scenario={scenario}
      targets={targets}
      activeTarget={activeTarget}
      labels={labels}
    />
  )
}

interface ScenarioLaunchFormProps {
  scenario: RegisteredScenario
  targets: TargetInstance[]
  activeTarget: TargetInstance | null
  labels: Record<string, string>
}

function ScenarioLaunchForm({ scenario, targets, activeTarget, labels }: ScenarioLaunchFormProps) {
  const styles = useScenarioDetailStyles()
  const navigate = useNavigate()

  const { aggregates, concrete } = uniqueTechniqueOptions(scenario)
  const dynamicParameters = scenario.supported_parameters.filter(
    (parameter) => !COMMON_SCENARIO_PARAMETER_NAMES.has(parameter.name),
  )
  const isBaselineForbidden = scenario.baseline_policy === 'forbidden'

  const [targetName, setTargetName] = useState(() => {
    if (activeTarget && targets.some((t) => t.target_registry_name === activeTarget.target_registry_name)) {
      return activeTarget.target_registry_name
    }
    return targets[0].target_registry_name
  })
  const [selectedTechniques, setSelectedTechniques] = useState<string[]>([scenario.default_technique])
  const [baselineChecked, setBaselineChecked] = useState(
    () => !isBaselineForbidden && scenario.include_baseline_by_default,
  )
  const [datasetOverride, setDatasetOverride] = useState('')
  const [maxDatasetSize, setMaxDatasetSize] = useState('')
  const [maxConcurrency, setMaxConcurrency] = useState(DEFAULT_MAX_CONCURRENCY)
  const [maxRetries, setMaxRetries] = useState(DEFAULT_MAX_RETRIES)
  const [scenarioParamValues, setScenarioParamValues] = useState<Record<string, ParameterFormValue>>(() =>
    getInitialFormValues(dynamicParameters),
  )
  const [validationError, setValidationError] = useState<string | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  // Synchronous guard against a double-submit (e.g. a fast double click) racing
  // ahead of the `submitting` state update, which only takes effect next render.
  const isSubmittingRef = useRef(false)

  const toggleTechnique = (name: string, checked: boolean): void => {
    setSelectedTechniques((prev) => {
      if (checked) {
        return prev.includes(name) ? prev : [...prev, name]
      }
      return prev.filter((entry) => entry !== name)
    })
  }

  const updateScenarioParam = (name: string, value: ParameterFormValue): void => {
    setScenarioParamValues((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (): Promise<void> => {
    if (isSubmittingRef.current) {
      return
    }

    setApiError(null)
    if (!targetName) {
      setValidationError('Select a target.')
      return
    }
    if (selectedTechniques.length === 0) {
      setValidationError('Select at least one technique.')
      return
    }

    let scenarioParams: Record<string, unknown> | null = null
    if (dynamicParameters.length > 0) {
      const result = buildParametersFromForm(dynamicParameters, scenarioParamValues)
      if (!result.ok) {
        setValidationError(result.error)
        return
      }
      scenarioParams = result.parameters
    }

    let maxDatasetSizeValue: number | undefined
    const trimmedMaxDatasetSize = maxDatasetSize.trim()
    if (trimmedMaxDatasetSize.length > 0) {
      const parsed = Number(trimmedMaxDatasetSize)
      if (!Number.isInteger(parsed) || parsed < 1) {
        setValidationError('Max dataset size must be a positive integer.')
        return
      }
      maxDatasetSizeValue = parsed
    }
    if (
      !Number.isInteger(maxConcurrency)
      || maxConcurrency < MIN_MAX_CONCURRENCY
      || maxConcurrency > MAX_MAX_CONCURRENCY
    ) {
      setValidationError(
        `Max concurrency must be an integer from ${MIN_MAX_CONCURRENCY} to ${MAX_MAX_CONCURRENCY}.`,
      )
      return
    }
    if (
      !Number.isInteger(maxRetries)
      || maxRetries < MIN_MAX_RETRIES
      || maxRetries > MAX_MAX_RETRIES
    ) {
      setValidationError(`Max retries must be an integer from ${MIN_MAX_RETRIES} to ${MAX_MAX_RETRIES}.`)
      return
    }

    const datasetNames = datasetOverride
      .split(',')
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0)

    isSubmittingRef.current = true
    setSubmitting(true)
    setValidationError(null)

    const request: RunScenarioRequest = {
      scenario_name: scenario.scenario_name,
      target_name: targetName,
      techniques: selectedTechniques,
      max_concurrency: maxConcurrency,
      max_retries: maxRetries,
      include_baseline: isBaselineForbidden ? false : baselineChecked,
      labels,
    }
    if (datasetNames.length > 0) {
      request.dataset_names = datasetNames
    }
    if (maxDatasetSizeValue !== undefined) {
      request.max_dataset_size = maxDatasetSizeValue
    }
    if (scenarioParams) {
      request.scenario_params = scenarioParams
    }

    try {
      const summary = await scenariosApi.startRun(request)
      navigate(`/scenario-history/${encodeURIComponent(summary.scenario_result_id)}`, {
        state: { scenarioName: scenario.scenario_name },
      })
    } catch (err) {
      setApiError(toApiError(err).detail)
    } finally {
      isSubmittingRef.current = false
      setSubmitting(false)
    }
  }

  return (
    <div className={styles.root} data-testid="scenario-detail">
      <Link to="/scenarios" className={styles.backLink}>
        <ArrowLeftRegular /> Back to scenarios
      </Link>

      <div className={styles.headerText}>
        <Text as="h1" size={600} weight="semibold">{scenario.scenario_name}</Text>
        <Text size={300} className={styles.description}>{scenario.description}</Text>
      </div>

      {validationError && (
        <MessageBar intent="warning">
          <MessageBarBody role="alert">{validationError}</MessageBarBody>
        </MessageBar>
      )}
      {apiError && (
        <MessageBar intent="error">
          <MessageBarBody role="alert">{apiError}</MessageBarBody>
        </MessageBar>
      )}

      <div className={styles.section}>
        <Field label="Target" hint="The registered target this scenario will run against.">
          <Select
            value={targetName}
            disabled={submitting}
            onChange={(_, data) => setTargetName(data.value)}
            data-testid="scenario-target-select"
          >
            {targets.map((target) => (
              <option key={target.target_registry_name} value={target.target_registry_name}>
                {target.target_registry_name}
              </option>
            ))}
          </Select>
        </Field>
      </div>

      <div className={styles.section}>
        <Text weight="semibold">Techniques</Text>
        <div className={styles.techniqueGroups}>
          {aggregates.length > 0 && (
            <Field label="Aggregate techniques">
              <div className={styles.checkboxGroup} role="group" aria-label="Aggregate techniques">
                {aggregates.map((name) => (
                  <Checkbox
                    key={name}
                    label={name}
                    checked={selectedTechniques.includes(name)}
                    disabled={submitting}
                    onChange={(_, data) => toggleTechnique(name, data.checked === true)}
                    data-testid={`technique-${name}`}
                  />
                ))}
              </div>
            </Field>
          )}
          <Field label="Individual techniques">
            <div className={styles.checkboxGroup} role="group" aria-label="Individual techniques">
              {concrete.map((name) => (
                <Checkbox
                  key={name}
                  label={name}
                  checked={selectedTechniques.includes(name)}
                  disabled={submitting}
                  onChange={(_, data) => toggleTechnique(name, data.checked === true)}
                  data-testid={`technique-${name}`}
                />
              ))}
            </div>
          </Field>
        </div>
      </div>

      <div className={styles.section}>
        <Field label="Baseline attack">
          <Checkbox
            checked={baselineChecked}
            disabled={submitting || isBaselineForbidden}
            label={isBaselineForbidden ? 'Not available for this scenario' : 'Include baseline attack'}
            onChange={(_, data) => setBaselineChecked(data.checked === true)}
            data-testid="baseline-checkbox"
          />
        </Field>
        {isBaselineForbidden && (
          <Text size={200} className={styles.hint}>
            This scenario forbids a baseline comparison run.
          </Text>
        )}
      </div>

      {dynamicParameters.length > 0 && (
        <div className={styles.section}>
          <Text weight="semibold">Scenario parameters</Text>
          <div className={styles.dynamicParameters}>
            {dynamicParameters.map((parameter) => (
              <ParameterField
                key={parameter.name}
                parameter={parameter}
                value={scenarioParamValues[parameter.name]}
                disabled={submitting}
                onChange={updateScenarioParam}
                testIdPrefix="scenario-param"
              />
            ))}
          </div>
        </div>
      )}

      <Accordion collapsible>
        <AccordionItem value="advanced">
          <AccordionHeader>Advanced options</AccordionHeader>
          <AccordionPanel>
            <div className={styles.advancedFields}>
              <Field
                label="Dataset override"
                hint="Comma-separated dataset names. Leave blank to use the scenario's default datasets."
              >
                <Input
                  value={datasetOverride}
                  disabled={submitting}
                  onChange={(_, data) => setDatasetOverride(data.value)}
                  placeholder={scenario.default_datasets.join(', ') || undefined}
                  data-testid="dataset-override-input"
                />
              </Field>
              <Field label="Max dataset size" hint="Optional. Leave blank for no cap.">
                <Input
                  className={styles.numberInput}
                  type="number"
                  min={1}
                  value={maxDatasetSize}
                  disabled={submitting}
                  onChange={(_, data) => setMaxDatasetSize(data.value)}
                  data-testid="max-dataset-size-input"
                />
              </Field>
              <Field label="Max concurrency">
                <SpinButton
                  className={styles.numberInput}
                  value={maxConcurrency}
                  min={MIN_MAX_CONCURRENCY}
                  max={MAX_MAX_CONCURRENCY}
                  disabled={submitting}
                  onChange={(_, data) => setMaxConcurrency(resolveSpinButtonValue(data, maxConcurrency))}
                  data-testid="max-concurrency-input"
                />
              </Field>
              <Field label="Max retries">
                <SpinButton
                  className={styles.numberInput}
                  value={maxRetries}
                  min={MIN_MAX_RETRIES}
                  max={MAX_MAX_RETRIES}
                  disabled={submitting}
                  onChange={(_, data) => setMaxRetries(resolveSpinButtonValue(data, maxRetries))}
                  data-testid="max-retries-input"
                />
              </Field>
            </div>
          </AccordionPanel>
        </AccordionItem>
      </Accordion>

      <div className={styles.actionsRow}>
        <Button
          className={styles.touchTarget}
          appearance="primary"
          disabled={submitting}
          onClick={() => void handleSubmit()}
          data-testid="launch-scenario-btn"
        >
          {submitting ? 'Launching...' : 'Launch scenario'}
        </Button>
      </div>
    </div>
  )
}
