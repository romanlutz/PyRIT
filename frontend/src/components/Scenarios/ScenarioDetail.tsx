import { type FormEvent, useEffect, useMemo, useRef, useState } from 'react'

import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Badge,
  Button,
  Checkbox,
  Field,
  Input,
  MessageBar,
  MessageBarBody,
  Radio,
  RadioGroup,
  Select,
  Spinner,
  SpinButton,
  Text,
} from '@fluentui/react-components'
import { ArrowLeftRegular, ArrowSyncRegular, SettingsRegular } from '@fluentui/react-icons'
import { Link, useNavigate, useParams } from 'react-router-dom'

import MarkdownContent from '@/components/Markdown/MarkdownContent'
import ParameterField from '@/components/Parameters/ParameterField'
import {
  buildParametersFromForm,
  getInitialFormValues,
  type ParameterFormValue,
} from '@/components/Parameters/parameterForm'
import type { ViewName } from '@/components/Sidebar/Navigation'
import { datasetsApi, scenariosApi, targetsApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type {
  Parameter,
  RegisteredScenario,
  RunScenarioRequest,
  ScenarioRunEstimateResult,
  ScenarioRunSizeEstimateRequest,
  ScenarioRunEstimateState,
  TargetInstance,
} from '@/types'
import { fetchAllPages } from '@/utils/fetchAllPages'
import { routerPathParamValue } from '@/utils/routeParams'

import { useScenarioDetailStyles } from './ScenarioDetail.styles'
import { ScenarioRunEstimateDetails } from './ScenarioRunEstimate'
import { normalizeScenarioMarkdown } from './scenarioMarkdown'
import { mapScenarioRunEstimate } from './scenarioRunEstimateAdapter'
import {
  techniqueSetDisplayName,
  techniqueSetMembers,
  techniqueSetOptionLabel,
} from './scenarioTechniqueSets'

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
const ESTIMATE_DEBOUNCE_MS = 300
const TEXT_ADAPTIVE_SCENARIO_NAME = 'adaptive.text_adaptive'
const CUSTOM_TECHNIQUE_SET_VALUE = '__custom__'
const MAX_ATTEMPTS_PARAMETER_NAME = 'max_attempts_per_objective'
const MAX_ATTEMPTS_DISPLAY_LABEL = 'Techniques tried per objective'
const MAX_ATTEMPTS_DISPLAY_HINT = [
  'Maximum different compatible techniques Adaptive may try for one objective.',
  'Adaptive stops after the first success. This is separate from retries.',
].join(' ')

/** Resolves a Fluent `SpinButton` change event to a numeric value, preferring the parsed `value` over the raw `displayValue`. */
function resolveSpinButtonValue(data: { value?: number | null; displayValue?: string }, previous: number): number {
  if (typeof data.value === 'number') {
    return data.value
  }
  const parsed = data.displayValue !== undefined ? Number(data.displayValue) : NaN
  return Number.isFinite(parsed) ? parsed : previous
}

type LoadStatus = 'loading' | 'success' | 'not-found' | 'error'

type TechniqueSelection =
  | {
      mode: 'preset'
      preset: string
    }
  | {
      mode: 'custom'
    }

interface TechniqueOptions {
  presets: string[]
  concrete: string[]
  defaultSelection: TechniqueSelection
  initialCustomTechniques: string[]
}

/** Options rendered for technique selection: exclusive presets first, then concrete techniques. */
function uniqueTechniqueOptions(scenario: RegisteredScenario): TechniqueOptions {
  const aggregateNames = new Set(scenario.aggregate_techniques)
  const defaultIsPreset = aggregateNames.has(scenario.default_technique)
  const seenPresets = new Set<string>()
  const presets: string[] = []
  for (const name of scenario.aggregate_techniques) {
    if (!seenPresets.has(name)) {
      seenPresets.add(name)
      presets.push(name)
    }
  }
  const seenConcrete = new Set<string>()
  const concrete: string[] = []
  const concreteCandidates = defaultIsPreset
    ? scenario.all_techniques
    : [scenario.default_technique, ...scenario.all_techniques]
  for (const name of concreteCandidates) {
    if (!aggregateNames.has(name) && !seenConcrete.has(name)) {
      seenConcrete.add(name)
      concrete.push(name)
    }
  }
  const defaultSelection: TechniqueSelection = defaultIsPreset
    ? { mode: 'preset', preset: scenario.default_technique }
    : { mode: 'custom' }
  const initialCustomTechniques = defaultIsPreset ? [] : [scenario.default_technique]
  return { presets, concrete, defaultSelection, initialCustomTechniques }
}

function sameStringSet(left: string[], right: string[]): boolean {
  const leftSet = new Set(left)
  const rightSet = new Set(right)
  return leftSet.size === rightSet.size && [...leftSet].every((value) => rightSet.has(value))
}

function parameterDisplayLabel(parameter: Parameter, usesAdaptiveTechniqueSelection: boolean): string {
  return usesAdaptiveTechniqueSelection && parameter.name === MAX_ATTEMPTS_PARAMETER_NAME
    ? MAX_ATTEMPTS_DISPLAY_LABEL
    : parameter.name
}

function formatParameterPreview(value: ParameterFormValue | undefined): string {
  if (Array.isArray(value)) {
    return value.length > 0 ? value.join(', ') : 'Not set'
  }
  return value?.trim() || 'Not set'
}

function effectivePositiveInteger(
  value: ParameterFormValue | undefined,
  fallback: unknown,
): string | undefined {
  const candidate = typeof value === 'string' && value.trim().length > 0 ? value : fallback
  const parsed = typeof candidate === 'number' ? candidate : Number(candidate)
  return Number.isInteger(parsed) && parsed > 0 ? parsed.toLocaleString() : undefined
}

interface BuildRunRequestInput {
  scenario: RegisteredScenario
  targetName: string
  techniques: string[]
  dynamicParameters: Parameter[]
  scenarioParamValues: Record<string, ParameterFormValue>
  selectedDatasets: string[]
  maxDatasetSize: string
  maxConcurrency: number
  maxRetries: number
  includeBaseline: boolean
  labels: Record<string, string>
}

type BuildRunRequestResult =
  | {
      ok: true
      request: RunScenarioRequest
    }
  | {
      ok: false
      error: string
    }

type SuccessfulEstimateResult = Extract<
  ScenarioRunEstimateResult,
  { status: 'available' | 'conditional' }
>

type EstimateRequestState =
  | {
      status: 'resolved'
      requestKey: string
      result: ScenarioRunEstimateResult
    }
  | {
      status: 'error'
      requestKey: string
      error: string
    }

function buildRunRequest({
  scenario,
  targetName,
  techniques,
  dynamicParameters,
  scenarioParamValues,
  selectedDatasets,
  maxDatasetSize,
  maxConcurrency,
  maxRetries,
  includeBaseline,
  labels,
}: BuildRunRequestInput): BuildRunRequestResult {
  if (!targetName) {
    return { ok: false, error: 'Select a target.' }
  }
  if (techniques.length === 0) {
    return { ok: false, error: 'Select at least one technique.' }
  }
  if (scenario.default_datasets.length > 0 && selectedDatasets.length === 0) {
    return { ok: false, error: 'Select at least one dataset.' }
  }

  let scenarioParams: Record<string, unknown> | null = null
  if (dynamicParameters.length > 0) {
    const result = buildParametersFromForm(dynamicParameters, scenarioParamValues)
    if (!result.ok) {
      return result
    }
    scenarioParams = result.parameters
  }

  let maxDatasetSizeValue: number | undefined
  const trimmedMaxDatasetSize = maxDatasetSize.trim()
  if (trimmedMaxDatasetSize.length > 0) {
    const parsed = Number(trimmedMaxDatasetSize)
    if (!Number.isInteger(parsed) || parsed < 1) {
      return { ok: false, error: 'Max dataset size must be a positive integer.' }
    }
    maxDatasetSizeValue = parsed
  }
  if (
    !Number.isInteger(maxConcurrency)
    || maxConcurrency < MIN_MAX_CONCURRENCY
    || maxConcurrency > MAX_MAX_CONCURRENCY
  ) {
    return {
      ok: false,
      error: `Max concurrency must be an integer from ${MIN_MAX_CONCURRENCY} to ${MAX_MAX_CONCURRENCY}.`,
    }
  }
  if (
    !Number.isInteger(maxRetries)
    || maxRetries < MIN_MAX_RETRIES
    || maxRetries > MAX_MAX_RETRIES
  ) {
    return {
      ok: false,
      error: `Max retries must be an integer from ${MIN_MAX_RETRIES} to ${MAX_MAX_RETRIES}.`,
    }
  }

  const request: RunScenarioRequest = {
    scenario_name: scenario.scenario_name,
    target_name: targetName,
    techniques,
    max_concurrency: maxConcurrency,
    max_retries: maxRetries,
    include_baseline: includeBaseline,
    labels,
  }
  if (!sameStringSet(selectedDatasets, scenario.default_datasets)) {
    request.dataset_names = selectedDatasets
  }
  if (maxDatasetSizeValue !== undefined) {
    request.max_dataset_size = maxDatasetSizeValue
  }
  if (scenarioParams) {
    request.scenario_params = scenarioParams
  }
  return { ok: true, request }
}

function buildEstimateRequest(request: RunScenarioRequest): ScenarioRunSizeEstimateRequest {
  const estimateRequest: ScenarioRunSizeEstimateRequest = {
    target_name: request.target_name,
    techniques: request.techniques,
    include_baseline: request.include_baseline,
  }
  if (request.dataset_names !== undefined) {
    estimateRequest.dataset_names = request.dataset_names
  }
  if (request.max_dataset_size !== undefined) {
    estimateRequest.max_dataset_size = request.max_dataset_size
  }
  if (request.dataset_filters !== undefined) {
    estimateRequest.dataset_filters = request.dataset_filters
  }
  if (request.scenario_params !== undefined) {
    estimateRequest.scenario_params = request.scenario_params
  }
  return estimateRequest
}

type DatasetCatalogStatus = 'loading' | 'success' | 'error'

interface DatasetPickerProps {
  availableDatasets: string[]
  defaultDatasets: string[]
  selectedDatasets: string[]
  status: DatasetCatalogStatus
  error: string | null
  disabled: boolean
  invalid: boolean
  onChange: (name: string, checked: boolean) => void
  onRestoreDefaults: () => void
}

function DatasetPicker({
  availableDatasets,
  defaultDatasets,
  selectedDatasets,
  status,
  error,
  disabled,
  invalid,
  onChange,
  onRestoreDefaults,
}: DatasetPickerProps) {
  const styles = useScenarioDetailStyles()
  const [query, setQuery] = useState('')
  const selectedSet = useMemo(() => new Set(selectedDatasets), [selectedDatasets])
  const defaultSet = useMemo(() => new Set(defaultDatasets), [defaultDatasets])
  const orderedDatasets = useMemo(() => {
    const names = [...new Set([...availableDatasets, ...defaultDatasets])]
    return names.sort((left, right) => {
      const leftPriority = selectedSet.has(left) ? 0 : defaultSet.has(left) ? 1 : 2
      const rightPriority = selectedSet.has(right) ? 0 : defaultSet.has(right) ? 1 : 2
      return leftPriority - rightPriority || left.localeCompare(right)
    })
  }, [availableDatasets, defaultDatasets, defaultSet, selectedSet])
  const normalizedQuery = query.trim().toLocaleLowerCase()
  const visibleDatasets = normalizedQuery.length === 0
    ? orderedDatasets
    : orderedDatasets.filter((name) => name.toLocaleLowerCase().includes(normalizedQuery))
  const selectedCount = selectedDatasets.length

  return (
    <>
      <div className={styles.datasetPickerHeader}>
        <Text aria-live="polite">
          {selectedCount.toLocaleString()} dataset{selectedCount === 1 ? '' : 's'} selected
        </Text>
        <Button
          className={styles.touchTarget}
          appearance="subtle"
          icon={<ArrowSyncRegular />}
          disabled={disabled || sameStringSet(selectedDatasets, defaultDatasets)}
          onClick={onRestoreDefaults}
          data-testid="restore-default-datasets"
        >
          Restore defaults
        </Button>
      </div>
      <Field
        label="Search datasets"
        hint="Search registered datasets, then select the datasets this run should use."
        validationState={invalid ? 'error' : 'none'}
        validationMessage={invalid ? 'Select at least one dataset.' : undefined}
      >
        <Input
          className={styles.control}
          value={query}
          disabled={disabled}
          onChange={(_, data) => setQuery(data.value)}
          placeholder="Search datasets"
          aria-label="Search datasets"
          data-testid="dataset-search-input"
        />
      </Field>
      {status === 'loading' && (
        <Text size={200} className={styles.hint} role="status" data-testid="dataset-catalog-loading">
          Loading registered datasets…
        </Text>
      )}
      {status === 'error' && (
        <MessageBar intent="error" data-testid="dataset-catalog-error">
          <MessageBarBody role="alert">
            Registered datasets couldn’t be loaded. Scenario defaults remain available.
            {error ? ` ${error}` : ''}
          </MessageBarBody>
        </MessageBar>
      )}
      <div
        className={styles.datasetList}
        role="group"
        aria-label="Datasets"
        data-testid="dataset-picker-list"
      >
        {visibleDatasets.length > 0 ? (
          visibleDatasets.map((name) => (
            <Checkbox
              className={styles.selectionControl}
              key={name}
              label={name}
              checked={selectedSet.has(name)}
              disabled={disabled}
              onChange={(_, data) => onChange(name, data.checked === true)}
              data-testid={`dataset-${name}`}
            />
          ))
        ) : (
          <Text className={styles.datasetEmptyState}>No datasets match this search.</Text>
        )}
      </div>
    </>
  )
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
  // detail page directly to another.
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
      <section className={styles.root} data-testid="scenario-detail" aria-label="Scenario detail">
        <div className={styles.centeredState}>
          <Spinner label="Loading scenario..." />
        </div>
      </section>
    )
  }

  if (scenarioStatus === 'not-found') {
    return (
      <section className={styles.root} data-testid="scenario-detail" aria-label="Scenario detail">
        <div className={styles.content}>
          <Link to="/scenarios" className={styles.backLink}>
            <ArrowLeftRegular /> Back to scenarios
          </Link>
          <div className={styles.centeredState} data-testid="scenario-not-found">
            <Text size={400}>Scenario &quot;{decodedScenarioName}&quot; was not found</Text>
            <Text size={200}>It may have been renamed or is no longer registered.</Text>
          </div>
        </div>
      </section>
    )
  }

  if (scenarioStatus === 'error' || targetsError) {
    return (
      <section className={styles.root} data-testid="scenario-detail" aria-label="Scenario detail">
        <div className={styles.content}>
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
      </section>
    )
  }

  // scenarioStatus === 'success' from here on; both values are set together.
  if (!scenario) {
    return null
  }

  if (targets.length === 0) {
    return (
      <section className={styles.root} data-testid="scenario-detail" aria-label="Scenario detail">
        <div className={styles.content}>
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
      </section>
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
  const formId = `scenario-launch-${encodeURIComponent(scenario.scenario_name).replace(/%/g, '-')}`

  const { presets, concrete, defaultSelection, initialCustomTechniques } = useMemo(
    () => uniqueTechniqueOptions(scenario),
    [scenario],
  )
  const dynamicParameters = useMemo(
    () => scenario.supported_parameters.filter(
      (parameter) => !COMMON_SCENARIO_PARAMETER_NAMES.has(parameter.name),
    ),
    [scenario.supported_parameters],
  )
  const isBaselineForbidden = scenario.baseline_policy === 'forbidden'
  const usesAdaptiveTechniqueSelection = scenario.scenario_name === TEXT_ADAPTIVE_SCENARIO_NAME

  const [targetName, setTargetName] = useState(() => {
    if (activeTarget && targets.some((target) =>
      target.target_registry_name === activeTarget.target_registry_name)) {
      return activeTarget.target_registry_name
    }
    return targets[0].target_registry_name
  })
  const [techniqueSelection, setTechniqueSelection] = useState<TechniqueSelection>(() => defaultSelection)
  const [customTechniques, setCustomTechniques] = useState<string[]>(() => initialCustomTechniques)
  const [baselineChecked, setBaselineChecked] = useState(
    () => !isBaselineForbidden && scenario.include_baseline_by_default,
  )
  const [availableDatasets, setAvailableDatasets] = useState<string[]>(() => [...scenario.default_datasets])
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>(() => [...scenario.default_datasets])
  const [datasetCatalogStatus, setDatasetCatalogStatus] = useState<DatasetCatalogStatus>('loading')
  const [datasetCatalogError, setDatasetCatalogError] = useState<string | null>(null)
  const [maxDatasetSize, setMaxDatasetSize] = useState('')
  const [maxConcurrency, setMaxConcurrency] = useState(DEFAULT_MAX_CONCURRENCY)
  const [maxRetries, setMaxRetries] = useState(DEFAULT_MAX_RETRIES)
  const [scenarioParamValues, setScenarioParamValues] = useState<Record<string, ParameterFormValue>>(() =>
    getInitialFormValues(dynamicParameters),
  )
  const [validationError, setValidationError] = useState<string | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [estimateRequestState, setEstimateRequestState] = useState<EstimateRequestState | null>(null)
  const [lastGoodEstimate, setLastGoodEstimate] = useState<SuccessfulEstimateResult | null>(null)
  // Synchronous guard against a double-submit racing ahead of the state update.
  const isSubmittingRef = useRef(false)
  const estimateSequenceRef = useRef(0)
  const customTechniquesInitializedRef = useRef(defaultSelection.mode === 'custom')

  const techniques = useMemo(
    () => techniqueSelection.mode === 'preset'
      ? [techniqueSelection.preset]
      : customTechniques,
    [customTechniques, techniqueSelection],
  )
  const requestResult = useMemo(
    () => buildRunRequest({
      scenario,
      targetName,
      techniques,
      dynamicParameters,
      scenarioParamValues,
      selectedDatasets,
      maxDatasetSize,
      maxConcurrency,
      maxRetries,
      includeBaseline: isBaselineForbidden ? false : baselineChecked,
      labels,
    }),
    [
      baselineChecked,
      dynamicParameters,
      isBaselineForbidden,
      labels,
      maxConcurrency,
      maxDatasetSize,
      maxRetries,
      scenario,
      scenarioParamValues,
      selectedDatasets,
      targetName,
      techniques,
    ],
  )
  const estimateRequest = useMemo(
    () => requestResult.ok ? buildEstimateRequest(requestResult.request) : null,
    [requestResult],
  )
  const estimateRequestKey = useMemo(
    () => estimateRequest === null
      ? null
      : JSON.stringify({ scenarioName: scenario.scenario_name, request: estimateRequest }),
    [estimateRequest, scenario.scenario_name],
  )

  useEffect(() => {
    let cancelled = false
    datasetsApi
      .listDatasets()
      .then((response) => {
        if (cancelled) return
        setAvailableDatasets([...new Set([
          ...scenario.default_datasets,
          ...response.items.map((item) => item.name),
        ])])
        setDatasetCatalogStatus('success')
        setDatasetCatalogError(null)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setAvailableDatasets([...scenario.default_datasets])
        setDatasetCatalogStatus('error')
        setDatasetCatalogError(toApiError(err).detail)
      })
    return () => {
      cancelled = true
    }
  }, [scenario.default_datasets])

  useEffect(() => {
    if (estimateRequest === null || estimateRequestKey === null) {
      return
    }

    const requestSequence = estimateSequenceRef.current + 1
    estimateSequenceRef.current = requestSequence
    const controller = new AbortController()

    const debounceTimer = window.setTimeout(() => {
      scenariosApi
        .estimateRun(scenario.scenario_name, estimateRequest, controller.signal)
        .then((response) => {
          if (
            controller.signal.aborted
            || requestSequence !== estimateSequenceRef.current
          ) {
            return
          }
          const result = mapScenarioRunEstimate(response, 'request')
          setEstimateRequestState({
            status: 'resolved',
            requestKey: estimateRequestKey,
            result,
          })
          if (result.status === 'available' || result.status === 'conditional') {
            setLastGoodEstimate(result)
          }
        })
        .catch((err: unknown) => {
          if (
            controller.signal.aborted
            || requestSequence !== estimateSequenceRef.current
          ) {
            return
          }
          setEstimateRequestState({
            status: 'error',
            requestKey: estimateRequestKey,
            error: toApiError(err).detail,
          })
        })
    }, ESTIMATE_DEBOUNCE_MS)

    return () => {
      window.clearTimeout(debounceTimer)
      controller.abort()
    }
  }, [estimateRequest, estimateRequestKey, scenario.scenario_name])

  let estimateState: ScenarioRunEstimateState
  if (!requestResult.ok) {
    estimateState = {
      status: 'unavailable',
      scope: 'request',
      label: 'Complete the required configuration to request an estimate.',
      note: requestResult.error,
    }
  } else if (
    estimateRequestState?.requestKey === estimateRequestKey
    && estimateRequestState.status === 'resolved'
  ) {
    estimateState = estimateRequestState.result
  } else if (
    estimateRequestState?.requestKey === estimateRequestKey
    && estimateRequestState.status === 'error'
  ) {
    estimateState = lastGoodEstimate
      ? {
          status: 'stale',
          estimate: lastGoodEstimate.estimate,
          label: 'Showing the last successful estimate.',
          error: estimateRequestState.error,
        }
      : {
          status: 'unavailable',
          scope: 'request',
          label: 'Run size couldn’t be updated.',
          note: estimateRequestState.error,
        }
  } else if (lastGoodEstimate) {
    estimateState = {
      status: 'refreshing',
      estimate: lastGoodEstimate.estimate,
      label: 'Updating for the current configuration…',
    }
  } else {
    estimateState = { status: 'loading', scope: 'request' }
  }
  const estimateRequestBlocked = estimateRequestState?.requestKey === estimateRequestKey
    && estimateRequestState.status === 'error'

  const handleTechniqueModeChange = (value: string): void => {
    if (value === CUSTOM_TECHNIQUE_SET_VALUE) {
      if (!customTechniquesInitializedRef.current) {
        const members = techniqueSelection.mode === 'preset'
          ? techniqueSetMembers(scenario, techniqueSelection.preset)
          : []
        const concreteSet = new Set(concrete)
        setCustomTechniques(members.filter((member) => concreteSet.has(member)))
        customTechniquesInitializedRef.current = true
      }
      setTechniqueSelection({ mode: 'custom' })
    } else {
      setTechniqueSelection({ mode: 'preset', preset: value })
    }
    setValidationError(null)
  }

  const handleConcreteChange = (name: string, checked: boolean): void => {
    setCustomTechniques((current) => {
      if (checked) {
        return current.includes(name)
          ? current
          : [...current, name]
      }
      return current.filter((technique) => technique !== name)
    })
    setValidationError(null)
  }

  const handleDatasetChange = (name: string, checked: boolean): void => {
    setSelectedDatasets((current) => {
      if (checked) {
        return current.includes(name) ? current : [...current, name]
      }
      return current.filter((dataset) => dataset !== name)
    })
    setValidationError(null)
  }

  const updateScenarioParam = (name: string, value: ParameterFormValue): void => {
    setScenarioParamValues((current) => ({ ...current, [name]: value }))
  }

  const handleSubmit = async (): Promise<void> => {
    if (isSubmittingRef.current) {
      return
    }

    setApiError(null)
    if (!requestResult.ok) {
      setValidationError(requestResult.error)
      return
    }

    isSubmittingRef.current = true
    setSubmitting(true)
    setValidationError(null)

    try {
      const summary = await scenariosApi.startRun(requestResult.request)
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

  const handleFormSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault()
    void handleSubmit()
  }

  const techniqueSelectionInvalid =
    techniqueSelection.mode === 'custom' && customTechniques.length === 0
  const datasetSelectionInvalid = scenario.default_datasets.length > 0 && selectedDatasets.length === 0
  const datasetsAreDefaults = sameStringSet(selectedDatasets, scenario.default_datasets)
  const presetMembers = techniqueSelection.mode === 'preset'
    ? techniqueSetMembers(scenario, techniqueSelection.preset)
    : []
  const effectiveMaxAdaptiveAttempts = effectivePositiveInteger(
    scenarioParamValues.max_attempts_per_objective,
    dynamicParameters.find(
      (parameter) => parameter.name === 'max_attempts_per_objective',
    )?.default,
  )

  return (
    <section
      className={styles.root}
      data-testid="scenario-detail"
      aria-labelledby="scenario-detail-title"
    >
      <div className={styles.content}>
        <Link to="/scenarios" className={styles.backLink}>
          <ArrowLeftRegular /> Back to scenarios
        </Link>

        <div className={styles.headerText}>
          <Text id="scenario-detail-title" as="h1" size={600} weight="semibold">
            {scenario.scenario_name}
          </Text>
          <Text size={200} className={styles.scenarioMetadata}>
            {scenario.scenario_type} · v{scenario.scenario_version}
          </Text>
          <MarkdownContent
            content={normalizeScenarioMarkdown(
              scenario.description_markdown || scenario.description,
            )}
            className={styles.description}
            testId="scenario-detail-description"
          />
        </div>

        <div className={styles.layout}>
          <form
            id={formId}
            className={styles.formColumn}
            aria-label="Scenario run configuration"
            onSubmit={handleFormSubmit}
            noValidate
          >
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

            <section className={styles.section} aria-labelledby="target-section-title">
              <Text id="target-section-title" as="h2" size={400} weight="semibold">Target</Text>
              <Field hint="The registered target this scenario will run against.">
                <Select
                  className={styles.control}
                  value={targetName}
                  disabled={submitting}
                  onChange={(_, data) => setTargetName(data.value)}
                  data-testid="scenario-target-select"
                  aria-label="Target"
                >
                  {targets.map((target) => (
                    <option key={target.target_registry_name} value={target.target_registry_name}>
                      {target.target_registry_name}
                    </option>
                  ))}
                </Select>
              </Field>
            </section>

            <section className={styles.section} aria-labelledby="techniques-section-title">
              <Text id="techniques-section-title" as="h2" size={400} weight="semibold">
                Techniques
              </Text>
              <Text size={200} className={styles.hint}>
                Choose a predefined set, or choose Custom to select techniques individually.
              </Text>
              {usesAdaptiveTechniqueSelection && (
                <>
                  <Text size={200} className={styles.hint}>
                    Core, Extra, Light, Multi-turn, and Single-turn reflect tags on PyRIT&apos;s registered
                    techniques. All is generated from the catalog; Recommended is curated for this scenario.
                  </Text>
                  <MessageBar intent="info">
                    <MessageBarBody>
                      Adaptive uses these as a candidate pool. It tracks one progress step per compatible objective
                      {effectiveMaxAdaptiveAttempts
                        ? ` and may try up to ${effectiveMaxAdaptiveAttempts} compatible techniques for that objective, stopping after the first success`
                        : ''}
                      . Adding techniques changes the candidate pool, not the number of progress steps; compatibility
                      can still change how many objectives can run.
                    </MessageBarBody>
                  </MessageBar>
                </>
              )}
              <div className={styles.techniqueGroups}>
                <Field label="Technique set">
                  <RadioGroup
                    value={techniqueSelection.mode === 'preset'
                      ? techniqueSelection.preset
                      : CUSTOM_TECHNIQUE_SET_VALUE}
                    onChange={(_, data) => handleTechniqueModeChange(data.value)}
                    aria-label="Technique set"
                  >
                    {presets.map((name) => (
                      <Radio
                        className={styles.selectionControl}
                        key={name}
                        label={techniqueSetOptionLabel(scenario, name)}
                        value={name}
                        disabled={submitting}
                        data-testid={`technique-${name}`}
                      />
                    ))}
                    <Radio
                      className={styles.selectionControl}
                      label="Custom"
                      value={CUSTOM_TECHNIQUE_SET_VALUE}
                      disabled={submitting}
                      data-testid="technique-mode-custom"
                    />
                  </RadioGroup>
                </Field>
                {techniqueSelection.mode === 'preset' && (
                  <div
                    className={styles.resolvedMembers}
                    data-testid="selected-technique-set-members"
                    aria-live="polite"
                  >
                    <Text size={200} weight="semibold">Included techniques</Text>
                    {presetMembers.length > 0 ? (
                      <div className={styles.previewBadges}>
                        {presetMembers.map((name) => (
                          <Badge key={name} appearance="outline">{name}</Badge>
                        ))}
                      </div>
                    ) : (
                      <Text size={200} className={styles.hint}>
                        No concrete members were supplied for this technique set.
                      </Text>
                    )}
                  </div>
                )}
                {techniqueSelection.mode === 'custom' && (
                  <Field
                    label="Individual techniques"
                    validationState={techniqueSelectionInvalid ? 'error' : 'none'}
                    validationMessage={techniqueSelectionInvalid
                      ? 'Select at least one technique.'
                      : undefined}
                  >
                    {concrete.length > 0 ? (
                      <div className={styles.checkboxGroup} role="group" aria-label="Individual techniques">
                        {concrete.map((name) => (
                          <Checkbox
                            className={styles.selectionControl}
                            key={name}
                            label={name}
                            checked={customTechniques.includes(name)}
                            disabled={submitting}
                            onChange={(_, data) => handleConcreteChange(name, data.checked === true)}
                            data-testid={`technique-${name}`}
                          />
                        ))}
                      </div>
                    ) : (
                      <Text size={200} className={styles.hint}>
                        No concrete techniques are registered for custom selection.
                      </Text>
                    )}
                  </Field>
                )}
              </div>
            </section>

            <section className={styles.section} aria-labelledby="baseline-section-title">
              <Text id="baseline-section-title" as="h2" size={400} weight="semibold">
                Baseline
              </Text>
              <Field
                hint={isBaselineForbidden
                  ? 'This scenario does not support sending objectives directly without an attack technique, so a direct comparison cannot be included.'
                  : 'Also send each selected objective directly, without an attack technique. This provides a comparison point for measuring whether the selected techniques improve results and adds one planned attack per objective.'}
              >
                <Checkbox
                  className={styles.selectionControl}
                  checked={baselineChecked}
                  disabled={submitting || isBaselineForbidden}
                  label="Include direct baseline comparison"
                  onChange={(_, data) => setBaselineChecked(data.checked === true)}
                  data-testid="baseline-checkbox"
                />
              </Field>
            </section>

            {dynamicParameters.length > 0 && (
              <section className={styles.section} aria-labelledby="parameters-section-title">
                <Text id="parameters-section-title" as="h2" size={400} weight="semibold">
                  Scenario parameters
                </Text>
                <div className={styles.dynamicParameters}>
                  {dynamicParameters.map((parameter) => (
                    <ParameterField
                      key={parameter.name}
                      parameter={parameter}
                      value={scenarioParamValues[parameter.name]}
                      disabled={submitting}
                      onChange={updateScenarioParam}
                      displayLabel={usesAdaptiveTechniqueSelection
                        && parameter.name === MAX_ATTEMPTS_PARAMETER_NAME
                        ? MAX_ATTEMPTS_DISPLAY_LABEL
                        : undefined}
                      displayHint={usesAdaptiveTechniqueSelection
                        && parameter.name === MAX_ATTEMPTS_PARAMETER_NAME
                        ? MAX_ATTEMPTS_DISPLAY_HINT
                        : undefined}
                      testIdPrefix="scenario-param"
                    />
                  ))}
                </div>
              </section>
            )}

            <section className={styles.section} aria-labelledby="datasets-section-title">
              <Text id="datasets-section-title" as="h2" size={400} weight="semibold">
                Datasets
              </Text>
              <Text size={200} className={styles.hint}>
                Choose the registered datasets that provide objectives for this run.
              </Text>
              <DatasetPicker
                availableDatasets={availableDatasets}
                defaultDatasets={scenario.default_datasets}
                selectedDatasets={selectedDatasets}
                status={datasetCatalogStatus}
                error={datasetCatalogError}
                disabled={submitting}
                invalid={datasetSelectionInvalid}
                onChange={handleDatasetChange}
                onRestoreDefaults={() => {
                  setSelectedDatasets([...scenario.default_datasets])
                  setValidationError(null)
                }}
              />
            </section>

            <Accordion collapsible className={styles.advancedSection}>
              <AccordionItem value="advanced">
                <AccordionHeader>Advanced options</AccordionHeader>
                <AccordionPanel>
                  <div className={styles.advancedFields}>
                    <Field
                      label="Max dataset size"
                      hint="Optional cap applied to each selected dataset. Leave blank for no additional cap."
                    >
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
                    <Field
                      label="Max retries"
                      hint={usesAdaptiveTechniqueSelection
                        ? 'Maximum times to resume the scenario after an exception. This is separate from Adaptive trying another technique.'
                        : 'Maximum times to resume the scenario after an exception.'}
                    >
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
          </form>

          <aside className={styles.previewRail} aria-labelledby="run-preview-title">
            <div className={styles.previewHeader}>
              <Text id="run-preview-title" as="h2" size={500} weight="semibold">Run preview</Text>
              <Text size={200} className={styles.hint}>
                Review the exact configuration used for this run.
              </Text>
            </div>
            <dl className={styles.previewList}>
              <div className={styles.previewGroup}>
                <dt>Target</dt>
                <dd>{targetName}</dd>
              </div>
              <div className={styles.previewGroup}>
                <dt>Techniques</dt>
                <dd>
                  {techniqueSelection.mode === 'preset' ? (
                    <div className={styles.previewStack}>
                      <Text weight="semibold">
                        Technique set: {techniqueSetDisplayName(scenario, techniqueSelection.preset)}
                      </Text>
                      {presetMembers.length > 0 && (
                        <Text size={200} className={styles.hint}>
                          Resolves to {presetMembers.join(', ')}
                        </Text>
                      )}
                    </div>
                  ) : customTechniques.length > 0 ? (
                    <div className={styles.previewBadges}>
                      {customTechniques.map((name) => (
                        <Badge key={name} appearance="outline">{name}</Badge>
                      ))}
                    </div>
                  ) : (
                    <Text className={styles.errorText}>No custom techniques selected</Text>
                  )}
                </dd>
              </div>
              <div className={styles.previewGroup}>
                <dt>Datasets</dt>
                <dd>
                  <div className={styles.previewStack}>
                    <Text>
                      {selectedDatasets.length > 0 ? selectedDatasets.join(', ') : 'No datasets selected'}
                    </Text>
                    <Text size={200} className={styles.hint}>
                      {datasetsAreDefaults ? 'Scenario defaults' : 'Selected datasets'}
                      {maxDatasetSize.trim() ? ` · capped at ${maxDatasetSize.trim()} each` : ''}
                    </Text>
                  </div>
                </dd>
              </div>
              <div className={styles.previewGroup}>
                <dt>Scenario parameters</dt>
                <dd>
                  {dynamicParameters.length > 0 ? (
                    <dl className={styles.parameterPreview}>
                      {dynamicParameters.map((parameter) => (
                        <div className={styles.parameterPreviewRow} key={parameter.name}>
                          <dt>{parameterDisplayLabel(parameter, usesAdaptiveTechniqueSelection)}</dt>
                          <dd>{formatParameterPreview(scenarioParamValues[parameter.name])}</dd>
                        </div>
                      ))}
                    </dl>
                  ) : (
                    'No scenario-specific parameters'
                  )}
                </dd>
              </div>
              <div className={styles.previewGroup}>
                <dt>Baseline</dt>
                <dd>
                  {isBaselineForbidden
                    ? 'Not included — this scenario does not support direct comparison'
                    : baselineChecked
                      ? 'Included — direct objective without an attack technique'
                      : 'Not included'}
                </dd>
              </div>
            </dl>
            <div className={styles.estimateGroup}>
              <Text as="h3" size={400} weight="semibold">Planned run size</Text>
              <ScenarioRunEstimateDetails
                state={estimateState}
                idPrefix={`${formId}-estimate`}
              />
            </div>
            <div className={styles.previewActions}>
              <Button
                className={styles.launchButton}
                appearance="primary"
                type="submit"
                form={formId}
                disabled={
                  submitting
                  || techniqueSelectionInvalid
                  || datasetSelectionInvalid
                  || estimateRequestBlocked
                }
                data-testid="launch-scenario-btn"
              >
                {submitting ? 'Launching...' : 'Launch scenario'}
              </Button>
            </div>
          </aside>
        </div>
      </div>
    </section>
  )
}
