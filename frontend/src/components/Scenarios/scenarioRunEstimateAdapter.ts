import type {
  ScenarioDefaultRunSizeEstimate,
  ScenarioRunEstimate,
  ScenarioRunEstimateDataset,
  ScenarioRunEstimateDatasetCap,
  ScenarioRunEstimateFactor,
  ScenarioRunEstimateResult,
} from '@/types'

function nextStableId(prefix: string, label: string, occurrences: Map<string, number>): string {
  const occurrence = (occurrences.get(label) ?? 0) + 1
  occurrences.set(label, occurrence)
  return `${prefix}:${label}:${occurrence}`
}

function mapFactors(
  componentId: string,
  factors: ScenarioDefaultRunSizeEstimate['components'][number]['factors'],
): ScenarioRunEstimateFactor[] {
  const occurrences = new Map<string, number>()
  return factors.map((factor) => ({
    id: nextStableId(`${componentId}:factor`, factor.label, occurrences),
    label: factor.label,
    count: factor.count,
  }))
}

function mapDatasetCaps(
  datasetId: string,
  caps: ScenarioDefaultRunSizeEstimate['datasets'][number]['configured_caps'],
): ScenarioRunEstimateDatasetCap[] {
  const occurrences = new Map<string, number>()
  return caps.map((cap) => ({
    id: nextStableId(`${datasetId}:cap`, cap.label, occurrences),
    label: cap.label,
    count: cap.count,
    configuredOn: cap.configured_on,
    datasetName: cap.dataset_name,
  }))
}

function mapDatasets(
  datasets: ScenarioDefaultRunSizeEstimate['datasets'],
): ScenarioRunEstimateDataset[] {
  const occurrences = new Map<string, number>()
  return datasets.map((dataset) => {
    const id = nextStableId('dataset', dataset.name, occurrences)
    return {
      id,
      name: dataset.name,
      kind: dataset.kind,
      logicalSeedGroupCount: dataset.logical_seed_group_count,
      selectedSeedGroupCount: dataset.selected_seed_group_count,
      configuredCaps: mapDatasetCaps(id, dataset.configured_caps),
      selectionNote: dataset.selection_note,
    }
  })
}

export function mapScenarioRunEstimate(
  response: ScenarioDefaultRunSizeEstimate,
  scope: ScenarioRunEstimate['scope'],
): ScenarioRunEstimateResult {
  if (response.status === 'unavailable') {
    return {
      status: 'unavailable',
      scope,
      label: scope === 'default'
        ? 'Default run size unavailable'
        : 'Configured run size unavailable',
      note: response.note ?? undefined,
    }
  }

  const componentOccurrences = new Map<string, number>()
  const estimate: ScenarioRunEstimate = {
    version: response.version,
    scope,
    total: response.total_attack_count,
    components: response.components.map((component) => {
      const id = nextStableId('component', component.label, componentOccurrences)
      return {
        id,
        label: component.label,
        count: component.count,
        factors: mapFactors(id, component.factors),
        isBaseline: component.is_baseline,
        note: component.note,
      }
    }),
    datasets: mapDatasets(response.datasets),
    note: response.note,
    retriesIncluded: response.retries_included,
  }

  return response.status === 'exact'
    ? { status: 'available', estimate }
    : { status: 'conditional', estimate }
}
