import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  Badge,
  Button,
  Input,
  mergeClasses,
  MessageBar,
  MessageBarBody,
  Spinner,
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableHeaderCell,
  TableRow,
  Text,
} from '@fluentui/react-components'
import {
  ArrowSyncRegular,
  ChevronDownRegular,
  ChevronRightRegular,
  SearchRegular,
} from '@fluentui/react-icons'
import { Link } from 'react-router-dom'

import MarkdownContent from '@/components/Markdown/MarkdownContent'
import { scenariosApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type { RegisteredScenario, ScenarioDatasetSummary } from '@/types'
import { fetchAllPages } from '@/utils/fetchAllPages'

import { useScenarioCatalogStyles } from './ScenarioCatalog.styles'
import {
  ScenarioRunEstimateDetails,
  ScenarioRunEstimateSummary,
} from './ScenarioRunEstimate'
import { normalizeScenarioMarkdown } from './scenarioMarkdown'
import { mapScenarioRunEstimate } from './scenarioRunEstimateAdapter'

/** Items requested per catalog page while paging through the full list. */
const CATALOG_PAGE_SIZE = 200

function matchesSearch(scenario: RegisteredScenario, query: string): boolean {
  if (!query) {
    return true
  }
  const haystack = [
    scenario.scenario_name,
    scenario.description,
    scenario.description_markdown,
    scenario.scenario_type,
    scenario.default_technique,
    ...scenario.default_techniques,
    ...scenario.aggregate_techniques,
    ...Object.values(scenario.aggregate_technique_expansions).flat(),
    ...scenario.all_techniques,
    ...scenario.default_datasets,
    ...scenario.default_dataset_summaries.flatMap((dataset) => [
      dataset.name,
      dataset.selection_note ?? '',
      ...dataset.configured_caps.map((cap) => cap.label),
    ]),
  ]
    .join(' ')
    .toLowerCase()
  return haystack.includes(query.toLowerCase())
}

function uniqueNames(names: string[]): string[] {
  return [...new Set(names)]
}

function formatCount(value: number): string {
  return value.toLocaleString()
}

function formatSeedGroupCount(value: number): string {
  return `${formatCount(value)} selected seed group${value === 1 ? '' : 's'}`
}

function DefaultDatasetSizeSummary({
  datasets,
  hasDeclaredDatasets,
}: {
  datasets: ScenarioDatasetSummary[]
  hasDeclaredDatasets: boolean
}) {
  const styles = useScenarioCatalogStyles()

  if (datasets.length === 0) {
    return (
      <Text weight="semibold">
        {hasDeclaredDatasets ? 'Population counts unavailable' : 'No default dataset'}
      </Text>
    )
  }

  if (datasets.length === 1) {
    const dataset = datasets[0]
    return (
      <div className={styles.compactStack}>
        <Text weight="semibold">{formatSeedGroupCount(dataset.selected_seed_group_count)}</Text>
        <Text size={200} className={styles.secondaryText}>
          of {formatCount(dataset.logical_seed_group_count)} available · {dataset.name}
        </Text>
      </div>
    )
  }

  return (
    <Text size={200} weight="semibold">
      {datasets
        .map((dataset) => `${dataset.name}: ${formatCount(dataset.selected_seed_group_count)}`)
        .join(' · ')}
    </Text>
  )
}

function baselineDescription(scenario: RegisteredScenario): string {
  if (scenario.baseline_policy === 'forbidden') {
    return 'Baseline execution is not supported for this scenario.'
  }
  if (scenario.include_baseline_by_default) {
    return 'Baseline execution is allowed and included by default.'
  }
  return 'Baseline execution is allowed and excluded by default.'
}

function DatasetPopulation({ dataset }: { dataset: ScenarioDatasetSummary }) {
  const styles = useScenarioCatalogStyles()

  return (
    <article className={styles.datasetCard}>
      <div className={styles.datasetHeader}>
        <Text weight="semibold">{dataset.name}</Text>
        <Badge appearance="tint" color="informative">{dataset.kind}</Badge>
      </div>
      <dl className={styles.datasetCounts}>
        <div className={styles.datasetCountRow}>
          <dt>Logical seed groups</dt>
          <dd>{formatCount(dataset.logical_seed_group_count)}</dd>
        </div>
        <div className={styles.datasetCountRow}>
          <dt>Selected seed groups</dt>
          <dd>{formatCount(dataset.selected_seed_group_count)}</dd>
        </div>
      </dl>
      {dataset.configured_caps.length > 0 && (
        <div className={styles.metadataGroup}>
          <Text size={200} weight="semibold">Configured caps</Text>
          <ul className={styles.capList}>
            {dataset.configured_caps.map((cap) => (
              <li key={`${cap.label}:${cap.configured_on}:${cap.dataset_name ?? ''}:${cap.count}`}>
                <Text size={200}>
                  {cap.label}: {formatCount(cap.count)}
                  {' '}({cap.configured_on}{cap.dataset_name ? `: ${cap.dataset_name}` : ''})
                </Text>
              </li>
            ))}
          </ul>
        </div>
      )}
      {dataset.selection_note && (
        <Text size={200} className={styles.secondaryText}>{dataset.selection_note}</Text>
      )}
    </article>
  )
}

interface ScenarioCatalogRowProps {
  scenario: RegisteredScenario
  expanded: boolean
  onToggle: (scenarioName: string) => void
}

function ScenarioCatalogRow({ scenario, expanded, onToggle }: ScenarioCatalogRowProps) {
  const styles = useScenarioCatalogStyles()
  const detailsId = `scenario-details-${encodeURIComponent(scenario.scenario_name).replace(/%/g, '-')}`
  const aggregateTechniques = uniqueNames(scenario.aggregate_techniques)
  const defaultIsAggregate = aggregateTechniques.includes(scenario.default_technique)
  const concreteTechniques = uniqueNames(scenario.all_techniques)
  const defaultConcreteTechniques = uniqueNames(scenario.default_techniques)
  const description = normalizeScenarioMarkdown(
    scenario.description_markdown || scenario.description,
  )
  const estimateState = mapScenarioRunEstimate(scenario.default_run_size, 'default')
  const scenarioPath = `/scenarios/${encodeURIComponent(scenario.scenario_name)}`

  return (
    <>
      <TableRow
        className={mergeClasses(styles.summaryRow, expanded && styles.expandedSummaryRow)}
        data-testid={`scenario-card-${scenario.scenario_name}`}
      >
        <TableCell
          className={mergeClasses(styles.tableCell, styles.tableCellPadding, 'scenario-catalog-cell-padding')}
        >
          <Text className={styles.mobileLabel} size={200} weight="semibold">
            Scenario / purpose
          </Text>
          <div className={styles.scenarioSummary}>
            <Link to={scenarioPath} className={styles.scenarioLink}>
              {scenario.scenario_name}
            </Link>
            <Text size={200} className={styles.scenarioType}>
              {scenario.scenario_type} · v{scenario.scenario_version}
            </Text>
            <Text size={200} className={styles.purposePreview}>{scenario.description}</Text>
          </div>
        </TableCell>
        <TableCell
          className={mergeClasses(styles.tableCell, styles.tableCellPadding, 'scenario-catalog-cell-padding')}
        >
          <Text className={styles.mobileLabel} size={200} weight="semibold">
            Default dataset size
          </Text>
          <DefaultDatasetSizeSummary
            datasets={scenario.default_dataset_summaries}
            hasDeclaredDatasets={scenario.default_datasets.length > 0}
          />
        </TableCell>
        <TableCell
          className={mergeClasses(styles.tableCell, styles.tableCellPadding, 'scenario-catalog-cell-padding')}
        >
          <Text className={styles.mobileLabel} size={200} weight="semibold">
            Default techniques
          </Text>
          <Text weight="semibold">
            {defaultConcreteTechniques.length === 0
              ? 'No default techniques'
              : `${defaultConcreteTechniques.length} technique${defaultConcreteTechniques.length === 1 ? '' : 's'}`}
          </Text>
        </TableCell>
        <TableCell
          className={mergeClasses(styles.tableCell, styles.tableCellPadding, 'scenario-catalog-cell-padding')}
        >
          <Text className={styles.mobileLabel} size={200} weight="semibold">
            Default run size
          </Text>
          <ScenarioRunEstimateSummary state={estimateState} />
        </TableCell>
        <TableCell
          className={mergeClasses(
            styles.tableCell,
            styles.tableCellPadding,
            styles.actionCell,
            'scenario-catalog-cell-padding',
          )}
        >
          <Text className={styles.mobileLabel} size={200} weight="semibold">
            Action
          </Text>
          <div className={styles.actionGroup}>
            <Button
              className={styles.touchTarget}
              appearance="subtle"
              icon={expanded ? <ChevronDownRegular /> : <ChevronRightRegular />}
              aria-expanded={expanded}
              aria-controls={detailsId}
              onClick={() => onToggle(scenario.scenario_name)}
            >
              {expanded ? 'Hide details' : 'Show details'}
            </Button>
            <Link className={styles.actionLink} to={scenarioPath}>
              Configure run
            </Link>
          </div>
        </TableCell>
      </TableRow>
      {expanded && (
        <TableRow className={styles.detailsRow}>
          <TableCell className={styles.detailsCell} colSpan={5}>
            <div
              className={styles.detailsPanel}
              id={detailsId}
              role="region"
              aria-label={`${scenario.scenario_name} details`}
            >
              <section className={mergeClasses(styles.detailGroup, styles.descriptionGroup)}>
                <Text as="h3" size={400} weight="semibold">Purpose and behavior</Text>
                <MarkdownContent
                  content={description}
                  testId={`scenario-description-${scenario.scenario_name}`}
                />
              </section>
              <section className={styles.detailGroup}>
                <Text as="h3" size={400} weight="semibold">Default run size</Text>
                <ScenarioRunEstimateDetails
                  state={estimateState}
                  idPrefix={`${detailsId}-estimate`}
                />
              </section>
              <section className={styles.detailGroup}>
                <Text as="h3" size={400} weight="semibold">Techniques</Text>
                <div className={styles.metadataGroup}>
                  <Text weight="semibold">
                    {defaultIsAggregate ? 'Default aggregate preset' : 'Default concrete technique'}
                  </Text>
                  <div className={styles.badgeGroup}>
                    <Badge appearance="tint" color="brand">{scenario.default_technique}</Badge>
                  </div>
                </div>
                <div className={styles.metadataGroup}>
                  <Text weight="semibold">Included by default</Text>
                  {defaultConcreteTechniques.length > 0 ? (
                    <div className={styles.badgeGroup}>
                      {defaultConcreteTechniques.map((technique) => (
                        <Badge key={technique} appearance="outline">{technique}</Badge>
                      ))}
                    </div>
                  ) : (
                    <Text size={200} className={styles.secondaryText}>
                      No concrete default members were supplied.
                    </Text>
                  )}
                </div>
                <div className={styles.metadataGroup}>
                  <Text weight="semibold">Aggregate presets and members</Text>
                  {aggregateTechniques.length > 0 ? (
                    <ul className={styles.presetList}>
                      {aggregateTechniques.map((technique) => (
                        <li className={styles.presetItem} key={technique}>
                          <Badge appearance="tint">{technique}</Badge>
                          <Text size={200} className={styles.secondaryText}>
                            {(scenario.aggregate_technique_expansions[technique] ?? []).length > 0
                              ? scenario.aggregate_technique_expansions[technique].join(', ')
                              : 'No concrete members supplied.'}
                          </Text>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <Text size={200} className={styles.secondaryText}>None registered.</Text>
                  )}
                </div>
                <div className={styles.metadataGroup}>
                  <Text weight="semibold">Compatible concrete techniques</Text>
                  {concreteTechniques.length > 0 ? (
                    <div className={styles.badgeGroup}>
                      {concreteTechniques.map((technique) => (
                        <Badge key={technique} appearance="outline">{technique}</Badge>
                      ))}
                    </div>
                  ) : (
                    <Text size={200} className={styles.secondaryText}>None registered.</Text>
                  )}
                </div>
              </section>
              <section className={styles.detailGroup}>
                <Text as="h3" size={400} weight="semibold">
                  Default datasets and populations
                </Text>
                {scenario.default_dataset_summaries.length > 0 ? (
                  <ul className={styles.datasetList}>
                    {scenario.default_dataset_summaries.map((dataset) => (
                      <li key={`${dataset.kind}:${dataset.name}`}>
                        <DatasetPopulation dataset={dataset} />
                      </li>
                    ))}
                  </ul>
                ) : scenario.default_datasets.length > 0 ? (
                  <ul className={styles.datasetList}>
                    {uniqueNames(scenario.default_datasets).map((dataset) => (
                      <li key={dataset}>
                        <article className={styles.datasetCard}>
                          <div className={styles.datasetHeader}>
                            <Text weight="semibold">{dataset}</Text>
                            <Badge appearance="tint" color="warning">
                              Population unavailable
                            </Badge>
                          </div>
                          <Text size={200} className={styles.secondaryText}>
                            Population counts and configured caps aren’t available.
                          </Text>
                        </article>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <Text size={200} className={styles.secondaryText}>
                    This scenario does not declare a default dataset.
                  </Text>
                )}
              </section>
              <section className={styles.detailGroup}>
                <Text as="h3" size={400} weight="semibold">Baseline policy</Text>
                <div className={styles.badgeGroup}>
                  <Badge
                    appearance="tint"
                    color={scenario.baseline_policy === 'forbidden' ? 'warning' : 'subtle'}
                  >
                    {scenario.baseline_policy}
                  </Badge>
                  <Badge appearance="outline">
                    {scenario.include_baseline_by_default ? 'Included by default' : 'Excluded by default'}
                  </Badge>
                </div>
                <Text size={200} className={styles.secondaryText}>
                  {baselineDescription(scenario)}
                </Text>
              </section>
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  )
}

export default function ScenarioCatalog() {
  const styles = useScenarioCatalogStyles()
  const [scenarios, setScenarios] = useState<RegisteredScenario[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [refetchCount, setRefetchCount] = useState(0)
  const [expandedScenarios, setExpandedScenarios] = useState<Set<string>>(() => new Set())

  useEffect(() => {
    let cancelled = false

    fetchAllPages(
      (cursor) => scenariosApi.listCatalog(CATALOG_PAGE_SIZE, cursor),
      undefined,
      (scenario) => scenario.scenario_name,
    )
      .then((items) => {
        if (cancelled) return
        setScenarios(items)
        setError(null)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setScenarios([])
        setError(toApiError(err).detail)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [refetchCount])

  const handleRetry = useCallback(() => {
    setLoading(true)
    setError(null)
    setRefetchCount((count) => count + 1)
  }, [])

  const handleToggle = useCallback((scenarioName: string) => {
    setExpandedScenarios((current) => {
      const next = new Set(current)
      if (next.has(scenarioName)) {
        next.delete(scenarioName)
      } else {
        next.add(scenarioName)
      }
      return next
    })
  }, [])

  const filteredScenarios = useMemo(
    () => scenarios.filter((scenario) => matchesSearch(scenario, query)),
    [scenarios, query],
  )

  return (
    <section
      className={styles.root}
      data-testid="scenario-catalog"
      aria-labelledby="scenario-catalog-title"
    >
      <div className={styles.header}>
        <div className={styles.headerText}>
          <Text id="scenario-catalog-title" as="h1" size={600} weight="semibold">
            Scenarios
          </Text>
          <Text size={300} className={styles.subtitle}>
            Browse registered scenarios and launch a run against a configured target.
          </Text>
          <Text as="p" size={300} className={styles.explanation}>
            A scenario packages objective datasets, selected or aggregate techniques, baseline policy,
            and scenario-specific axes into a run plan.
          </Text>
        </div>
        <div className={styles.headerActions}>
          <Input
            className={styles.search}
            contentBefore={<SearchRegular />}
            placeholder="Search scenarios..."
            value={query}
            onChange={(_, data) => setQuery(data.value)}
            aria-label="Search scenarios"
          />
          <Button
            className={styles.touchTarget}
            appearance="subtle"
            icon={<ArrowSyncRegular />}
            onClick={handleRetry}
            disabled={loading}
          >
            Refresh
          </Button>
        </div>
      </div>

      {loading ? (
        <div className={styles.centeredState}>
          <Spinner label="Loading scenarios..." />
        </div>
      ) : error ? (
        <div className={styles.centeredState} data-testid="error-state">
          <MessageBar intent="error">
            <MessageBarBody>{error}</MessageBarBody>
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
      ) : scenarios.length === 0 ? (
        <div className={styles.centeredState} data-testid="empty-state">
          <Text size={400}>No scenarios are registered</Text>
          <Text size={200}>Register a scenario via your PyRIT initializers to see it here.</Text>
        </div>
      ) : filteredScenarios.length === 0 ? (
        <div className={styles.centeredState} data-testid="no-results-state">
          <Text size={400}>No scenarios match &quot;{query}&quot;</Text>
          <Text size={200}>Try a different search term.</Text>
        </div>
      ) : (
        <div className={styles.tableContainer}>
          <Table className={styles.table} size="small" aria-label="Registered scenarios">
            <TableHeader className={styles.tableHeader}>
              <TableRow>
                <TableHeaderCell
                  className={mergeClasses(
                    styles.scenarioColumn,
                    styles.tableHeaderCell,
                    'scenario-catalog-cell-padding',
                  )}
                >
                  Scenario / purpose
                </TableHeaderCell>
                <TableHeaderCell
                  className={mergeClasses(
                    styles.datasetColumn,
                    styles.tableHeaderCell,
                    'scenario-catalog-cell-padding',
                  )}
                >
                  Default dataset size
                </TableHeaderCell>
                <TableHeaderCell
                  className={mergeClasses(
                    styles.techniqueColumn,
                    styles.tableHeaderCell,
                    'scenario-catalog-cell-padding',
                  )}
                >
                  Default techniques
                </TableHeaderCell>
                <TableHeaderCell
                  className={mergeClasses(
                    styles.sizeColumn,
                    styles.tableHeaderCell,
                    'scenario-catalog-cell-padding',
                  )}
                >
                  Default run size
                </TableHeaderCell>
                <TableHeaderCell
                  className={mergeClasses(
                    styles.actionColumn,
                    styles.tableHeaderCell,
                    'scenario-catalog-cell-padding',
                  )}
                >
                  Action
                </TableHeaderCell>
              </TableRow>
            </TableHeader>
            <TableBody className={styles.tableBody}>
              {filteredScenarios.map((scenario) => (
                <ScenarioCatalogRow
                  key={scenario.scenario_name}
                  scenario={scenario}
                  expanded={expandedScenarios.has(scenario.scenario_name)}
                  onToggle={handleToggle}
                />
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </section>
  )
}
