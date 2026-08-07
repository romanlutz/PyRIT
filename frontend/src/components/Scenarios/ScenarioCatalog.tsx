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
import type { RegisteredScenario, ScenarioRunEstimateState } from '@/types'
import { fetchAllPages } from '@/utils/fetchAllPages'

import { useScenarioCatalogStyles } from './ScenarioCatalog.styles'
import {
  ScenarioRunEstimateDetails,
  ScenarioRunEstimateSummary,
} from './ScenarioRunEstimate'
import { normalizeScenarioMarkdown } from './scenarioMarkdown'

/** Items requested per catalog page while paging through the full list. */
const CATALOG_PAGE_SIZE = 200

const DEFAULT_ESTIMATE_STATE: ScenarioRunEstimateState = {
  status: 'unavailable',
  scope: 'default',
  label: 'The backend has not supplied a default-run estimate.',
  caveat: 'The authoritative total, ordered formula, and dataset counts will appear after sizing is configured.',
}

function matchesSearch(scenario: RegisteredScenario, query: string): boolean {
  if (!query) {
    return true
  }
  const haystack = [
    scenario.scenario_name,
    scenario.description,
    scenario.scenario_type,
    scenario.default_technique,
    ...scenario.aggregate_techniques,
    ...scenario.all_techniques,
    ...scenario.default_datasets,
  ]
    .join(' ')
    .toLowerCase()
  return haystack.includes(query.toLowerCase())
}

function uniqueNames(names: string[]): string[] {
  return [...new Set(names)]
}

interface ScenarioCatalogRowProps {
  scenario: RegisteredScenario
  expanded: boolean
  onToggle: (scenarioName: string) => void
}

function ScenarioCatalogRow({ scenario, expanded, onToggle }: ScenarioCatalogRowProps) {
  const styles = useScenarioCatalogStyles()
  const detailsId = `scenario-details-${encodeURIComponent(scenario.scenario_name).replace(/%/g, '-')}`
  const aggregateNames = new Set(scenario.aggregate_techniques)
  const defaultIsAggregate = aggregateNames.has(scenario.default_technique)
  const aggregateTechniques = uniqueNames(scenario.aggregate_techniques).filter(
    (technique) => !defaultIsAggregate || technique !== scenario.default_technique,
  )
  const concreteCandidates = defaultIsAggregate
    ? scenario.all_techniques
    : [scenario.default_technique, ...scenario.all_techniques]
  const concreteTechniques = uniqueNames(concreteCandidates).filter(
    (technique) => !aggregateNames.has(technique),
  )
  const description = normalizeScenarioMarkdown(scenario.description)
  const scenarioPath = `/scenarios/${encodeURIComponent(scenario.scenario_name)}`

  return (
    <>
      <TableRow
        className={mergeClasses(styles.summaryRow, expanded && styles.expandedSummaryRow)}
        data-testid={`scenario-card-${scenario.scenario_name}`}
      >
        <TableCell className={styles.tableCell}>
          <Text className={styles.mobileLabel} size={200} weight="semibold">
            Scenario / purpose
          </Text>
          <div className={styles.scenarioSummary}>
            <Link to={scenarioPath} className={styles.scenarioLink}>
              {scenario.scenario_name}
            </Link>
            <Text size={200} className={styles.scenarioType}>{scenario.scenario_type}</Text>
            <Text size={200} className={styles.purposePreview}>{scenario.description}</Text>
          </div>
        </TableCell>
        <TableCell className={styles.tableCell}>
          <Text className={styles.mobileLabel} size={200} weight="semibold">
            Default run size
          </Text>
          <ScenarioRunEstimateSummary state={DEFAULT_ESTIMATE_STATE} />
        </TableCell>
        <TableCell className={styles.tableCell}>
          <Text className={styles.mobileLabel} size={200} weight="semibold">
            Techniques
          </Text>
          <div className={styles.compactStack}>
            <Badge appearance="tint" color="brand">{scenario.default_technique}</Badge>
            <Text size={200} className={styles.secondaryText}>
              {aggregateTechniques.length} additional preset{aggregateTechniques.length === 1 ? '' : 's'}
              {' · '}
              {concreteTechniques.length} concrete
            </Text>
          </div>
        </TableCell>
        <TableCell className={styles.tableCell}>
          <Text className={styles.mobileLabel} size={200} weight="semibold">
            Datasets
          </Text>
          <Text>
            {scenario.default_datasets.length === 0
              ? 'No default datasets'
              : `${scenario.default_datasets.length} default dataset${scenario.default_datasets.length === 1 ? '' : 's'}`}
          </Text>
        </TableCell>
        <TableCell className={mergeClasses(styles.tableCell, styles.actionCell)}>
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
                <ScenarioRunEstimateDetails state={DEFAULT_ESTIMATE_STATE} />
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
                  <Text weight="semibold">Other aggregate presets</Text>
                  {aggregateTechniques.length > 0 ? (
                    <div className={styles.badgeGroup}>
                      {aggregateTechniques.map((technique) => (
                        <Badge key={technique} appearance="tint">{technique}</Badge>
                      ))}
                    </div>
                  ) : (
                    <Text size={200} className={styles.secondaryText}>None registered.</Text>
                  )}
                </div>
                <div className={styles.metadataGroup}>
                  <Text weight="semibold">Concrete techniques</Text>
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
                <Text as="h3" size={400} weight="semibold">Default datasets</Text>
                {scenario.default_datasets.length > 0 ? (
                  <ul className={styles.datasetList}>
                    {uniqueNames(scenario.default_datasets).map((dataset) => (
                      <li className={styles.datasetItem} key={dataset}>
                        <Text>{dataset}</Text>
                        <Badge appearance="outline">Count unavailable</Badge>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <Text size={200} className={styles.secondaryText}>
                    This scenario does not declare a default dataset.
                  </Text>
                )}
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
            and scenario-specific axes into a backend run plan.
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
                <TableHeaderCell className={styles.scenarioColumn}>Scenario / purpose</TableHeaderCell>
                <TableHeaderCell className={styles.sizeColumn}>Default run size</TableHeaderCell>
                <TableHeaderCell className={styles.techniqueColumn}>Techniques</TableHeaderCell>
                <TableHeaderCell className={styles.datasetColumn}>Datasets</TableHeaderCell>
                <TableHeaderCell className={styles.actionColumn}>Action</TableHeaderCell>
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
