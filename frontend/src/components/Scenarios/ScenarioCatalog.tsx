import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  Badge,
  Button,
  Input,
  MessageBar,
  MessageBarBody,
  Spinner,
  Text,
} from '@fluentui/react-components'
import { ArrowSyncRegular, SearchRegular } from '@fluentui/react-icons'
import { Link } from 'react-router-dom'

import { scenariosApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type { RegisteredScenario } from '@/types'
import { fetchAllPages } from '@/utils/fetchAllPages'

import { useScenarioCatalogStyles } from './ScenarioCatalog.styles'

/** Items requested per catalog page while paging through the full list. */
const CATALOG_PAGE_SIZE = 200
/** Concrete techniques (beyond the default) shown on a card before collapsing to "+N more". */
const OTHER_TECHNIQUE_DISPLAY_LIMIT = 4

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

export default function ScenarioCatalog() {
  const styles = useScenarioCatalogStyles()
  const [scenarios, setScenarios] = useState<RegisteredScenario[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [refetchCount, setRefetchCount] = useState(0)

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

  const filteredScenarios = useMemo(
    () => scenarios.filter((scenario) => matchesSearch(scenario, query)),
    [scenarios, query],
  )

  return (
    <div className={styles.root} data-testid="scenario-catalog">
      <div className={styles.header}>
        <div className={styles.headerText}>
          <Text as="h1" size={600} weight="semibold">Scenarios</Text>
          <Text size={300} className={styles.subtitle}>
            Browse registered scenarios and launch a run against a configured target.
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
        <div className={styles.grid}>
          {filteredScenarios.map((scenario) => {
            const aggregateTechniques = scenario.aggregate_techniques.filter(
              (technique) => technique !== scenario.default_technique,
            )
            const concreteTechniques = scenario.all_techniques.filter(
              (technique) => technique !== scenario.default_technique,
            )
            const shownTechniques = concreteTechniques.slice(0, OTHER_TECHNIQUE_DISPLAY_LIMIT)
            const remainingCount = concreteTechniques.length - shownTechniques.length

            return (
              <Link
                key={scenario.scenario_name}
                to={`/scenarios/${encodeURIComponent(scenario.scenario_name)}`}
                className={styles.card}
                data-testid={`scenario-card-${scenario.scenario_name}`}
              >
                <Text weight="semibold" size={400}>{scenario.scenario_name}</Text>
                <Text size={200} className={styles.description}>{scenario.description}</Text>
                <div className={styles.metadataGroup}>
                  <Text size={200} weight="semibold">Default technique</Text>
                  <div className={styles.badgeGroup}>
                    <Badge appearance="tint" color="brand">{scenario.default_technique}</Badge>
                  </div>
                </div>
                {aggregateTechniques.length > 0 && (
                  <div className={styles.metadataGroup}>
                    <Text size={200} weight="semibold">Aggregate techniques</Text>
                    <div className={styles.badgeGroup}>
                      {aggregateTechniques.map((technique) => (
                        <Badge key={technique} appearance="tint">{technique}</Badge>
                      ))}
                    </div>
                  </div>
                )}
                <div className={styles.metadataGroup}>
                  <Text size={200} weight="semibold">Available techniques</Text>
                  <div className={styles.badgeGroup}>
                    {shownTechniques.map((technique) => (
                      <Badge key={technique} appearance="outline">{technique}</Badge>
                    ))}
                    {remainingCount > 0 && (
                      <Badge appearance="ghost">+{remainingCount} more</Badge>
                    )}
                  </div>
                </div>
                {scenario.default_datasets.length > 0 && (
                  <Text size={200} className={styles.datasets}>
                    Datasets: {scenario.default_datasets.join(', ')}
                  </Text>
                )}
              </Link>
            )
          })}
        </div>
      )}
    </div>
  )
}
