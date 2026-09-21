import { useMemo, useState } from 'react'

import {
  Button, Field, MessageBar, MessageBarBody, Select, Skeleton, SkeletonItem, Text,
} from '@fluentui/react-components'
import type { SelectOnChangeData } from '@fluentui/react-components'
import { ArrowSyncRegular } from '@fluentui/react-icons'
import { useSearchParams } from 'react-router'

import { ErrorBoundary } from '@/components/ErrorBoundary'
import { useAttackAnalytics } from '@/hooks/useAttackAnalytics'
import type { AttackAnalyticsDimension, AttackAnalyticsFilter, AttackAnalyticsFilters, AttackAnalyticsViewState, AttackOutcome } from '@/types'
import {
  ANALYTICS_ASR_DEFINITION, ANALYTICS_ASR_NOTE, DEFAULT_ANALYTICS_VIEW,
  analyticsDimensionKey, analyticsQuery, analyticsViewError, analyticsViewFromSearchParams,
  analyticsViewToSearchParams, appendAnalyticsDrilldown, formatAnalyticsTime, hasAnalyticsFilters,
} from '@/utils/attackAnalytics'

import AnalyticsDimensionPicker from './AnalyticsDimensionPicker'
import AnalyticsFilters from './AnalyticsFilters'
import AnalyticsGroups from './AnalyticsGroups'
import AnalyticsHeatmap from './AnalyticsHeatmap'
import AnalyticsResultsTable from './AnalyticsResultsTable'
import AnalyticsStats from './AnalyticsStats'
import { useAnalyticsPageStyles } from './AnalyticsPage.styles'

interface AnalyticsPageProps {
  readonly onOpenAttack: (attackResultId: string) => void
}

interface GroupPageState {
  readonly queryKey: string
  readonly offset: number
}

/**
 * Coordinates URL-owned filters/presentation and local pagination. SDK reports
 * supply every aggregate; child charts only render them or return drill-down
 * predicates. Filter drafts and cursors are deliberately not shareable URL state.
 */
export default function AnalyticsPage({ onOpenAttack }: AnalyticsPageProps) {
  const styles = useAnalyticsPageStyles()
  const [searchParams, setSearchParams] = useSearchParams()
  const parsed = useMemo(() => analyticsViewFromSearchParams(searchParams), [searchParams])
  const view = parsed.view
  // Presentation-only controls must not reset the group page. The first-page
  // server query identifies this scope independently of chart colors/rendering.
  const groupQueryKey = JSON.stringify(analyticsQuery(view))
  const [groupPage, setGroupPage] = useState<GroupPageState | null>(null)
  const [interactionError, setInteractionError] = useState<string | null>(null)
  const offset = groupPage?.queryKey === groupQueryKey ? groupPage.offset : 0
  const query = parsed.error ? null : analyticsQuery(view, offset)
  const analytics = useAttackAnalytics(query)
  const report = analytics.report
  const filtersActive = hasAnalyticsFilters(view.filters)

  /** Validate before writing browser history so rejected edits keep the current cohort intact. */
  function changeView(next: AttackAnalyticsViewState): boolean {
    const error = analyticsViewError(next)
    setInteractionError(error)
    if (error) return false
    setSearchParams(analyticsViewToSearchParams(next))
    return true
  }

  function changeFilters(filters: AttackAnalyticsFilters): boolean {
    return changeView({ ...view, filters })
  }

  function drilldown(predicates: AttackAnalyticsFilter[], outcome?: AttackOutcome): void {
    changeFilters(appendAnalyticsDrilldown(view.filters, predicates, outcome))
  }

  function reload(): void {
    setGroupPage({ queryKey: groupQueryKey, offset: 0 })
    analytics.reload()
  }

  function changeGroupPage(nextOffset: number): void {
    setGroupPage({ queryKey: groupQueryKey, offset: nextOffset })
    // After a failure, the requested offset can differ from the still-visible
    // report's offset. Selecting that same request again must actually retry it.
    if (offset === nextOffset && analytics.error) analytics.reload()
  }

  function changeDimension(
    field: 'groupBy' | 'heatmapRow' | 'heatmapColumn',
    dimension: AttackAnalyticsDimension | null,
  ): void {
    if (dimension) changeView({ ...view, [field]: dimension })
  }

  return (
    <div className={styles.root} data-testid="analytics-page">
      <header className={styles.header}>
        <Text as="h1" className={styles.title} size={600} weight="semibold">Analytics</Text>
        <div className={styles.row}>
          <Text size={200}>
            Last refreshed: {report
              ? <time dateTime={report.computed_at}>{formatAnalyticsTime(report.computed_at)}</time>
              : 'Not loaded'}
          </Text>
          <Button className={styles.button} appearance="subtle" icon={<ArrowSyncRegular />}
            disabled={analytics.loading || parsed.error !== null}
            onClick={reload}>Reload</Button>
        </div>
      </header>
      <div className={styles.body}>
        {parsed.error ? (
          <MessageBar intent="error"><MessageBarBody>
            {parsed.error}{' '}
            <Button className={styles.button} onClick={() => { changeView(DEFAULT_ANALYTICS_VIEW) }}>Reset analytics view</Button>
          </MessageBarBody></MessageBar>
        ) : (
          <>
            <Text className={styles.note} size={200}>
              All saved AttackResults, across operations and targets. Filters apply to the entire dashboard.
            </Text>
            <AnalyticsFilters filters={view.filters} refreshVersion={analytics.refreshVersion} onChange={changeFilters} />
            {interactionError && <MessageBar intent="error"><MessageBarBody>{interactionError}</MessageBarBody></MessageBar>}
            <div className={styles.toolbar} aria-label="Chart controls">
              <Field label="Chart view">
                <Select className={styles.button} value={view.chart}
                  onChange={(_event: React.ChangeEvent<HTMLSelectElement>, data: SelectOnChangeData) => {
                    if (data.value === 'outcomes' || data.value === 'success-rate' || data.value === 'heatmap') {
                      changeView({ ...view, chart: data.value })
                    }
                  }}>
                  <option value="outcomes">Outcome breakdown</option>
                  <option value="success-rate">Success rate</option>
                  <option value="heatmap">Heatmap</option>
                </Select>
              </Field>
              {view.chart === 'heatmap' ? (
                <>
                  <AnalyticsDimensionPicker key={`row:${analyticsDimensionKey(view.heatmapRow)}`}
                    label="Rows" dimension={view.heatmapRow}
                    onChange={(dimension: AttackAnalyticsDimension | null) => { changeDimension('heatmapRow', dimension) }} />
                  <AnalyticsDimensionPicker key={`column:${analyticsDimensionKey(view.heatmapColumn)}`}
                    label="Compare by" dimension={view.heatmapColumn}
                    onChange={(dimension: AttackAnalyticsDimension | null) => { changeDimension('heatmapColumn', dimension) }} />
                  <Field label="Cell color">
                    <Select className={styles.button} value={view.heatmapMetric}
                      onChange={(_event: React.ChangeEvent<HTMLSelectElement>, data: SelectOnChangeData) => {
                        if (data.value === 'success_rate' || data.value === 'total_results') {
                          changeView({ ...view, heatmapMetric: data.value })
                        }
                      }}>
                      <option value="success_rate">Attack success rate</option>
                      <option value="total_results">Result count</option>
                    </Select>
                  </Field>
                </>
              ) : (
                <AnalyticsDimensionPicker key={analyticsDimensionKey(view.groupBy)} label="Group by" dimension={view.groupBy}
                  onChange={(dimension: AttackAnalyticsDimension | null) => { changeDimension('groupBy', dimension) }} />
              )}
            </div>
            {analytics.error && (
              <MessageBar intent="error"><MessageBarBody>
                {report ? 'Reload failed. Showing the last successful report; data may be stale.' : 'Could not load analytics for these filters.'}
                {' '}{analytics.error}{' '}
                <Button className={styles.button} onClick={reload}>Retry</Button>
              </MessageBarBody></MessageBar>
            )}
            {analytics.loading && (
              <div role="status" aria-label={report ? 'Reloading saved results' : 'Loading analytics for current filters'}>
                <Text>{report ? 'Reloading saved results...' : 'Loading analytics for current filters...'}</Text>
                {!report && <Skeleton className={styles.skeleton} aria-label="Loading report">
                  <SkeletonItem size={48} /><SkeletonItem size={128} /><SkeletonItem size={96} />
                </Skeleton>}
              </div>
            )}
            {report && (
              <>
                {report.warnings.map((warning: string) => (
                  <MessageBar key={warning} intent="warning"><MessageBarBody>{warning}</MessageBarBody></MessageBar>
                ))}
                <AnalyticsStats statistics={report.summary} marked={report.outcome_filter_applied}
                  onOutcome={(outcome: AttackOutcome) => { changeFilters({ ...view.filters, outcomes: [outcome] }) }} />
                <Text role="note" aria-label={report.outcome_filter_applied ? 'Outcome-filtered ASR' : 'ASR definition'}
                  size={200} className={styles.note}>{report.outcome_filter_applied ? ANALYTICS_ASR_NOTE : ANALYTICS_ASR_DEFINITION}</Text>
                {report.summary.total_results === 0 ? (
                  <section className={styles.empty} aria-label="No analytics results">
                    <Text as="h2" size={500} weight="semibold" className={styles.title}>
                      {filtersActive ? 'No results match these filters' : 'No saved AttackResults yet'}
                    </Text>
                    <Text>{filtersActive
                      ? 'Remove a filter or widen the last updated range to explore other saved results.'
                      : 'Analytics reads persisted results only. Save attack results, then use Reload to see them here.'}</Text>
                    {filtersActive && <Button className={styles.button}
                      onClick={() => { changeFilters({ dimensions: [], outcomes: [] }) }}>Clear filters</Button>}
                  </section>
                ) : (
                  <ErrorBoundary>
                    <section className={styles.section} aria-label="Aggregate visualization">
                      {report.groups_overlap && <Text size={200} className={styles.note}>
                        Groups overlap: a result can appear in multiple groups or cells. Do not add their totals together.
                      </Text>}
                      {view.chart === 'heatmap'
                        ? <AnalyticsHeatmap report={report} metric={view.heatmapMetric} onDrilldown={drilldown} />
                        : <>
                          <AnalyticsGroups report={report} successRate={view.chart === 'success-rate'} onDrilldown={drilldown} />
                          <div className={styles.row} aria-label="Group pagination">
                            <Button className={styles.button} disabled={analytics.reportGroupOffset === 0 || analytics.loading}
                              onClick={() => { changeGroupPage(0) }}>First groups</Button>
                            <Button className={styles.button} disabled={!report.has_more_groups || analytics.loading}
                              onClick={() => {
                                if (report.next_group_offset !== null) {
                                  changeGroupPage(report.next_group_offset)
                                }
                              }}>Next groups</Button>
                            {report.has_more_groups && <Text size={200}>More groups are available. No combined &quot;Other&quot; bucket is used.</Text>}
                          </div>
                        </>}
                    </section>
                  </ErrorBoundary>
                )}
                {analytics.results && <AnalyticsResultsTable results={analytics.results} totalResults={report.summary.total_results}
                  page={analytics.page} loading={analytics.resultsLoading} disabled={analytics.loading}
                  error={analytics.resultsError} onFirst={analytics.firstPage} onNext={analytics.nextPage}
                  onRetry={analytics.retryResults} onOpenAttack={onOpenAttack} />}
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}
