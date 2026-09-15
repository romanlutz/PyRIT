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
  readonly viewKey: string
  readonly offset: number
  readonly allGroups: boolean
}

export default function AnalyticsPage({ onOpenAttack }: AnalyticsPageProps) {
  const styles = useAnalyticsPageStyles()
  const [searchParams, setSearchParams] = useSearchParams()
  const parsed = useMemo(() => analyticsViewFromSearchParams(searchParams), [searchParams])
  const view = parsed.view
  const viewKey = JSON.stringify(view)
  const [groupPage, setGroupPage] = useState<GroupPageState | null>(null)
  const [interactionError, setInteractionError] = useState<string | null>(null)
  const offset = groupPage?.viewKey === viewKey ? groupPage.offset : 0
  const allGroups = groupPage?.viewKey === viewKey && groupPage.allGroups
  const query = useMemo(
    () => parsed.error ? null : analyticsQuery(view, offset, allGroups),
    [parsed.error, view, offset, allGroups],
  )
  const analytics = useAttackAnalytics(query)
  const report = analytics.report

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
    setGroupPage({ viewKey, offset: 0, allGroups })
    analytics.reload()
  }

  function changeDimension(
    field: 'groupBy' | 'heatmapRow' | 'heatmapColumn',
    dimension: AttackAnalyticsDimension | null,
  ): void {
    if (dimension) changeView({ ...view, [field]: dimension })
  }

  return (
    <div className={styles.root}>
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
                      {hasAnalyticsFilters(view.filters) ? 'No results match these filters' : 'No saved AttackResults yet'}
                    </Text>
                    <Text>{hasAnalyticsFilters(view.filters)
                      ? 'Remove a filter or widen the last updated range to explore other saved results.'
                      : 'Analytics reads persisted results only. Save attack results, then use Reload to see them here.'}</Text>
                    {hasAnalyticsFilters(view.filters) && <Button className={styles.button}
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
                          <AnalyticsGroups report={report} successRate={view.chart === 'success-rate'} allGroups={allGroups} onDrilldown={drilldown} />
                          <div className={styles.row} aria-label="Group pagination">
                            {!allGroups && <Button className={styles.button} disabled={analytics.loading}
                              onClick={() => { setGroupPage({ viewKey, offset: 0, allGroups: true }) }}>Show all groups</Button>}
                            {allGroups && <Text size={200}>Browse all groups, 50 at a time.</Text>}
                            <Button className={styles.button} disabled={analytics.reportGroupOffset === 0 || analytics.loading}
                              onClick={() => {
                                setGroupPage({ viewKey, offset: 0, allGroups })
                                if (offset === 0 && analytics.error) analytics.reload()
                              }}>First groups</Button>
                            <Button className={styles.button} disabled={!report.has_more_groups || analytics.loading}
                              onClick={() => {
                                if (report.next_group_offset !== null) {
                                  setGroupPage({ viewKey, offset: report.next_group_offset, allGroups })
                                  if (offset === report.next_group_offset && analytics.error) analytics.reload()
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
