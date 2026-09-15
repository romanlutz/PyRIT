import { useState } from 'react'

import {
  Button, mergeClasses, Table, TableBody, TableCell, TableHeader,
  TableHeaderCell, TableRow, Text, Tooltip,
} from '@fluentui/react-components'

import type { AttackAnalyticsFilter, AttackAnalyticsGroup, AttackAnalyticsReport, AttackOutcome } from '@/types'
import {
  analyticsDimensionLabel, analyticsOptionLabel, analyticsStatisticsLabel, analyticsValueKey,
  formatAnalyticsCount, formatAnalyticsPercent,
} from '@/utils/attackAnalytics'

import { AnalyticsAsrLabel } from './AnalyticsStats'
import { useAnalyticsGroupsStyles } from './AnalyticsGroups.styles'

interface AnalyticsGroupsProps {
  readonly report: AttackAnalyticsReport
  readonly successRate: boolean
  readonly allGroups: boolean
  readonly onDrilldown: (filters: AttackAnalyticsFilter[], outcome?: AttackOutcome) => void
}

export default function AnalyticsGroups({ report, successRate, allGroups, onDrilldown }: AnalyticsGroupsProps) {
  const styles = useAnalyticsGroupsStyles()
  const [showData, setShowData] = useState(false)
  const marked = report.outcome_filter_applied
  const title = `${successRate ? `Success rate${marked ? '*' : ''}` : 'Outcome breakdown'} by ${analyticsDimensionLabel(report.group_by)}`

  return (
    <div className={styles.root}>
      <div className={styles.legend}>
        {successRate ? (
          <><AnalyticsAsrLabel marked={marked} /><Text>Fixed scale: 0% to 100%. Compare decided sample sizes, not rates alone.</Text></>
        ) : <Text>Bar width shows each outcome's share of all results in that group, not ASR.</Text>}
      </div>
      <section className={styles.groups} aria-label={title} tabIndex={0}>
        {report.groups.map((group: AttackAnalyticsGroup) => {
          const label = analyticsOptionLabel(group)
          const statistics = group.statistics
          const outcomes: Array<{ outcome: AttackOutcome; count: number }> = [
            { outcome: 'success', count: statistics.successes },
            { outcome: 'failure', count: statistics.failures },
            { outcome: 'error', count: statistics.errors },
            { outcome: 'undetermined', count: statistics.undetermined },
          ]
          return (
            <div className={styles.group} key={analyticsValueKey(group.key)}>
              <div className={styles.groupLabel}>
                <Button className={styles.groupButton} appearance="subtle"
                  disabled={statistics.total_results === 0}
                  aria-label={`Filter to ${label}`}
                  onClick={() => { onDrilldown(group.drilldown_filters) }}>{label}</Button>
                <Text size={200}>{formatAnalyticsCount(statistics.total_results)} results</Text>
              </div>
              <div className={styles.visualization}>
                {successRate ? (
                  <>
                    <Button
                      appearance="transparent"
                      className={mergeClasses(styles.bar, styles.rateButton, statistics.success_rate === null && styles.unavailable)}
                      disabled={statistics.total_results === 0}
                      aria-label={`Inspect ${label} success rate: ${analyticsStatisticsLabel(statistics, marked)}`}
                      onClick={() => { onDrilldown(group.drilldown_filters) }}
                    >
                      {statistics.success_rate !== null && (
                        <span aria-hidden="true" className={mergeClasses(styles.segment, styles.rate)} style={{ width: `${statistics.success_rate * 100}%` }} />
                      )}
                    </Button>
                    <Text className={styles.number}>
                      {formatAnalyticsPercent(statistics.success_rate)}{marked ? '*' : ''}{' '}
                      ({formatAnalyticsCount(statistics.successes)} / {formatAnalyticsCount(statistics.total_decided)} decided)
                    </Text>
                  </>
                ) : (
                  <div className={styles.bar} role="group" aria-label={`Outcome shares for ${label}`}>
                    {outcomes.map((item: { outcome: AttackOutcome; count: number }) => (
                      statistics.outcome_shares[item.outcome] > 0 && (
                        <Tooltip key={item.outcome} relationship="description"
                          content={`${formatAnalyticsPercent(statistics.outcome_shares[item.outcome])} of this group's results. The labeled outcome counts below provide the same action.`}>
                          <Button
                            appearance="transparent"
                            className={mergeClasses(styles.segmentButton, styles[item.outcome])}
                            style={{ width: `${statistics.outcome_shares[item.outcome] * 100}%` }}
                            aria-label={`${label}: ${item.outcome} segment; ${formatAnalyticsCount(item.count)} results; filter dashboard`}
                            onClick={() => { onDrilldown(group.drilldown_filters, item.outcome) }}
                          />
                        </Tooltip>
                      )
                    ))}
                  </div>
                )}
                <div className={styles.outcomes}>
                  {outcomes.map((item: { outcome: AttackOutcome; count: number }) => (
                    <Tooltip key={item.outcome} relationship="description"
                      content={`${formatAnalyticsPercent(statistics.outcome_shares[item.outcome])} of this group's results. Filter the entire dashboard.`}>
                      <Button className={styles.button} appearance="subtle" size="small"
                        disabled={item.count === 0}
                        aria-label={`${label}: ${formatAnalyticsCount(item.count)} ${item.outcome}; filter dashboard`}
                        onClick={() => { onDrilldown(group.drilldown_filters, item.outcome) }}>
                        <span aria-hidden="true" className={mergeClasses(styles.swatch, styles[item.outcome])} />
                        {item.outcome} {formatAnalyticsCount(item.count)}
                      </Button>
                    </Tooltip>
                  ))}
                </div>
              </div>
            </div>
          )
        })}
        {report.groups.length === 0 && <Text>No groups on this page. Return to the first groups or narrow the filters.</Text>}
      </section>
      {!allGroups && (
        <div>
          <Button className={styles.button} aria-expanded={showData}
            onClick={() => { setShowData(!showData) }}>{showData ? 'Hide aggregate data' : 'Show aggregate data'}</Button>
        </div>
      )}
      {(showData || allGroups) && (
        <div className={styles.scroll} role="region" aria-label="Grouped aggregate data" tabIndex={0}>
          <Table className={styles.table} size="small">
            <caption className={styles.caption}>Aggregate data by {analyticsDimensionLabel(report.group_by)}</caption>
            <TableHeader><TableRow>
              <TableHeaderCell>Group</TableHeaderCell>
              <TableHeaderCell>Total results</TableHeaderCell>
              <TableHeaderCell><AnalyticsAsrLabel marked={marked} /></TableHeaderCell>
              <TableHeaderCell>Decided</TableHeaderCell>
              <TableHeaderCell>Decided share</TableHeaderCell>
              <TableHeaderCell>Successes</TableHeaderCell>
              <TableHeaderCell>Failures</TableHeaderCell>
              <TableHeaderCell>Errors</TableHeaderCell>
              <TableHeaderCell>Undetermined</TableHeaderCell>
            </TableRow></TableHeader>
            <TableBody>
              {report.groups.map((group: AttackAnalyticsGroup) => (
                <TableRow key={analyticsValueKey(group.key)}>
                  <TableCell>
                    <Button appearance="subtle" className={styles.groupButton}
                      aria-label={`Inspect ${analyticsOptionLabel(group)}: ${analyticsStatisticsLabel(group.statistics, marked)}`}
                      onClick={() => { onDrilldown(group.drilldown_filters) }}>{analyticsOptionLabel(group)}</Button>
                  </TableCell>
                  <TableCell>{formatAnalyticsCount(group.statistics.total_results)}</TableCell>
                  <TableCell>{formatAnalyticsPercent(group.statistics.success_rate)}{marked ? '*' : ''}</TableCell>
                  <TableCell>{formatAnalyticsCount(group.statistics.total_decided)}</TableCell>
                  <TableCell>{formatAnalyticsPercent(group.statistics.decided_share)}</TableCell>
                  <TableCell>{formatAnalyticsCount(group.statistics.successes)}</TableCell>
                  <TableCell>{formatAnalyticsCount(group.statistics.failures)}</TableCell>
                  <TableCell>{formatAnalyticsCount(group.statistics.errors)}</TableCell>
                  <TableCell>{formatAnalyticsCount(group.statistics.undetermined)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}
