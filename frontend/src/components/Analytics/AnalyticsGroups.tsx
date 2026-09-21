import { useId } from 'react'

import { Button, mergeClasses, Text, Tooltip } from '@fluentui/react-components'

import type { AttackAnalyticsFilter, AttackAnalyticsGroup, AttackAnalyticsReport, AttackOutcome } from '@/types'
import {
  analyticsDimensionLabel, analyticsOptionLabel, analyticsStatisticsLabel, analyticsSuccessCountsLabel, analyticsValueKey,
  formatAnalyticsCount, formatAnalyticsPercent,
} from '@/utils/attackAnalytics'

import { AnalyticsAsrLabel } from './AnalyticsStats'
import { useAnalyticsGroupsStyles } from './AnalyticsGroups.styles'

interface AnalyticsGroupsProps {
  readonly report: AttackAnalyticsReport
  readonly successRate: boolean
  readonly onDrilldown: (filters: AttackAnalyticsFilter[], outcome?: AttackOutcome) => void
}

/**
 * Two presentations of the same server group page: outcome shares or ASR.
 * Widths use SDK ratios, never a sum of visible rows. Both bars and labeled
 * counts drill down with the supplied predicates; labels keep tiny segments usable.
 */
export default function AnalyticsGroups({ report, successRate, onDrilldown }: AnalyticsGroupsProps) {
  const styles = useAnalyticsGroupsStyles()
  const reasonId = useId()
  const drilldownUnavailable = report.drilldown_unavailable_reason !== null
  const drilldownDescription = drilldownUnavailable ? reasonId : undefined
  const marked = report.outcome_filter_applied
  const title = `${successRate ? `Success rate${marked ? '*' : ''}` : 'Outcome breakdown'} by ${analyticsDimensionLabel(report.group_by)}`

  return (
    <div className={styles.root}>
      <div className={styles.legend}>
        {successRate ? (
          <><AnalyticsAsrLabel marked={marked} /><Text>Fixed scale: 0% to 100%. Compare decided sample sizes, not rates alone.</Text></>
        ) : <Text>Bar width shows each outcome's share of all results in that group, not ASR.</Text>}
      </div>
      {drilldownUnavailable && <Text id={reasonId} role="note">
        {report.drilldown_unavailable_reason}
      </Text>}
      <section className={styles.groups} aria-label={title} aria-describedby={drilldownDescription} tabIndex={0}>
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
                  disabled={drilldownUnavailable || statistics.total_results === 0}
                  aria-label={`Filter to ${label}`}
                  aria-describedby={drilldownDescription}
                  onClick={() => { onDrilldown(group.drilldown_filters) }}>{label}</Button>
                <Text size={200}>{formatAnalyticsCount(statistics.total_results)} results</Text>
              </div>
              <div className={styles.visualization}>
                {successRate ? (
                  <>
                    <Button
                      appearance="transparent"
                      className={mergeClasses(styles.bar, styles.rateButton, statistics.success_rate === null && styles.unavailable)}
                      disabled={drilldownUnavailable || statistics.total_results === 0}
                      aria-label={`Inspect ${label} success rate: ${analyticsStatisticsLabel(statistics, marked)}`}
                      aria-describedby={drilldownDescription}
                      onClick={() => { onDrilldown(group.drilldown_filters) }}
                    >
                      {statistics.success_rate !== null && (
                        <span aria-hidden="true" className={mergeClasses(styles.segment, styles.rate)} style={{ width: `${statistics.success_rate * 100}%` }} />
                      )}
                    </Button>
                    <Text className={styles.number}>
                      {formatAnalyticsPercent(statistics.success_rate)}{marked ? '*' : ''}{' '}
                      ({analyticsSuccessCountsLabel(statistics)}, {formatAnalyticsCount(statistics.total_results)} total)
                    </Text>
                  </>
                ) : (
                  <div className={styles.bar} role="group" aria-label={`Outcome shares for ${label}`}>
                    {outcomes.map((item: { outcome: AttackOutcome; count: number }) => (
                      statistics.outcome_shares[item.outcome] > 0 && (
                        <Tooltip key={item.outcome} relationship="description"
                          content={report.drilldown_unavailable_reason ??
                            `${formatAnalyticsPercent(statistics.outcome_shares[item.outcome])} of this group's results. The labeled outcome counts below provide the same action.`}>
                          <Button
                            appearance="transparent"
                            className={mergeClasses(styles.segmentButton, styles[item.outcome])}
                            style={{ width: `${statistics.outcome_shares[item.outcome] * 100}%` }}
                            disabled={drilldownUnavailable}
                            aria-label={`${label}: ${item.outcome} segment; ${formatAnalyticsCount(item.count)} results; filter dashboard`}
                            aria-describedby={drilldownDescription}
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
                      content={report.drilldown_unavailable_reason ??
                        `${formatAnalyticsPercent(statistics.outcome_shares[item.outcome])} of this group's results. Filter the entire dashboard.`}>
                      <Button className={styles.button} appearance="subtle" size="small"
                        disabled={drilldownUnavailable || item.count === 0}
                        aria-label={`${label}: ${formatAnalyticsCount(item.count)} ${item.outcome}; filter dashboard`}
                        aria-describedby={drilldownDescription}
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
    </div>
  )
}
