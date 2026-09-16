import { Button, Text, Tooltip } from '@fluentui/react-components'
import { InfoRegular } from '@fluentui/react-icons'

import OutcomeBadge from '@/components/OutcomeBadge'
import type { AttackAnalyticsStatistics, AttackOutcome } from '@/types'
import {
  ANALYTICS_ASR_DEFINITION, ANALYTICS_ASR_NOTE, analyticsSuccessCountsLabel,
  formatAnalyticsCount, formatAnalyticsPercent,
} from '@/utils/attackAnalytics'

import { useAnalyticsStatsStyles } from './AnalyticsStats.styles'

interface AnalyticsAsrLabelProps {
  readonly marked: boolean
}

export function AnalyticsAsrLabel({ marked }: AnalyticsAsrLabelProps) {
  const styles = useAnalyticsStatsStyles()
  const explanation = marked ? ANALYTICS_ASR_NOTE : ANALYTICS_ASR_DEFINITION
  return (
    <Tooltip content={explanation} relationship="description">
      <Button
        appearance="transparent"
        size="small"
        className={styles.asrLabel}
        icon={<InfoRegular />}
        iconPosition="after"
        aria-label={`ASR${marked ? '*' : ''}: ${explanation}`}
      >
        ASR{marked ? '*' : ''}
      </Button>
    </Tooltip>
  )
}

interface AnalyticsStatsProps {
  readonly statistics: AttackAnalyticsStatistics
  readonly marked: boolean
  readonly onOutcome: (outcome: AttackOutcome) => void
}

export default function AnalyticsStats({ statistics, marked, onOutcome }: AnalyticsStatsProps) {
  const styles = useAnalyticsStatsStyles()
  const outcomes: Array<{ outcome: AttackOutcome; count: number }> = [
    { outcome: 'success', count: statistics.successes },
    { outcome: 'failure', count: statistics.failures },
    { outcome: 'error', count: statistics.errors },
    { outcome: 'undetermined', count: statistics.undetermined },
  ]
  return (
    <section aria-label="Outcome summary" className={styles.root}>
      <div className={styles.metric}>
        <Text>Total results</Text>
        <Text size={600} weight="semibold" className={styles.number}>{formatAnalyticsCount(statistics.total_results)}</Text>
      </div>
      <div className={styles.metric}>
        <AnalyticsAsrLabel marked={marked} />
        <Text size={500} weight="semibold" className={styles.number}>
          {formatAnalyticsPercent(statistics.success_rate)}{marked ? '*' : ''}
        </Text>
        <Text size={200}>{analyticsSuccessCountsLabel(statistics)}</Text>
        <Text size={200}>{formatAnalyticsCount(statistics.total_results)} total</Text>
      </div>
      <div className={styles.metric}>
        <Text>Decided share</Text>
        <Text size={500} className={styles.number}>{formatAnalyticsPercent(statistics.decided_share)}</Text>
        <Text size={200}>{formatAnalyticsCount(statistics.total_decided)} / {formatAnalyticsCount(statistics.total_results)} results</Text>
      </div>
      <div className={styles.outcomes} aria-label="Filter entire dashboard by outcome">
        {outcomes.map((item: { outcome: AttackOutcome; count: number }) => (
          <Button
            key={item.outcome}
            className={styles.button}
            appearance="subtle"
            aria-label={`Filter to ${item.outcome}: ${formatAnalyticsCount(item.count)}`}
            onClick={() => { onOutcome(item.outcome) }}
          >
            <OutcomeBadge outcome={item.outcome} appearance="filled" /> {formatAnalyticsCount(item.count)}
          </Button>
        ))}
      </div>
    </section>
  )
}
