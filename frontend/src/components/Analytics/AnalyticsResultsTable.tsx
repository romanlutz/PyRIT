import {
  Button, MessageBar, MessageBarBody, Spinner, Table, TableBody,
  TableCell, TableHeader, TableHeaderCell, TableRow, Text,
} from '@fluentui/react-components'
import { OpenRegular } from '@fluentui/react-icons'

import HistoryPagination from '@/components/History/HistoryPagination'
import OutcomeBadge from '@/components/OutcomeBadge'
import type { AttackAnalyticsResultRow, AttackAnalyticsResults } from '@/types'
import { formatAnalyticsCount, formatAnalyticsTime } from '@/utils/attackAnalytics'

import { useAnalyticsResultsTableStyles } from './AnalyticsResultsTable.styles'

interface AnalyticsResultsTableProps {
  readonly results: AttackAnalyticsResults
  readonly totalResults: number
  readonly page: number
  readonly loading: boolean
  readonly disabled: boolean
  readonly error: string | null
  readonly onFirst: () => void
  readonly onNext: () => void
  readonly onRetry: () => void
  readonly onOpenAttack: (attackResultId: string) => void
}

export default function AnalyticsResultsTable({
  results, totalResults, page, loading, disabled, error, onFirst, onNext, onRetry, onOpenAttack,
}: AnalyticsResultsTableProps) {
  const styles = useAnalyticsResultsTableStyles()
  return (
    <section className={styles.root} aria-label="Matching AttackResults" aria-busy={loading}>
      <div className={styles.header}>
        <Text as="h2" size={500} weight="semibold" className={styles.heading}>Matching AttackResults</Text>
        <Text>{formatAnalyticsCount(totalResults)} results in the report</Text>
      </div>
      {error && (
        <MessageBar intent="error"><MessageBarBody>
          Could not load the requested results page. Still showing page {page + 1}. {error}{' '}
          <Button className={styles.button} onClick={onRetry}>Retry results</Button>
        </MessageBarBody></MessageBar>
      )}
      {loading && <Spinner size="tiny" label="Loading results page" />}
      <div className={styles.scroll} role="region" aria-label="Result rows" tabIndex={0}>
        <Table className={styles.table} size="small" aria-label="Saved attack results">
          <TableHeader><TableRow>
            <TableHeaderCell>Outcome</TableHeaderCell><TableHeaderCell>Objective</TableHeaderCell>
            <TableHeaderCell>Operation</TableHeaderCell><TableHeaderCell>Operator</TableHeaderCell>
            <TableHeaderCell>Attack type / targeted harms</TableHeaderCell><TableHeaderCell>Target model / identity</TableHeaderCell>
            <TableHeaderCell>Last updated</TableHeaderCell><TableHeaderCell>Open</TableHeaderCell>
          </TableRow></TableHeader>
          <TableBody>
            {results.items.map((result: AttackAnalyticsResultRow) => (
              <TableRow key={result.attack_result_id}>
                <TableCell><OutcomeBadge outcome={result.outcome} /></TableCell>
                <TableCell className={styles.objective}>
                  {result.objective_preview || '(No objective recorded)'}
                  <Text size={100} className={styles.secondary}>{result.attack_result_id}</Text>
                </TableCell>
                <TableCell className={styles.metadata}>{result.operation ?? '(Missing operation)'}</TableCell>
                <TableCell className={styles.metadata}>{result.operator ?? '(Missing operator)'}</TableCell>
                <TableCell className={styles.metadata}>
                  {result.attack_type ?? '(Missing attack type)'}
                  <Text size={200} className={styles.secondary}>{result.targeted_harm_categories.join(', ') || '(No targeted harms recorded)'}</Text>
                </TableCell>
                <TableCell className={styles.metadata}>
                  {result.target_model ?? '(Missing model)'}
                  <Text size={100} className={styles.secondary}>{result.target_identifier_hash ?? '(Missing target identity)'}</Text>
                </TableCell>
                <TableCell className={styles.date}><time dateTime={result.updated_at}>{formatAnalyticsTime(result.updated_at)}</time></TableCell>
                <TableCell>
                  <Button className={styles.button} appearance="subtle" icon={<OpenRegular />}
                    aria-label={`Open result ${result.attack_result_id}`}
                    onClick={() => { onOpenAttack(result.attack_result_id) }} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      {results.items.length === 0 && <Text>No results on this page. Return to the first page or reload the report.</Text>}
      <Text size={200} className={styles.secondary}>
        Page read: <time dateTime={results.computed_at}>{formatAnalyticsTime(results.computed_at)}</time>.
        {' '}Each page is a fresh read; saved results may change between requests.
      </Text>
      <HistoryPagination
        page={page}
        disabled={disabled || loading}
        isLastPage={!results.has_more}
        onPrevPage={onFirst}
        onNextPage={onNext}
      />
    </section>
  )
}
