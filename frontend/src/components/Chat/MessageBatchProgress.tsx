import {
  Button,
  MessageBar,
  MessageBarActions,
  MessageBarBody,
  MessageBarTitle,
  Spinner,
  Tooltip,
} from '@fluentui/react-components'
import { DismissRegular } from '@fluentui/react-icons'

import type { MessageBatchBranch, TrackedMessageBatch } from '@/types'

import { useMessageBatchProgressStyles } from './MessageBatchProgress.styles'

interface MessageBatchProgressProps {
  batches: TrackedMessageBatch[]
  attackResultId: string | null
  onSelectConversation: (conversationId: string) => void
  onRetry: (batchId: string) => Promise<void>
  onDismiss: (batchId: string) => void
}

export default function MessageBatchProgress({
  batches,
  attackResultId,
  onSelectConversation,
  onRetry,
  onDismiss,
}: MessageBatchProgressProps) {
  const styles = useMessageBatchProgressStyles()
  const visible = batches.filter((entry: TrackedMessageBatch) => entry.status.attack_result_id === attackResultId)
  if (visible.length === 0) return null

  return (
    <div className={styles.root} aria-label="Prompt repetition progress">
      {visible.map(({ status, trackingError }: TrackedMessageBatch) => {
        const failed = status.branches.filter((branch: MessageBatchBranch) => branch.state === 'failed')
        const completed = status.branches.filter((branch: MessageBatchBranch) =>
          branch.state === 'completed' || branch.state === 'failed',
        ).length
        const done = status.state === 'completed' || status.state === 'failed'
        const failedPreparation = status.state === 'failed' && status.branches.length < status.requested_count
        const intent = trackingError || failed.length || status.state === 'failed'
          ? 'warning'
          : done ? 'success' : 'info'
        return (
          <MessageBar
            key={status.batch_id}
            intent={intent}
            icon={!done && !trackingError ? <Spinner size="tiny" /> : undefined}
            data-testid={`message-batch-${status.batch_id}`}
          >
            <MessageBarBody>
              <MessageBarTitle>
                {failedPreparation
                  ? `Could not prepare ${status.requested_count} sends`
                  : status.state === 'preparing'
                  ? `Preparing ${status.requested_count} sends`
                  : `${completed} of ${status.requested_count} sends finished`}
              </MessageBarTitle>
              {trackingError || status.error || (status.state === 'queued' ? 'Waiting for target capacity.' : '')}
              {failed.length > 0 && (
                <div className={styles.details}>
                  {failed.map((branch: MessageBatchBranch) => (
                    <Tooltip
                      key={branch.conversation_id}
                      content={branch.error || 'Open the conversation to inspect its saved error.'}
                      relationship="description"
                    >
                      <Button
                        className={styles.button}
                        appearance="transparent"
                        size="small"
                        onClick={() => onSelectConversation(branch.conversation_id)}
                      >
                        Inspect failed conversation {branch.conversation_id.slice(0, 8)}
                      </Button>
                    </Tooltip>
                  ))}
                </div>
              )}
            </MessageBarBody>
            <MessageBarActions
              containerAction={(done || trackingError) ? (
                <Button
                  appearance="transparent"
                  className={styles.button}
                  icon={<DismissRegular />}
                  aria-label="Dismiss repetition progress"
                  onClick={() => onDismiss(status.batch_id)}
                />
              ) : undefined}
            >
              {trackingError && (
                <Button
                  className={styles.button}
                  appearance="transparent"
                  onClick={() => { void onRetry(status.batch_id) }}
                >
                  Refresh progress
                </Button>
              )}
            </MessageBarActions>
          </MessageBar>
        )
      })}
    </div>
  )
}
