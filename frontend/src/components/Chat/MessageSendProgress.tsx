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

import type { MessageSendBranch, TrackedMessageSend } from '@/types'

import { useMessageSendProgressStyles } from './MessageSendProgress.styles'

interface MessageSendProgressProps {
  sends: TrackedMessageSend[]
  attackResultId: string | null
  onSelectConversation: (conversationId: string) => void
  onRetry: (sendId: string) => Promise<void>
  onDismiss: (sendId: string) => void
}

export default function MessageSendProgress({
  sends,
  attackResultId,
  onSelectConversation,
  onRetry,
  onDismiss,
}: MessageSendProgressProps) {
  const styles = useMessageSendProgressStyles()
  const visible = sends.filter(({ status, trackingError }: TrackedMessageSend) =>
    status.attack_result_id === attackResultId
    && (status.requested_count > 1 || trackingError || status.state === 'failed'
      || status.branches.some((branch: MessageSendBranch) => branch.state === 'failed')),
  )
  if (visible.length === 0) return null

  return (
    <div className={styles.root} role="region" aria-label="Prompt repetition progress">
      {visible.map(({ status, trackingError }: TrackedMessageSend) => {
        const failed = status.branches.filter((branch: MessageSendBranch) => branch.state === 'failed')
        const completed = status.branches.filter((branch: MessageSendBranch) =>
          branch.state === 'completed' || branch.state === 'failed',
        ).length
        const done = status.state === 'completed' || status.state === 'failed'
        const failedPreparation = status.failure_stage === 'preparation'
        const countLabel = status.requested_count === 1 ? 'send' : `${status.requested_count} sends`
        const intent = trackingError || failed.length || status.state === 'failed'
          ? 'warning'
          : done ? 'success' : 'info'
        return (
          <MessageBar
            key={status.send_id}
            intent={intent}
            icon={!done && !trackingError ? <Spinner size="tiny" /> : undefined}
            data-testid={`message-send-${status.send_id}`}
          >
            <MessageBarBody>
              <MessageBarTitle>
                {failedPreparation
                  ? `Could not prepare ${countLabel}`
                  : status.requested_count === 1 && status.state === 'failed'
                  ? 'Send failed'
                  : status.state === 'preparing'
                  ? `Preparing ${countLabel}`
                  : `${completed} of ${status.requested_count} sends finished`}
              </MessageBarTitle>
              {trackingError || status.error || (status.state === 'queued' ? 'Waiting for target capacity.' : '')}
              {status.branches.length > 0 && (
                <div className={styles.details}>
                  {status.branches.map((branch: MessageSendBranch, index: number) => (
                    <Tooltip
                      key={branch.conversation_id}
                      content={branch.error || `Open conversation ${branch.conversation_id}`}
                      relationship="description"
                    >
                      <Button
                        className={styles.button}
                        appearance="transparent"
                        size="small"
                        aria-label={branch.state === 'failed'
                          ? `Inspect failed conversation ${branch.conversation_id.slice(0, 8)}`
                          : undefined}
                        onClick={() => onSelectConversation(branch.conversation_id)}
                      >
                        Conversation {index + 1}: {branch.state}
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
                  onClick={() => onDismiss(status.send_id)}
                />
              ) : undefined}
            >
              {trackingError && (
                <Button
                  className={styles.button}
                  appearance="transparent"
                  onClick={() => { void onRetry(status.send_id) }}
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
