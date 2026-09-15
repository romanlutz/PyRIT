import type { ReactNode } from 'react'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { TrackedMessageBatch } from '@/types'

import MessageBatchProgress from './MessageBatchProgress'

function TestWrapper({ children }: { children: ReactNode }) {
  return <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
}

const batch: TrackedMessageBatch = {
  status: {
    batch_id: 'batch',
    attack_result_id: 'attack',
    source_conversation_id: 'source',
    requested_count: 2,
    state: 'completed',
    branches: [
      { conversation_id: 'source', state: 'completed', error: null, new_message_piece_ids: ['reply'] },
      { conversation_id: 'failed-copy', state: 'failed', error: 'A converter failed', new_message_piece_ids: ['error'] },
    ],
    error: null,
  },
  trackingError: null,
}

describe('MessageBatchProgress', () => {
  it('offers direct access to a failed conversation and dismissal', async () => {
    const user = userEvent.setup()
    const onSelect = jest.fn()
    const onDismiss = jest.fn()
    render(
      <TestWrapper>
        <MessageBatchProgress
          batches={[batch]}
          attackResultId="attack"
          onSelectConversation={onSelect}
          onDismiss={onDismiss}
          onRetry={jest.fn()}
        />
      </TestWrapper>,
    )
    expect(screen.getByText('2 of 2 sends finished')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Inspect failed conversation/ }))
    expect(onSelect).toHaveBeenCalledWith('failed-copy')
    await user.click(screen.getByRole('button', { name: 'Dismiss repetition progress' }))
    expect(onDismiss).toHaveBeenCalledWith('batch')
  })

  it('hides progress from a different attack', () => {
    render(
      <TestWrapper>
        <MessageBatchProgress
          batches={[batch]}
          attackResultId="another-attack"
          onSelectConversation={jest.fn()}
          onDismiss={jest.fn()}
          onRetry={jest.fn()}
        />
      </TestWrapper>,
    )
    expect(screen.queryByText(/sends finished/)).not.toBeInTheDocument()
  })

  it('offers a status refresh instead of a new send when tracking is lost', async () => {
    const user = userEvent.setup()
    const onRetry = jest.fn<Promise<void>, [string]>().mockResolvedValue(undefined)
    render(
      <TestWrapper>
        <MessageBatchProgress
          batches={[{ ...batch, trackingError: 'Connection lost; sends may still finish.' }]}
          attackResultId="attack"
          onSelectConversation={jest.fn()}
          onDismiss={jest.fn()}
          onRetry={onRetry}
        />
      </TestWrapper>,
    )
    await user.click(screen.getByRole('button', { name: 'Refresh progress' }))
    expect(onRetry).toHaveBeenCalledWith('batch')
  })
})
