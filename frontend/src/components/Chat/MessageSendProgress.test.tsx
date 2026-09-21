import type { ReactNode } from 'react'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { MessageSendState, TrackedMessageSend } from '@/types'

import MessageSendProgress from './MessageSendProgress'

function TestWrapper({ children }: { children: ReactNode }) {
  return <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
}

const send: TrackedMessageSend = {
  status: {
    send_id: 'send',
    attack_result_id: 'attack',
    source_conversation_id: 'source',
    requested_count: 2,
    state: 'failed',
    branches: [
      { conversation_id: 'source', state: 'completed', error: null },
      { conversation_id: 'failed-copy', state: 'failed', error: 'A converter failed' },
    ],
    error: null,
    failure_stage: 'sending',
  },
  trackingError: null,
}

describe('MessageSendProgress', () => {
  it('offers direct access to a failed conversation and dismissal', async () => {
    const user = userEvent.setup()
    const onSelect = jest.fn()
    const onDismiss = jest.fn()
    render(
      <TestWrapper>
        <MessageSendProgress
          sends={[send]}
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
    expect(onDismiss).toHaveBeenCalledWith('send')
  })

  it('hides progress from a different attack', () => {
    render(
      <TestWrapper>
        <MessageSendProgress
          sends={[send]}
          attackResultId="another-attack"
          onSelectConversation={jest.fn()}
          onDismiss={jest.fn()}
          onRetry={jest.fn()}
        />
      </TestWrapper>,
    )
    expect(screen.queryByText(/sends finished/)).not.toBeInTheDocument()
  })

  it('shows each branch state and opens successful or in-flight conversations', async () => {
    const user = userEvent.setup()
    const onSelect = jest.fn()
    render(
      <TestWrapper>
        <MessageSendProgress
          sends={[{
            ...send,
            status: {
              ...send.status,
              state: 'running',
              failure_stage: null,
              branches: [
                send.status.branches[0],
                { conversation_id: 'copy', state: 'sending', error: null },
              ],
            },
          }]}
          attackResultId="attack"
          onSelectConversation={onSelect}
          onDismiss={jest.fn()}
          onRetry={jest.fn()}
        />
      </TestWrapper>,
    )
    await user.click(screen.getByRole('button', { name: 'Conversation 1: completed' }))
    expect(onSelect).toHaveBeenLastCalledWith('source')
    await user.click(screen.getByRole('button', { name: 'Conversation 2: sending' }))
    expect(onSelect).toHaveBeenLastCalledWith('copy')
    expect(screen.queryByRole('button', { name: 'Dismiss repetition progress' })).not.toBeInTheDocument()
  })

  it('offers a status refresh instead of a new send when tracking is lost', async () => {
    const user = userEvent.setup()
    const onRetry = jest.fn<Promise<void>, [string]>().mockResolvedValue(undefined)
    render(
      <TestWrapper>
        <MessageSendProgress
          sends={[{
            ...send,
            status: { ...send.status, requested_count: 1 },
            trackingError: 'Connection lost; sends may still finish.',
          }]}
          attackResultId="attack"
          onSelectConversation={jest.fn()}
          onDismiss={jest.fn()}
          onRetry={onRetry}
        />
      </TestWrapper>,
    )
    await user.click(screen.getByRole('button', { name: 'Refresh progress' }))
    expect(onRetry).toHaveBeenCalledWith('send')
  })

  it.each<MessageSendState>(['preparing', 'queued', 'running', 'completed'])(
    'keeps normal count-one %s progress inline rather than showing a card',
    (state: MessageSendState) => {
      render(
        <TestWrapper>
          <MessageSendProgress
            sends={[{
              ...send,
              status: { ...send.status, state, requested_count: 1, failure_stage: null, branches: [send.status.branches[0]] },
            }]}
            attackResultId="attack"
            onSelectConversation={jest.fn()}
            onDismiss={jest.fn()}
            onRetry={jest.fn()}
          />
        </TestWrapper>,
      )
      expect(screen.queryByRole('region', { name: 'Prompt repetition progress' })).not.toBeInTheDocument()
    },
  )

  it('uses the explicit preparation failure stage even when the saved branch count matches', () => {
    render(
      <TestWrapper>
        <MessageSendProgress
          sends={[{
            ...send,
            status: {
              ...send.status,
              requested_count: 1,
              failure_stage: 'preparation',
              branches: [{ conversation_id: 'source', state: 'failed', error: 'Invalid conversion' }],
            },
          }]}
          attackResultId="attack"
          onSelectConversation={jest.fn()}
          onDismiss={jest.fn()}
          onRetry={jest.fn()}
        />
      </TestWrapper>,
    )
    expect(screen.getByText('Could not prepare send')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Inspect failed conversation/ })).toBeInTheDocument()
  })
})
