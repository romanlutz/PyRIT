import { useState } from 'react'
import type { ReactNode } from 'react'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router'

import { attacksApi } from '@/services/api'
import { makeTarget } from '@/test-utils/targetFixtures'
import type { AddMessageResponse, AttackSummary, BackendMessage, MessageBatchStatus } from '@/types'

import ChatWindow from './ChatWindow'

jest.setTimeout(60_000)
jest.mock('@/services/api', () => ({
  attacksApi: {
    createAttack: jest.fn(),
    getAttack: jest.fn(),
    getMessages: jest.fn(),
    getConversations: jest.fn(),
    startMessageBatch: jest.fn(),
    getMessageBatch: jest.fn(),
    addMessage: jest.fn(),
  },
  scoresApi: {},
  convertersApi: {},
  labelsApi: {},
}))
jest.mock('./ConversationTree/ConversationTree', () => ({
  __esModule: true,
  default: ({ onSelectConversation }: { onSelectConversation: (id: string) => void }) => (
    <button onClick={() => onSelectConversation('copy')}>Open copied conversation</button>
  ),
}))

const api = jest.mocked(attacksApi)
const target = makeTarget({ target_registry_name: 'target' })
const summary: AttackSummary = {
  attack_result_id: 'attack',
  conversation_id: 'source',
  attack_type: 'ManualAttack',
  objective: '',
  converters: [],
  message_count: 2,
  related_conversation_ids: ['copy'],
  labels: {},
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:01Z',
}

function messages(conversationId: string): BackendMessage[] {
  return [{
    role: 'assistant',
    turn_number: 1,
    created_at: '2026-01-01T00:00:01Z',
    message_pieces: [{
      id: `${conversationId}-reply`,
      original_value_data_type: 'text',
      converted_value_data_type: 'text',
      original_value: `Answer in ${conversationId}`,
      converted_value: `Answer in ${conversationId}`,
      response_error: 'none',
      scores: [],
    }],
  }]
}

function status(state: MessageBatchStatus['state'] = 'completed'): MessageBatchStatus {
  return {
    batch_id: 'batch',
    attack_result_id: 'attack',
    source_conversation_id: 'source',
    requested_count: 2,
    state,
    error: null,
    branches: state === 'preparing' ? [] : [
      { conversation_id: 'source', state: 'completed', new_message_piece_ids: ['source-reply'], error: null },
      {
        conversation_id: 'copy',
        state: state === 'completed' ? 'completed' : 'sending',
        new_message_piece_ids: [],
        error: null,
      },
    ],
  }
}

function deferred<T>() {
  let resolve: (value: T) => void = () => { throw new Error('Uninitialized promise') }
  const promise = new Promise<T>((complete: (value: T) => void) => { resolve = complete })
  return { promise, resolve }
}

function TestWrapper({ children }: { children: ReactNode }) {
  return <FluentProvider theme={webLightTheme}><MemoryRouter>{children}</MemoryRouter></FluentProvider>
}

function Chat({ fresh = false }: { fresh?: boolean }) {
  const [attack, setAttack] = useState<string | null>(fresh ? null : 'attack')
  const [main, setMain] = useState<string | null>(fresh ? null : 'source')
  const [active, setActive] = useState<string | null>(fresh ? null : 'source')
  return (
    <ChatWindow
      activeTarget={target}
      attackResultId={attack}
      conversationId={main}
      activeConversationId={active}
      targetResolutionStatus="resolved"
      onNewAttack={() => { setAttack(null); setMain(null); setActive(null) }}
      onConversationCreated={(id: string, conversation: string) => {
        setAttack(id)
        setMain(conversation)
        setActive(conversation)
      }}
      onSelectConversation={setActive}
    />
  )
}

describe('ChatWindow multi-send integration', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    for (const method of Object.values(api)) method.mockReset()
    api.getConversations.mockResolvedValue({
      attack_result_id: 'attack', main_conversation_id: 'source', conversations: [],
    })
    api.getMessages.mockImplementation(async (_id: string, conversationId: string) => ({
      conversation_id: conversationId, messages: messages(conversationId),
    }))
    api.getAttack.mockResolvedValue(summary)
    api.createAttack.mockResolvedValue({
      attack_result_id: 'attack', conversation_id: 'source', created_at: '2026-01-01T00:00:00Z',
    })
    api.startMessageBatch.mockResolvedValue(status('preparing'))
    api.getMessageBatch.mockResolvedValue(status())
  })

  it('sends once through the batch API and keeps the current conversation', async () => {
    const user = userEvent.setup()
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await user.click(screen.getByRole('button', { name: 'Repetitions: 1' }))
    await user.click(screen.getByRole('button', { name: 'Increase repetitions' }))
    await user.click(screen.getByRole('radio', { name: 'Convert independently for each' }))
    await user.keyboard('{Escape}')
    await user.type(screen.getByRole('textbox'), 'Repeat me')
    await user.click(screen.getByRole('button', { name: 'Send in 2 conversations' }))

    await screen.findByText('2 of 2 sends finished')
    expect(api.startMessageBatch).toHaveBeenCalledTimes(1)
    expect(api.startMessageBatch.mock.calls[0]).toEqual([
      'attack',
      expect.objectContaining({
        count: 2,
        request_converter_mode: 'per_branch',
        target_conversation_id: 'source',
        pieces: [{ data_type: 'text', original_value: 'Repeat me' }],
      }),
    ])
    expect(api.addMessage).not.toHaveBeenCalled()
    expect(api.createAttack).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Repetitions: 1' })).toBeInTheDocument()
    expect(screen.getByText('Answer in source')).toBeInTheDocument()
  })

  it('creates only one attack before the first multi-send', async () => {
    const user = userEvent.setup()
    render(<TestWrapper><Chat fresh /></TestWrapper>)
    await user.click(screen.getByRole('button', { name: 'Repetitions: 1' }))
    await user.click(screen.getByRole('button', { name: 'Increase repetitions' }))
    await user.keyboard('{Escape}')
    await user.type(screen.getByRole('textbox'), 'First prompt')
    await user.keyboard('{Enter}')

    await screen.findByText('2 of 2 sends finished')
    expect(api.createAttack).toHaveBeenCalledTimes(1)
    expect(api.startMessageBatch.mock.calls[0][0]).toBe('attack')
    expect(api.startMessageBatch.mock.calls[0][1].target_conversation_id).toBe('source')
  })

  it.each([false, true])('restores the draft after a preparation failure (saved error: %s)', async (savedError: boolean) => {
    const user = userEvent.setup()
    api.getMessageBatch.mockResolvedValue({
      ...status('preparing'), state: 'failed', error: 'Preparation failed',
      branches: savedError ? [{
        conversation_id: 'source', state: 'failed', new_message_piece_ids: ['failed-request', 'error'], error: 'Preparation failed',
      }] : [],
    })
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await user.click(screen.getByRole('button', { name: 'Repetitions: 1' }))
    await user.click(screen.getByRole('button', { name: 'Increase repetitions' }))
    await user.keyboard('{Escape}')
    await user.type(screen.getByRole('textbox'), 'Keep this draft')
    await user.keyboard('{Enter}')

    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue('Keep this draft'))
    expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled()
    expect(api.startMessageBatch).toHaveBeenCalledTimes(1)
  })

  it('preserves the draft through tree/chat toggles and opens a selected conversation', async () => {
    const user = userEvent.setup()
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await user.type(screen.getByRole('textbox'), 'Unsent draft')
    await user.click(screen.getByRole('button', { name: 'Show conversation tree' }))
    await screen.findByRole('button', { name: 'Open copied conversation' })
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Return to conversation' }))
    expect(screen.getByRole('textbox')).toHaveValue('Unsent draft')
    expect(screen.getByText('Answer in source')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Show conversation tree' }))
    await user.click(screen.getByRole('button', { name: 'Open copied conversation' }))
    await screen.findByText('Answer in copy')
    expect(screen.getByRole('textbox')).toHaveValue('Unsent draft')
    expect(screen.getByRole('button', { name: 'Show conversation tree' })).toBeInTheDocument()
  })

  it('does not unlock a newer send when an older batch finishes', async () => {
    const user = userEvent.setup()
    const oldBatch = deferred<MessageBatchStatus>()
    const newSend = deferred<AddMessageResponse>()
    api.getMessageBatch.mockResolvedValueOnce(status('running')).mockReturnValue(oldBatch.promise)
    api.addMessage.mockReturnValue(newSend.promise)
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await user.click(screen.getByRole('button', { name: 'Repetitions: 1' }))
    await user.click(screen.getByRole('button', { name: 'Increase repetitions' }))
    await user.keyboard('{Escape}')
    await user.type(screen.getByRole('textbox'), 'Older')
    await user.keyboard('{Enter}')
    await screen.findByText('1 of 2 sends finished')
    await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
    await user.type(screen.getByRole('textbox'), 'Newer')
    await user.keyboard('{Enter}')
    await waitFor(() => expect(api.addMessage).toHaveBeenCalledTimes(1))

    await act(async () => { oldBatch.resolve(status()) })
    await screen.findByText('2 of 2 sends finished')
    expect(screen.getByRole('textbox')).toBeDisabled()
    await act(async () => {
      newSend.resolve({ attack: summary, messages: { conversation_id: 'source', messages: messages('source') } })
    })
    await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
  })
})
