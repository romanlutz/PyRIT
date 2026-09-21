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
  return (
    <FluentProvider theme={webLightTheme}>
      <MemoryRouter>{children}</MemoryRouter>
    </FluentProvider>
  )
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

async function selectTwoSends(user: ReturnType<typeof userEvent.setup>): Promise<void> {
  await user.click(screen.getByRole('button', { name: 'Repetitions: 1' }))
  await user.click(screen.getByRole('button', { name: 'Increase repetitions' }))
  await user.keyboard('{Escape}')
}

describe('ChatWindow multi-send integration', () => {
  beforeEach(() => {
    jest.useFakeTimers()
    jest.clearAllMocks()
    for (const method of Object.values(api)) method.mockReset()
    api.getConversations.mockResolvedValue({
      attack_result_id: 'attack',
      main_conversation_id: 'source',
      conversations: [
        { conversation_id: 'source', message_count: 2 },
        { conversation_id: 'copy', message_count: 2 },
      ],
    })
    api.getMessages.mockImplementation(async (_id: string, conversationId: string) => ({
      conversation_id: conversationId,
      messages: messages(conversationId),
      target_response_status: null,
    }))
    api.getAttack.mockResolvedValue(summary)
    api.createAttack.mockResolvedValue({
      attack_result_id: 'attack', conversation_id: 'source', created_at: '2026-01-01T00:00:00Z',
    })
    api.startMessageBatch.mockResolvedValue(status('preparing'))
    api.getMessageBatch.mockResolvedValue(status())
  })

  afterEach(() => { jest.useRealTimers() })

  it('uses one batch request and keeps the current conversation', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
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
    expect(screen.getByRole('textbox')).toHaveValue('')
  })

  it('creates only one attack before the first multi-send', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    render(<TestWrapper><Chat fresh /></TestWrapper>)
    await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'First prompt')
    await user.keyboard('{Enter}')

    await screen.findByText('2 of 2 sends finished')
    expect(api.createAttack).toHaveBeenCalledTimes(1)
    expect(api.startMessageBatch.mock.calls[0][0]).toBe('attack')
    expect(api.startMessageBatch.mock.calls[0][1].target_conversation_id).toBe('source')
  })

  it('keeps the draft when preparation fails before storing a request', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    api.getMessageBatch.mockResolvedValue({
      ...status('preparing'), state: 'failed', error: 'Preparation failed',
    })
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'Keep this draft')
    await user.keyboard('{Enter}')

    await screen.findByText('Could not prepare 2 sends')
    expect(screen.getByRole('textbox')).toHaveValue('Keep this draft')
    expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled()
    expect(api.startMessageBatch).toHaveBeenCalledTimes(1)
  })

  it('preserves the draft and clean-conversation recovery for a saved preparation failure', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    api.getMessageBatch.mockResolvedValue({
      ...status('preparing'), state: 'failed', error: 'Preparation failed',
      branches: [{
        conversation_id: 'source', state: 'failed', new_message_piece_ids: ['request', 'error'], error: 'Preparation failed',
      }],
    })
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    api.getMessages.mockResolvedValue({
      conversation_id: 'source',
      messages: [
        {
          ...messages('source')[0],
          role: 'user',
          turn_number: 0,
          message_pieces: [{
            ...messages('source')[0].message_pieces[0],
            id: 'request', original_value: 'Keep this draft', converted_value: 'Keep this draft',
          }],
        },
        {
          ...messages('source')[0],
          message_pieces: [{
            ...messages('source')[0].message_pieces[0],
            id: 'error', original_value: 'Preparation failed', converted_value: 'Preparation failed',
            response_error: 'processing',
          }],
        },
      ],
      target_response_status: { response_error: 'processing', request_turn_number: 0, response_turn_number: 1 },
    })
    await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'Keep this draft')
    await user.keyboard('{Enter}')

    await screen.findByRole('button', { name: 'Edit in clean conversation' })
    expect(screen.getByRole('textbox')).toHaveValue('Keep this draft')
    expect(api.startMessageBatch).toHaveBeenCalledTimes(1)
    expect(api.addMessage).not.toHaveBeenCalled()
  })

  it('does not unlock or clear a newer send when an older batch finishes', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    const oldBatch = deferred<MessageBatchStatus>()
    const newSend = deferred<AddMessageResponse>()
    api.getMessageBatch.mockResolvedValueOnce(status('running')).mockReturnValue(oldBatch.promise)
    api.addMessage.mockReturnValue(newSend.promise)
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'Older')
    await user.keyboard('{Enter}')
    await screen.findByText('1 of 2 sends finished')
    await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
    expect(screen.getByRole('textbox')).toHaveValue('')
    await user.type(screen.getByRole('textbox'), 'Newer')
    await user.keyboard('{Enter}')
    await waitFor(() => expect(api.addMessage).toHaveBeenCalledTimes(1))
    expect(api.addMessage.mock.calls[0][1].pieces[0].original_value).toBe('Newer')

    await act(async () => { jest.advanceTimersByTime(1_000); oldBatch.resolve(status()) })
    await screen.findByText('2 of 2 sends finished')
    expect(screen.getByRole('textbox')).toBeDisabled()
    expect(screen.getByRole('textbox')).toHaveValue('Newer')
    await act(async () => {
      newSend.resolve({
        attack: summary,
        messages: { conversation_id: 'source', messages: messages('source'), target_response_status: null },
      })
    })
    await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
    expect(screen.getByRole('textbox')).toHaveValue('')
  })

  it('navigates to a sending copy and repeats only on that selected conversation', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    const oldBatch = deferred<MessageBatchStatus>()
    api.getMessageBatch.mockResolvedValueOnce(status('running')).mockReturnValue(oldBatch.promise)
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'First')
    await user.keyboard('{Enter}')
    await screen.findByText('1 of 2 sends finished')
    await user.click(screen.getByRole('button', { name: 'Conversation 2: sending' }))
    await screen.findByText('Answer in copy')
    expect(screen.getByRole('textbox')).toBeDisabled()

    await act(async () => { jest.advanceTimersByTime(1_000); oldBatch.resolve(status()) })
    await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
    const nested = {
      ...status(),
      batch_id: 'nested',
      source_conversation_id: 'copy',
      branches: [
        { ...status().branches[0], conversation_id: 'copy' },
        { ...status().branches[1], conversation_id: 'nested-copy' },
      ],
    }
    api.startMessageBatch.mockResolvedValue(nested)
    await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'Nested')
    await user.keyboard('{Enter}')
    await waitFor(() => expect(api.startMessageBatch).toHaveBeenCalledTimes(2))
    expect(api.startMessageBatch.mock.calls[1][1].target_conversation_id).toBe('copy')
    expect(api.startMessageBatch.mock.calls[1][1].count).toBe(2)
    expect(api.addMessage).not.toHaveBeenCalled()
  })

  it('refreshes lost progress without restoring a safe-to-resend draft or submitting again', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    api.getMessageBatch.mockRejectedValueOnce(new Error('Disconnected'))
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'May already be sent')
    await user.keyboard('{Enter}')
    await screen.findByRole('button', { name: 'Refresh progress' })
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue(''))
    expect(screen.getAllByText(/do not resend automatically/).length).toBeGreaterThan(0)

    api.getMessageBatch.mockResolvedValue(status())
    await user.click(screen.getByRole('button', { name: 'Refresh progress' }))
    await screen.findByText('2 of 2 sends finished')
    expect(api.startMessageBatch).toHaveBeenCalledTimes(1)
    expect(api.addMessage).not.toHaveBeenCalled()
  })

  it('leaves a new attack draft alone when an older batch finishes', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    const oldBatch = deferred<MessageBatchStatus>()
    api.getMessageBatch.mockResolvedValueOnce(status('running')).mockReturnValue(oldBatch.promise)
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'Old attack')
    await user.keyboard('{Enter}')
    await screen.findByText('1 of 2 sends finished')
    await user.click(screen.getByRole('button', { name: 'New Attack' }))
    await user.type(screen.getByRole('textbox'), 'New attack draft')
    await act(async () => { jest.advanceTimersByTime(1_000); oldBatch.resolve(status()) })

    expect(screen.getByRole('textbox')).toHaveValue('New attack draft')
    expect(screen.queryByText(/sends finished/)).not.toBeInTheDocument()
    expect(api.getAttack).not.toHaveBeenCalled()
  })
})
