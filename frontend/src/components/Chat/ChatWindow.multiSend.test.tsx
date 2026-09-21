import { useState } from 'react'
import type { ReactNode } from 'react'

import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router'

import { attacksApi } from '@/services/api'
import { makeTarget } from '@/test-utils/targetFixtures'
import type { AttackSummary, BackendMessage, MessageSendStatus, RequestConverterMode } from '@/types'

import ChatWindow from './ChatWindow'

jest.mock('@/services/api', () => ({
  attacksApi: {
    createAttack: jest.fn(),
    getAttack: jest.fn(),
    getMessages: jest.fn(),
    getConversations: jest.fn(),
    startMessageSend: jest.fn(),
    getMessageSend: jest.fn(),
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

function status(state: MessageSendStatus['state'] = 'completed'): MessageSendStatus {
  return {
    send_id: 'send',
    attack_result_id: 'attack',
    source_conversation_id: 'source',
    requested_count: 2,
    state,
    error: null,
    failure_stage: null,
    branches: state === 'preparing' ? [] : [
      { conversation_id: 'source', state: 'completed', error: null },
      {
        conversation_id: 'copy',
        state: state === 'completed' ? 'completed' : 'sending',
        error: null,
      },
    ],
  }
}

function deferred<T>() {
  let resolve: (value: T) => void = () => { throw new Error('Uninitialized promise') }
  let reject: (error: Error) => void = () => { throw new Error('Uninitialized promise') }
  const promise = new Promise<T>((complete: (value: T) => void, fail: (error: Error) => void) => {
    resolve = complete
    reject = fail
  })
  return { promise, resolve, reject }
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
    api.startMessageSend.mockResolvedValue(status('preparing'))
    api.getMessageSend.mockResolvedValue(status())
  })

  afterEach(() => { jest.useRealTimers() })

  it.each<RequestConverterMode>(['shared', 'per_branch'])(
    'uses one send request with %s conversion and keeps the current conversation', async (mode: RequestConverterMode) => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await user.click(screen.getByRole('button', { name: 'Repetitions: 1' }))
    await user.click(screen.getByRole('button', { name: 'Increase repetitions' }))
    if (mode === 'per_branch') {
      await user.click(screen.getByRole('radio', { name: 'Convert independently for each' }))
    }
    await user.keyboard('{Escape}')
    await user.type(screen.getByRole('textbox'), 'Repeat me')
    await user.click(screen.getByRole('button', { name: 'Send in 2 conversations' }))

    await screen.findByText('2 of 2 sends finished')
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
    expect(api.startMessageSend.mock.calls[0]).toEqual([
      'attack',
      expect.objectContaining({
        count: 2,
        request_converter_mode: mode,
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
    expect(api.startMessageSend.mock.calls[0][0]).toBe('attack')
    expect(api.startMessageSend.mock.calls[0][1].target_conversation_id).toBe('source')
  })

  it.each([1, 2])('keeps the count-%s draft when preparation fails before storing a request', async (count: number) => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    api.startMessageSend.mockResolvedValue({ ...status('preparing'), requested_count: count })
    api.getMessageSend.mockResolvedValue({
      ...status('preparing'), requested_count: count, state: 'failed',
      failure_stage: 'preparation', error: 'Preparation failed',
    })
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    if (count > 1) await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'Keep this draft')
    await user.keyboard('{Enter}')

    await screen.findByText(count === 1 ? 'Could not prepare send' : 'Could not prepare 2 sends')
    expect(screen.getByRole('textbox')).toHaveValue('Keep this draft')
    expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled()
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
  })

  it.each([1, 2])('preserves count-%s draft and recovery for a preparation failure with one saved branch', async (count: number) => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    api.startMessageSend.mockResolvedValue({ ...status('preparing'), requested_count: count })
    api.getMessageSend.mockResolvedValue({
      ...status('preparing'), requested_count: count, state: 'failed',
      failure_stage: 'preparation', error: 'Preparation failed',
      branches: [{
        conversation_id: 'source', state: 'failed', error: 'Preparation failed',
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
    if (count > 1) await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'Keep this draft')
    await user.keyboard('{Enter}')

    await screen.findByRole('button', { name: 'Edit in clean conversation' })
    expect(screen.getByRole('textbox')).toHaveValue('Keep this draft')
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
    expect(api.addMessage).not.toHaveBeenCalled()
  })

  it('does not unlock or clear a newer count-one send when an older repeated send finishes', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    const oldSend = deferred<MessageSendStatus>()
    const newSend = deferred<MessageSendStatus>()
    api.startMessageSend
      .mockResolvedValueOnce(status('preparing'))
      .mockResolvedValue({ ...status('preparing'), send_id: 'new', requested_count: 1 })
    api.getMessageSend.mockResolvedValueOnce(status('running'))
      .mockImplementation((_attack: string, id: string) => id === 'new' ? newSend.promise : oldSend.promise)
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
    await waitFor(() => expect(api.startMessageSend).toHaveBeenCalledTimes(2))
    expect(api.startMessageSend.mock.calls[1][1].pieces[0].original_value).toBe('Newer')
    expect(api.startMessageSend.mock.calls[1][1].count).toBe(1)

    await act(async () => { oldSend.resolve(status()) })
    await screen.findByText('2 of 2 sends finished')
    expect(screen.getByRole('textbox')).toBeDisabled()
    expect(screen.getByRole('textbox')).toHaveValue('Newer')
    await act(async () => {
      newSend.resolve({ ...status(), send_id: 'new', requested_count: 1, branches: [status().branches[0]] })
    })
    await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
    expect(screen.getByRole('textbox')).toHaveValue('')
  })

  it('navigates to a sending copy and repeats only on that selected conversation', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    const oldSend = deferred<MessageSendStatus>()
    api.getMessageSend.mockResolvedValueOnce(status('running')).mockReturnValue(oldSend.promise)
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'First')
    await user.keyboard('{Enter}')
    await screen.findByText('1 of 2 sends finished')
    await user.click(screen.getByRole('button', { name: 'Conversation 2: sending' }))
    await screen.findByText('Answer in copy')
    expect(screen.getByRole('textbox')).toBeDisabled()

    await act(async () => { oldSend.resolve(status()) })
    await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
    const nested = {
      ...status(),
      send_id: 'nested',
      source_conversation_id: 'copy',
      branches: [
        { ...status().branches[0], conversation_id: 'copy' },
        { ...status().branches[1], conversation_id: 'nested-copy' },
      ],
    }
    api.startMessageSend.mockResolvedValue(nested)
    await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'Nested')
    await user.keyboard('{Enter}')
    await waitFor(() => expect(api.startMessageSend).toHaveBeenCalledTimes(2))
    expect(api.startMessageSend.mock.calls[1][1].target_conversation_id).toBe('copy')
    expect(api.startMessageSend.mock.calls[1][1].count).toBe(2)
    expect(api.addMessage).not.toHaveBeenCalled()
  })

  it.each([1, 2])('refreshes lost count-%s progress without restoring a safe-to-resend draft or submitting again', async (count: number) => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    api.startMessageSend.mockResolvedValue({ ...status('preparing'), requested_count: count })
    api.getMessageSend.mockRejectedValueOnce(new Error('Disconnected'))
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    if (count > 1) await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'May already be sent')
    await user.keyboard('{Enter}')
    await screen.findByRole('button', { name: 'Refresh progress' })
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue(''))
    expect(screen.getAllByText(/do not resend automatically/).length).toBeGreaterThan(0)

    api.getMessageSend.mockResolvedValue({
      ...status(), requested_count: count, branches: status().branches.slice(0, count),
    })
    await user.click(screen.getByRole('button', { name: 'Refresh progress' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh progress' })).not.toBeInTheDocument())
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
    expect(api.addMessage).not.toHaveBeenCalled()
  })

  it('leaves a new attack draft alone when an older send finishes', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    const oldSend = deferred<MessageSendStatus>()
    api.getMessageSend.mockResolvedValueOnce(status('running')).mockReturnValue(oldSend.promise)
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await selectTwoSends(user)
    await user.type(screen.getByRole('textbox'), 'Old attack')
    await user.keyboard('{Enter}')
    await screen.findByText('1 of 2 sends finished')
    await user.click(screen.getByRole('button', { name: 'New Attack' }))
    await user.type(screen.getByRole('textbox'), 'New attack draft')
    await act(async () => { oldSend.resolve(status()) })

    expect(screen.getByRole('textbox')).toHaveValue('New attack draft')
    expect(screen.queryByText(/sends finished/)).not.toBeInTheDocument()
    expect(api.getAttack).not.toHaveBeenCalled()
  })

  it('keeps count-one progress inline and avoids duplicate optimistic messages after switching', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    const completion = deferred<MessageSendStatus>()
    api.startMessageSend.mockResolvedValue({ ...status('preparing'), requested_count: 1 })
    api.getMessageSend.mockReturnValue(completion.promise)
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await user.type(screen.getByRole('textbox'), 'Stored while preparing')
    await user.keyboard('{Enter}')
    await waitFor(() => expect(api.getMessageSend).toHaveBeenCalledTimes(1))
    expect(screen.getByRole('textbox')).toHaveValue('Stored while preparing')
    expect(screen.queryByRole('region', { name: 'Prompt repetition progress' })).not.toBeInTheDocument()

    api.getMessages.mockImplementation(async (_id: string, conversationId: string) => ({
      conversation_id: conversationId,
      target_response_status: null,
      messages: conversationId === 'source' ? [
        ...messages('source'),
        {
          ...messages('source')[0],
          role: 'user',
          turn_number: 2,
          message_pieces: [{
            ...messages('source')[0].message_pieces[0],
            id: 'stored-request',
            original_value: 'Stored while preparing',
            converted_value: 'Stored while preparing',
          }],
        },
      ] : messages('copy'),
    }))
    await user.click(screen.getByRole('button', { name: 'Toggle conversations panel' }))
    await user.click(await screen.findByRole('button', { name: 'Select conversation copy' }))
    await screen.findByText('Answer in copy')
    await user.click(screen.getByRole('button', { name: 'Select conversation source' }))
    await screen.findByText('Answer in source')
    expect(within(screen.getByTestId('message-list')).getAllByText('Stored while preparing')).toHaveLength(1)
    expect(screen.getByRole('textbox')).toBeDisabled()
    await act(async () => {
      completion.resolve({ ...status(), requested_count: 1, branches: [status().branches[0]] })
    })
    expect(screen.getByRole('textbox')).toHaveValue('')
    expect(screen.queryByRole('region', { name: 'Prompt repetition progress' })).not.toBeInTheDocument()
    expect(api.addMessage).not.toHaveBeenCalled()
  })

  it.each(['transcript', 'metadata'])('refreshes a failed terminal %s read without resubmitting count-one', async (read: string) => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    api.startMessageSend.mockResolvedValue({ ...status('preparing'), requested_count: 1 })
    api.getMessageSend.mockResolvedValue({ ...status(), requested_count: 1, branches: [status().branches[0]] })
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    if (read === 'transcript') api.getMessages.mockRejectedValueOnce(new Error('Transcript read failed'))
    else api.getAttack.mockRejectedValueOnce(new Error('Metadata read failed'))
    await user.type(screen.getByRole('textbox'), 'Do not resend')
    await user.keyboard('{Enter}')
    await screen.findByRole('button', { name: 'Refresh progress' })
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue(''))
    await user.type(screen.getByRole('textbox'), 'Newer edit')
    await user.click(screen.getByRole('button', { name: 'Refresh progress' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh progress' })).not.toBeInTheDocument())
    expect(screen.getByRole('textbox')).toHaveValue('Newer edit')
    expect(screen.getByText('Answer in source')).toBeInTheDocument()
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
    expect(api.getMessageSend).toHaveBeenCalledTimes(2)
  })

  it.each(['finalization', 'interrupted'] as const)(
    'shows a count-one %s failure without presenting the old draft as safe to resend',
    async (failureStage: 'finalization' | 'interrupted') => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
      const terminal = deferred<MessageSendStatus>()
      api.startMessageSend.mockResolvedValue({ ...status('preparing'), requested_count: 1 })
      api.getMessageSend.mockResolvedValueOnce({
        ...status('running'), requested_count: 1,
        branches: [{ conversation_id: 'source', state: failureStage === 'finalization' ? 'completed' : 'sending', error: null }],
      }).mockReturnValue(terminal.promise)
      render(<TestWrapper><Chat /></TestWrapper>)
      await screen.findByText('Answer in source')
      await user.type(screen.getByRole('textbox'), 'Uncertain send')
      await user.keyboard('{Enter}')
      await waitFor(() => expect(api.getMessageSend).toHaveBeenCalledTimes(2))
      expect(screen.getByRole('textbox')).toBeDisabled()
      expect(screen.getByRole('textbox')).toHaveValue('Uncertain send')
      await act(async () => {
        terminal.resolve({
          ...status(), requested_count: 1, state: 'failed', failure_stage: failureStage,
          error: 'Sending may have completed; inspect saved history.',
          branches: failureStage === 'finalization' ? [status().branches[0]] : [],
        })
      })
      await screen.findByText('Send failed')
      await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue(''))
      expect(screen.getByText(/inspect saved history/)).toBeInTheDocument()
      expect(screen.queryByText('...')).not.toBeInTheDocument()
      expect(api.startMessageSend).toHaveBeenCalledTimes(1)
    },
  )

  it('releases count-one loading guards when resumed tracking fails and can refresh again', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime })
    const resumedRead = deferred<MessageSendStatus>()
    api.startMessageSend.mockResolvedValue({ ...status('preparing'), requested_count: 1 })
    api.getMessageSend.mockRejectedValueOnce(new Error('Initial status unavailable'))
    render(<TestWrapper><Chat /></TestWrapper>)
    await screen.findByText('Answer in source')
    await user.type(screen.getByRole('textbox'), 'Original')
    await user.keyboard('{Enter}')
    await screen.findByRole('button', { name: 'Refresh progress' })
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue(''))
    await user.type(screen.getByRole('textbox'), 'Newer draft')

    api.getMessageSend.mockResolvedValueOnce({
      ...status('running'), requested_count: 1,
      branches: [{ conversation_id: 'source', state: 'sending', error: null }],
    }).mockReturnValue(resumedRead.promise)
    await user.click(screen.getByRole('button', { name: 'Refresh progress' }))
    await waitFor(() => expect(screen.getByRole('textbox')).toBeDisabled())
    await act(async () => { resumedRead.reject(new Error('Status unavailable again')) })
    await screen.findByRole('button', { name: 'Refresh progress' })
    expect(screen.getByRole('textbox')).toBeEnabled()
    expect(screen.getByRole('textbox')).toHaveValue('Newer draft')

    api.getMessageSend.mockResolvedValue({ ...status(), requested_count: 1, branches: [status().branches[0]] })
    await user.click(screen.getByRole('button', { name: 'Refresh progress' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh progress' })).not.toBeInTheDocument())
    expect(screen.getByRole('textbox')).toBeEnabled()
    expect(screen.getByRole('textbox')).toHaveValue('Newer draft')
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
  })
})
