import { act, renderHook, waitFor } from '@testing-library/react'

import { attacksApi } from '@/services/api'
import type { MessageSendInput, MessageSendStatus } from '@/types'

import { MessageSendTrackingError, useMessageSend } from './useMessageSend'

jest.mock('@/services/api', () => ({
  attacksApi: {
    startMessageSend: jest.fn(),
    getMessageSend: jest.fn(),
  },
}))

const api = jest.mocked(attacksApi)
const request: MessageSendInput = {
  count: 2,
  request_converter_mode: 'shared',
  role: 'user',
  pieces: [{ data_type: 'text', original_value: 'Hello' }],
  send: true,
  target_registry_name: 'target',
  target_conversation_id: 'source',
}

function makeStatus(overrides: Partial<MessageSendStatus> = {}): MessageSendStatus {
  return {
    send_id: 'send',
    attack_result_id: 'attack',
    source_conversation_id: 'source',
    requested_count: 2,
    state: 'preparing',
    branches: [],
    error: null,
    failure_stage: null,
    ...overrides,
  }
}

function deferred<T>() {
  let resolve: (value: T) => void = () => { throw new Error('Deferred promise not initialized') }
  const promise = new Promise<T>((complete: (value: T) => void) => { resolve = complete })
  return { promise, resolve }
}

describe('useMessageSend', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    api.startMessageSend.mockReset()
    api.getMessageSend.mockReset()
  })

  it('publishes acceptance before a slow branch settles without resubmitting', async () => {
    const pending = deferred<MessageSendStatus>()
    api.startMessageSend.mockResolvedValue(makeStatus())
    api.getMessageSend.mockReturnValue(pending.promise)
    const progress = jest.fn<Promise<void>, [MessageSendStatus]>().mockResolvedValue(undefined)
    const { result } = renderHook(() => useMessageSend())

    let completion: Promise<MessageSendStatus> | undefined
    await act(async () => {
      completion = result.current.executeSend('attack', request, progress)
    })
    expect(result.current.sends[0].status.state).toBe('preparing')
    expect(progress).toHaveBeenCalledWith(makeStatus())
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
    expect(api.getMessageSend).toHaveBeenCalledTimes(1)

    const done = makeStatus({
      state: 'failed',
      failure_stage: 'sending',
      branches: [
        { conversation_id: 'source', state: 'completed', error: null },
        { conversation_id: 'copy', state: 'failed', error: 'Provider unavailable' },
      ],
    })
    await act(async () => {
      pending.resolve(done)
      await completion
    })
    expect(result.current.sends[0].status).toEqual(done)
    expect(progress).toHaveBeenLastCalledWith(done)
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
  })

  it('retries only progress reads after a tracking failure', async () => {
    api.startMessageSend.mockResolvedValue(makeStatus({ requested_count: 1 }))
    api.getMessageSend.mockRejectedValueOnce(new Error('Disconnected'))
    const progress = jest.fn<Promise<void>, [MessageSendStatus]>().mockResolvedValue(undefined)
    const { result } = renderHook(() => useMessageSend())

    await act(async () => {
      await expect(result.current.executeSend('attack', { ...request, count: 1 }, progress))
        .rejects.toBeInstanceOf(MessageSendTrackingError)
    })
    expect(result.current.sends[0].trackingError).toMatch(/do not resend automatically/)

    api.getMessageSend.mockResolvedValue(makeStatus({ requested_count: 1, state: 'completed' }))
    await act(async () => { await result.current.retryTracking('send') })
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
    expect(result.current.sends[0].trackingError).toBeNull()
    expect(result.current.sends[0].status.state).toBe('completed')
  })

  it('surfaces preparation failures without treating them as lost progress', async () => {
    const failed = makeStatus({
      state: 'failed', failure_stage: 'preparation', error: 'Converter could not prepare the request',
    })
    api.startMessageSend.mockResolvedValue(failed)
    const { result } = renderHook(() => useMessageSend())
    await act(async () => {
      await expect(result.current.executeSend('attack', request, async () => {})).resolves.toEqual(failed)
    })
    expect(api.getMessageSend).not.toHaveBeenCalled()
    expect(result.current.sends[0].trackingError).toBeNull()
    act(() => { result.current.dismissSend('send') })
    expect(result.current.sends).toEqual([])
  })

  it('aborts status reads on unmount, not the accepted send', async () => {
    api.startMessageSend.mockResolvedValue(makeStatus({ requested_count: 1 }))
    api.getMessageSend.mockImplementation((_attack: string, _id: string, signal?: AbortSignal) =>
      new Promise((_resolve: (status: MessageSendStatus) => void, reject: (error: Error) => void) => {
        signal?.addEventListener('abort', () => reject(new DOMException('Stopped', 'AbortError')), { once: true })
      }),
    )
    const { result, unmount } = renderHook(() => useMessageSend())
    let completion: Promise<MessageSendStatus> | undefined
    await act(async () => {
      completion = result.current.executeSend('attack', { ...request, count: 1 }, async () => {})
      void completion.catch(() => {})
    })
    await waitFor(() => expect(api.getMessageSend).toHaveBeenCalled())
    unmount()
    await expect(completion).rejects.toHaveProperty('name', 'AbortError')
    expect(api.getMessageSend.mock.calls[0][2]?.aborted).toBe(true)
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
  })

  it('bounds retained terminal progress', async () => {
    const { result } = renderHook(() => useMessageSend())
    for (let index = 0; index < 7; index++) {
      api.startMessageSend.mockResolvedValueOnce(makeStatus({ send_id: `send-${index}`, state: 'completed' }))
      await act(async () => {
        await result.current.executeSend('attack', request, async () => {})
      })
    }
    expect(result.current.sends).toHaveLength(5)
    expect(result.current.sends[0].status.send_id).toBe('send-2')
    expect(api.getMessageSend).not.toHaveBeenCalled()
  })

  it('does not retain a request rejected before acceptance', async () => {
    api.startMessageSend.mockRejectedValue(new Error('Target mismatch'))
    const { result } = renderHook(() => useMessageSend())
    await act(async () => {
      await expect(result.current.executeSend('attack', request, async () => {})).rejects.toThrow('Target mismatch')
    })
    expect(result.current.sends).toEqual([])
    expect(api.getMessageSend).not.toHaveBeenCalled()
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
  })

  it('completes a count-one send without advancing any client timer', async () => {
    jest.useFakeTimers()
    try {
      api.startMessageSend.mockResolvedValue(makeStatus({ requested_count: 1 }))
      api.getMessageSend
        .mockResolvedValueOnce(makeStatus({ requested_count: 1, state: 'running' }))
        .mockResolvedValueOnce(makeStatus({ requested_count: 1, state: 'completed' }))
      const { result } = renderHook(() => useMessageSend())
      await act(async () => {
        await expect(result.current.executeSend('attack', { ...request, count: 1 }, async () => {}))
          .resolves.toHaveProperty('state', 'completed')
      })
      expect(api.getMessageSend).toHaveBeenCalledTimes(2)
      expect(api.startMessageSend).toHaveBeenCalledTimes(1)
    } finally {
      jest.useRealTimers()
    }
  })

  it('retains terminal tracking failures so metadata can be refreshed without resending', async () => {
    const done = makeStatus({ requested_count: 1, state: 'completed' })
    api.startMessageSend.mockResolvedValue(done)
    api.getMessageSend.mockResolvedValue(done)
    const progress = jest.fn<Promise<void>, [MessageSendStatus]>()
      .mockRejectedValueOnce(new Error('Could not refresh attack details'))
      .mockResolvedValue(undefined)
    const { result } = renderHook(() => useMessageSend())
    await act(async () => {
      await expect(result.current.executeSend('attack', { ...request, count: 1 }, progress))
        .rejects.toBeInstanceOf(MessageSendTrackingError)
    })
    expect(result.current.sends[0].trackingError).toMatch(/Could not refresh attack details/)
    await act(async () => { await result.current.retryTracking('send') })
    expect(result.current.sends[0].trackingError).toBeNull()
    expect(progress).toHaveBeenCalledTimes(2)
    expect(api.getMessageSend).toHaveBeenCalledTimes(1)
    expect(api.startMessageSend).toHaveBeenCalledTimes(1)
  })

  it('never overlaps progress reads or dismisses an active operation', async () => {
    const pending = deferred<MessageSendStatus>()
    api.startMessageSend.mockResolvedValue(makeStatus())
    api.getMessageSend.mockReturnValue(pending.promise)
    const { result } = renderHook(() => useMessageSend())
    let completion: Promise<MessageSendStatus> | undefined
    await act(async () => {
      completion = result.current.executeSend('attack', request, async () => {})
    })
    await act(async () => {
      result.current.dismissSend('send')
      await result.current.retryTracking('send')
      await result.current.retryTracking('send')
    })
    expect(result.current.sends).toHaveLength(1)
    expect(api.getMessageSend).toHaveBeenCalledTimes(1)
    await act(async () => {
      pending.resolve(makeStatus({ state: 'completed' }))
      await completion
    })
  })
})
