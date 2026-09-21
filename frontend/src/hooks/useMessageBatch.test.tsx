import { act, renderHook, waitFor } from '@testing-library/react'

import { attacksApi } from '@/services/api'
import type { MessageBatchInput, MessageBatchStatus } from '@/types'

import { MessageBatchTrackingError, useMessageBatch } from './useMessageBatch'

jest.mock('@/services/api', () => ({
  attacksApi: {
    startMessageBatch: jest.fn(),
    getMessageBatch: jest.fn(),
  },
}))

const api = jest.mocked(attacksApi)
const request: MessageBatchInput = {
  count: 2,
  request_converter_mode: 'shared',
  role: 'user',
  pieces: [{ data_type: 'text', original_value: 'Hello' }],
  send: true,
  target_registry_name: 'target',
  target_conversation_id: 'source',
}

function makeStatus(overrides: Partial<MessageBatchStatus> = {}): MessageBatchStatus {
  return {
    batch_id: 'batch',
    attack_result_id: 'attack',
    source_conversation_id: 'source',
    requested_count: 2,
    state: 'preparing',
    branches: [],
    error: null,
    ...overrides,
  }
}

function deferred<T>() {
  let resolve: (value: T) => void = () => { throw new Error('Deferred promise not initialized') }
  const promise = new Promise<T>((complete: (value: T) => void) => { resolve = complete })
  return { promise, resolve }
}

describe('useMessageBatch', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    api.startMessageBatch.mockReset()
    api.getMessageBatch.mockReset()
  })

  it('publishes acceptance before a slow branch settles without resubmitting', async () => {
    const pending = deferred<MessageBatchStatus>()
    api.startMessageBatch.mockResolvedValue(makeStatus())
    api.getMessageBatch.mockReturnValue(pending.promise)
    const progress = jest.fn<Promise<void>, [MessageBatchStatus]>().mockResolvedValue(undefined)
    const { result } = renderHook(() => useMessageBatch())

    let completion: Promise<MessageBatchStatus> | undefined
    await act(async () => {
      completion = result.current.executeBatch('attack', request, progress)
    })
    expect(result.current.batches[0].status.state).toBe('preparing')
    expect(progress).toHaveBeenCalledWith(makeStatus())
    expect(api.startMessageBatch).toHaveBeenCalledTimes(1)
    expect(api.getMessageBatch).toHaveBeenCalledTimes(1)

    const done = makeStatus({
      state: 'completed',
      branches: [
        { conversation_id: 'source', state: 'completed', new_message_piece_ids: ['a'], error: null },
        { conversation_id: 'copy', state: 'failed', new_message_piece_ids: ['b'], error: 'Provider unavailable' },
      ],
    })
    await act(async () => {
      pending.resolve(done)
      await completion
    })
    expect(result.current.batches[0].status).toEqual(done)
    expect(progress).toHaveBeenLastCalledWith(done)
    expect(api.startMessageBatch).toHaveBeenCalledTimes(1)
  })

  it('retries only progress reads after a tracking failure', async () => {
    api.startMessageBatch.mockResolvedValue(makeStatus())
    api.getMessageBatch.mockRejectedValueOnce(new Error('Disconnected'))
    const progress = jest.fn<Promise<void>, [MessageBatchStatus]>().mockResolvedValue(undefined)
    const { result } = renderHook(() => useMessageBatch())

    await act(async () => {
      await expect(result.current.executeBatch('attack', request, progress)).rejects.toBeInstanceOf(MessageBatchTrackingError)
    })
    expect(result.current.batches[0].trackingError).toMatch(/do not resend automatically/)

    api.getMessageBatch.mockResolvedValue(makeStatus({ state: 'completed' }))
    await act(async () => { await result.current.retryTracking('batch') })
    expect(api.startMessageBatch).toHaveBeenCalledTimes(1)
    expect(result.current.batches[0].trackingError).toBeNull()
    expect(result.current.batches[0].status.state).toBe('completed')
  })

  it('surfaces preparation failures without treating them as lost progress', async () => {
    const failed = makeStatus({ state: 'failed', error: 'Converter could not prepare the request' })
    api.startMessageBatch.mockResolvedValue(failed)
    const { result } = renderHook(() => useMessageBatch())
    await act(async () => {
      await expect(result.current.executeBatch('attack', request, async () => {})).resolves.toEqual(failed)
    })
    expect(api.getMessageBatch).not.toHaveBeenCalled()
    expect(result.current.batches[0].trackingError).toBeNull()
    act(() => { result.current.dismissBatch('batch') })
    expect(result.current.batches).toEqual([])
  })

  it('aborts status reads on unmount, not the accepted send', async () => {
    api.startMessageBatch.mockResolvedValue(makeStatus())
    api.getMessageBatch.mockImplementation((_attack: string, _id: string, signal?: AbortSignal) =>
      new Promise((_resolve: (status: MessageBatchStatus) => void, reject: (error: Error) => void) => {
        signal?.addEventListener('abort', () => reject(new DOMException('Stopped', 'AbortError')), { once: true })
      }),
    )
    const { result, unmount } = renderHook(() => useMessageBatch())
    let completion: Promise<MessageBatchStatus> | undefined
    await act(async () => {
      completion = result.current.executeBatch('attack', request, async () => {})
      void completion.catch(() => {})
    })
    await waitFor(() => expect(api.getMessageBatch).toHaveBeenCalled())
    unmount()
    await expect(completion).rejects.toHaveProperty('name', 'AbortError')
    expect(api.getMessageBatch.mock.calls[0][2]?.aborted).toBe(true)
    expect(api.startMessageBatch).toHaveBeenCalledTimes(1)
  })

  it('bounds retained terminal progress', async () => {
    const { result } = renderHook(() => useMessageBatch())
    for (let index = 0; index < 7; index++) {
      api.startMessageBatch.mockResolvedValueOnce(makeStatus({ batch_id: `batch-${index}`, state: 'completed' }))
      await act(async () => {
        await result.current.executeBatch('attack', request, async () => {})
      })
    }
    expect(result.current.batches).toHaveLength(5)
    expect(result.current.batches[0].status.batch_id).toBe('batch-2')
    expect(api.getMessageBatch).not.toHaveBeenCalled()
  })

  it('does not retain a request rejected before acceptance', async () => {
    api.startMessageBatch.mockRejectedValue(new Error('Target mismatch'))
    const { result } = renderHook(() => useMessageBatch())
    await act(async () => {
      await expect(result.current.executeBatch('attack', request, async () => {})).rejects.toThrow('Target mismatch')
    })
    expect(result.current.batches).toEqual([])
    expect(api.getMessageBatch).not.toHaveBeenCalled()
  })
})
