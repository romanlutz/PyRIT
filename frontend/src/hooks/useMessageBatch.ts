import { useCallback, useEffect, useRef, useState } from 'react'

import { attacksApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type { MessageBatchInput, MessageBatchStatus, TrackedMessageBatch } from '@/types'

const POLL_INTERVAL_MS = 1_000
const RETAINED_TERMINAL_BATCHES = 5

type ProgressCallback = (status: MessageBatchStatus) => Promise<void>

interface BatchMonitor {
  status: MessageBatchStatus
  onProgress: ProgressCallback
}

export class MessageBatchTrackingError extends Error {
  constructor(detail: string) {
    super(`Progress is unavailable. Sends may still complete; do not resend automatically. ${detail}`)
    this.name = 'MessageBatchTrackingError'
  }
}

function isTerminal(status: MessageBatchStatus): boolean {
  return status.state === 'completed' || status.state === 'failed'
}

function pausePolling(signal: AbortSignal): Promise<void> {
  return new Promise((resolve: () => void, reject: (reason: Error) => void) => {
    if (signal.aborted) {
      reject(new DOMException('Progress tracking stopped', 'AbortError'))
      return
    }
    const onAbort = (): void => {
      clearTimeout(timer)
      reject(new DOMException('Progress tracking stopped', 'AbortError'))
    }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, POLL_INTERVAL_MS)
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

/** Tracks accepted sends independently of the conversation currently being viewed. */
export function useMessageBatch() {
  const [batches, setBatches] = useState<TrackedMessageBatch[]>([])
  const monitors = useRef(new Map<string, BatchMonitor>())
  const controllers = useRef(new Map<string, AbortController>())
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    const activeControllers = controllers.current
    const activeMonitors = monitors.current
    return () => {
      mounted.current = false
      for (const controller of activeControllers.values()) controller.abort()
      activeControllers.clear()
      activeMonitors.clear()
    }
  }, [])

  useEffect(() => {
    const retained = new Set(batches.map((batch: TrackedMessageBatch) => batch.status.batch_id))
    for (const id of monitors.current.keys()) {
      if (!retained.has(id) && !controllers.current.has(id)) monitors.current.delete(id)
    }
  }, [batches])

  const record = useCallback((status: MessageBatchStatus, trackingError: string | null = null): void => {
    if (!mounted.current) return
    setBatches((previous: TrackedMessageBatch[]) => {
      const next = [
        ...previous.filter((entry: TrackedMessageBatch) => entry.status.batch_id !== status.batch_id),
        { status, trackingError },
      ]
      const terminal = next.filter((entry: TrackedMessageBatch) => isTerminal(entry.status) || entry.trackingError)
      const retained = new Set(terminal.slice(-RETAINED_TERMINAL_BATCHES).map(
        (entry: TrackedMessageBatch) => entry.status.batch_id,
      ))
      return next.filter((entry: TrackedMessageBatch) =>
        (!isTerminal(entry.status) && !entry.trackingError) || retained.has(entry.status.batch_id),
      )
    })
  }, [])

  const monitor = useCallback(async (entry: BatchMonitor, refreshFirst = false): Promise<MessageBatchStatus> => {
    const controller = new AbortController()
    const id = entry.status.batch_id
    controllers.current.set(id, controller)
    try {
      if (refreshFirst) {
        entry.status = await attacksApi.getMessageBatch(entry.status.attack_result_id, id, controller.signal)
        if (controller.signal.aborted) throw new DOMException('Progress tracking stopped', 'AbortError')
      }
      record(entry.status)
      await entry.onProgress(entry.status)
      while (!isTerminal(entry.status)) {
        if (controller.signal.aborted) throw new DOMException('Progress tracking stopped', 'AbortError')
        entry.status = await attacksApi.getMessageBatch(
          entry.status.attack_result_id,
          id,
          controller.signal,
        )
        if (controller.signal.aborted) throw new DOMException('Progress tracking stopped', 'AbortError')
        record(entry.status)
        await entry.onProgress(entry.status)
        if (!isTerminal(entry.status)) await pausePolling(controller.signal)
      }
      return entry.status
    } catch (error: unknown) {
      if (controller.signal.aborted) throw error
      const trackingError = new MessageBatchTrackingError(toApiError(error).detail)
      record(entry.status, trackingError.message)
      throw trackingError
    } finally {
      controllers.current.delete(id)
      if (isTerminal(entry.status)) monitors.current.delete(id)
    }
  }, [record])

  const executeBatch = useCallback(async (
    attackResultId: string,
    request: MessageBatchInput,
    onProgress: ProgressCallback,
  ): Promise<MessageBatchStatus> => {
    const status = await attacksApi.startMessageBatch(attackResultId, request)
    if (!mounted.current) throw new DOMException('Progress tracking stopped', 'AbortError')
    const entry = { status, onProgress }
    monitors.current.set(status.batch_id, entry)
    return monitor(entry)
  }, [monitor])

  const retryTracking = useCallback(async (batchId: string): Promise<void> => {
    const entry = monitors.current.get(batchId)
    if (!entry || controllers.current.has(batchId)) return
    try {
      await monitor(entry, true)
    } catch (error: unknown) {
      // monitor records a recoverable read error; retry never submits another send.
      if (!(error instanceof MessageBatchTrackingError) && mounted.current) {
        record(entry.status, toApiError(error).detail)
      }
    }
  }, [monitor, record])

  const dismissBatch = useCallback((batchId: string): void => {
    const controller = controllers.current.get(batchId)
    if (controller) return
    monitors.current.delete(batchId)
    setBatches((previous: TrackedMessageBatch[]) =>
      previous.filter((entry: TrackedMessageBatch) => entry.status.batch_id !== batchId),
    )
  }, [])

  return { batches, executeBatch, retryTracking, dismissBatch }
}
