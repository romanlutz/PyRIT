import { useCallback, useEffect, useRef, useState } from 'react'

import { attacksApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type { MessageSendInput, MessageSendStatus, TrackedMessageSend } from '@/types'

const RETAINED_TERMINAL_SENDS = 5

type ProgressCallback = (status: MessageSendStatus) => Promise<void>

interface SendMonitor {
  status: MessageSendStatus
  onProgress: ProgressCallback
  onTrackingStopped?: () => void
}

export class MessageSendTrackingError extends Error {
  constructor(detail: string) {
    super(`Progress is unavailable. Sends may still complete; do not resend automatically. ${detail}`)
    this.name = 'MessageSendTrackingError'
  }
}

function isTerminal(status: MessageSendStatus): boolean {
  return status.state === 'completed' || status.state === 'failed'
}

/** Tracks accepted sends independently of the conversation currently being viewed. */
export function useMessageSend() {
  const [sends, setSends] = useState<TrackedMessageSend[]>([])
  const monitors = useRef(new Map<string, SendMonitor>())
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
    const retained = new Set(sends.map((send: TrackedMessageSend) => send.status.send_id))
    for (const id of monitors.current.keys()) {
      if (!retained.has(id) && !controllers.current.has(id)) monitors.current.delete(id)
    }
  }, [sends])

  const record = useCallback((status: MessageSendStatus, trackingError: string | null = null): void => {
    if (!mounted.current) return
    setSends((previous: TrackedMessageSend[]) => {
      const next = [
        ...previous.filter((entry: TrackedMessageSend) => entry.status.send_id !== status.send_id),
        { status, trackingError },
      ]
      const terminal = next.filter((entry: TrackedMessageSend) => isTerminal(entry.status) || entry.trackingError)
      const retained = new Set(terminal.slice(-RETAINED_TERMINAL_SENDS).map(
        (entry: TrackedMessageSend) => entry.status.send_id,
      ))
      return next.filter((entry: TrackedMessageSend) =>
        (!isTerminal(entry.status) && !entry.trackingError) || retained.has(entry.status.send_id),
      )
    })
  }, [])

  const monitor = useCallback(async (entry: SendMonitor, refreshFirst = false): Promise<MessageSendStatus> => {
    const controller = new AbortController()
    const id = entry.status.send_id
    controllers.current.set(id, controller)
    try {
      if (refreshFirst) {
        entry.status = await attacksApi.getMessageSend(entry.status.attack_result_id, id, controller.signal)
        if (controller.signal.aborted) throw new DOMException('Progress tracking stopped', 'AbortError')
      }
      record(entry.status)
      await entry.onProgress(entry.status)
      while (!isTerminal(entry.status)) {
        if (controller.signal.aborted) throw new DOMException('Progress tracking stopped', 'AbortError')
        entry.status = await attacksApi.getMessageSend(
          entry.status.attack_result_id,
          id,
          controller.signal,
        )
        if (controller.signal.aborted) throw new DOMException('Progress tracking stopped', 'AbortError')
        record(entry.status)
        await entry.onProgress(entry.status)
      }
      monitors.current.delete(id)
      return entry.status
    } catch (error: unknown) {
      if (controller.signal.aborted) throw error
      const apiError = toApiError(error)
      const trackingError = new MessageSendTrackingError(
        apiError.isTimeout ? 'The progress read timed out.' : apiError.detail,
      )
      record(entry.status, trackingError.message)
      throw trackingError
    } finally {
      controllers.current.delete(id)
      entry.onTrackingStopped?.()
    }
  }, [record])

  const executeSend = useCallback(async (
    attackResultId: string,
    request: MessageSendInput,
    onProgress: ProgressCallback,
    onTrackingStopped?: () => void,
  ): Promise<MessageSendStatus> => {
    const status = await attacksApi.startMessageSend(attackResultId, request)
    if (!mounted.current) throw new DOMException('Progress tracking stopped', 'AbortError')
    const entry = { status, onProgress, onTrackingStopped }
    monitors.current.set(status.send_id, entry)
    return monitor(entry)
  }, [monitor])

  const retryTracking = useCallback(async (sendId: string): Promise<void> => {
    const entry = monitors.current.get(sendId)
    if (!entry || controllers.current.has(sendId)) return
    try {
      await monitor(entry, true)
    } catch (error: unknown) {
      // monitor records a recoverable read error; retry never submits another send.
      if (!(error instanceof MessageSendTrackingError) && mounted.current) {
        record(entry.status, toApiError(error).detail)
      }
    }
  }, [monitor, record])

  const dismissSend = useCallback((sendId: string): void => {
    const controller = controllers.current.get(sendId)
    if (controller) return
    monitors.current.delete(sendId)
    setSends((previous: TrackedMessageSend[]) =>
      previous.filter((entry: TrackedMessageSend) => entry.status.send_id !== sendId),
    )
  }, [])

  return { sends, executeSend, retryTracking, dismissSend }
}
