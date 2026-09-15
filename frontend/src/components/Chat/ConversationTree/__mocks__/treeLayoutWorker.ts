import { layoutTree } from '../treeLayout'
import type { TreeLayoutReply, TreeLayoutRequest, TreeLayoutWorkerPort } from '../treeLayout.types'
import { installStructuredClone } from './structuredClone'

installStructuredClone()

export default class TestLayoutWorker implements TreeLayoutWorkerPort {
  onmessage: ((event: MessageEvent<TreeLayoutReply>) => void) | null = null
  onerror: ((event: ErrorEvent) => void) | null = null
  private stopped = false

  postMessage(request: TreeLayoutRequest): void {
    queueMicrotask(() => {
      if (this.stopped) return
      try {
        this.onmessage?.(new MessageEvent<TreeLayoutReply>('message', {
          data: { requestId: request.requestId, positions: layoutTree(request.nodes) },
        }))
      } catch (error: unknown) {
        this.onerror?.(new ErrorEvent('error', { message: error instanceof Error ? error.message : 'Layout failed' }))
      }
    })
  }

  terminate(): void {
    this.stopped = true
  }
}
