import { reservePositions, type LayoutNode, type TreePosition } from './treeGraph'
import type { TreeLayoutReply, TreeLayoutWorkerPort } from './treeLayout.types'

const LAYOUT_COALESCE_MS = 32

interface LayoutSnapshot {
  readonly positions: ReadonlyMap<string, TreePosition>
  readonly busy: boolean
  readonly ready: boolean
  readonly error?: string
}

export class TreeLayoutCoordinator {
  private snapshot: LayoutSnapshot = { positions: new Map(), busy: false, ready: false }
  private readonly listeners = new Set<() => void>()
  private worker: TreeLayoutWorkerPort | null = null
  private active = false
  private nodes: LayoutNode[] = []
  private signature = ''
  private version = 0
  private acceptedVersion = -1
  private runningVersion: number | null = null
  private anchorId: string | null = null
  private timer: ReturnType<typeof setTimeout> | null = null

  constructor(private readonly createWorker: () => TreeLayoutWorkerPort) {}

  getSnapshot = (): LayoutSnapshot => this.snapshot

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  private publish(update: Partial<LayoutSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...update }
    for (const listener of this.listeners) listener()
  }

  setActive(active: boolean): void {
    if (this.active === active) return
    this.active = active
    if (!active) {
      if (this.timer !== null) clearTimeout(this.timer)
      this.timer = null
      this.worker?.terminate()
      this.worker = null
      this.runningVersion = null
      this.publish({ busy: false })
    } else {
      this.schedule()
    }
  }

  setGraph(nodes: LayoutNode[], anchorId: string | null): void {
    this.anchorId = anchorId
    const signature = JSON.stringify(nodes)
    if (signature === this.signature) return
    this.signature = signature
    this.nodes = nodes
    this.version += 1
    this.publish({ positions: reservePositions(nodes, this.snapshot.positions), error: undefined })
    this.schedule()
  }

  retry = (): void => {
    this.publish({ error: undefined })
    this.schedule()
  }

  private schedule(): void {
    if (!this.active || this.timer !== null || this.runningVersion !== null
      || this.acceptedVersion === this.version || this.nodes.length === 0 || this.snapshot.error) return
    this.publish({ busy: true })
    // Do not keep resetting this timer as pages arrive: continuous discovery must not starve layout.
    this.timer = setTimeout(() => {
      this.timer = null
      this.dispatch()
    }, LAYOUT_COALESCE_MS)
  }

  private dispatch(): void {
    if (!this.active) return
    try {
      if (!this.worker) {
        const worker = this.createWorker()
        this.worker = worker
        worker.onmessage = (event: MessageEvent<TreeLayoutReply>) => {
          if (!this.active || worker !== this.worker || event.data.requestId !== this.runningVersion) return
          this.runningVersion = null
          if (event.data.requestId === this.version) {
            if ('error' in event.data) {
              this.publish({ busy: false, error: event.data.error })
              return
            }
            const next = new Map(event.data.positions)
            const previousAnchor = this.anchorId ? this.snapshot.positions.get(this.anchorId) : undefined
            const nextAnchor = this.anchorId ? next.get(this.anchorId) : undefined
            if (previousAnchor && nextAnchor) {
              const delta = { x: previousAnchor.x - nextAnchor.x, y: previousAnchor.y - nextAnchor.y }
              for (const [id, position] of next) next.set(id, { x: position.x + delta.x, y: position.y + delta.y })
            }
            this.acceptedVersion = this.version
            this.publish({ positions: next, busy: false, ready: true })
          }
          this.schedule()
        }
        worker.onerror = (event: ErrorEvent) => {
          if (worker !== this.worker) return
          this.fail(event.message || 'The tree layout worker could not start.')
        }
      }
      this.runningVersion = this.version
      this.worker.postMessage({ requestId: this.version, nodes: this.nodes })
    } catch (error: unknown) {
      this.fail(error instanceof Error ? error.message : 'The tree layout worker could not start.')
    }
  }

  private fail(error: string): void {
    this.worker?.terminate()
    this.worker = null
    this.runningVersion = null
    this.publish({ busy: false, error })
  }
}
