import { attacksApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type {
  ConversationTreeEndpoint,
  ConversationTreeNode,
  ConversationTreePage,
  ConversationTreePiecePreview,
  ConversationTreePreview,
  TreeMessageReference,
  TreePreviewLevel,
} from '@/types'

const TOPOLOGY_PAGE_SIZE = 100
export const PREVIEW_BATCH_SIZE = 64
const PAGE_YIELD_MS = 16

type PreviewPriority = 'interaction' | 'visible'

export interface PreviewResult {
  readonly preview?: ConversationTreePreview
  readonly loading: boolean
  readonly error?: string
}

export type NodePreviews = Partial<Record<TreePreviewLevel, PreviewResult>>

export function previewPieces(previews?: NodePreviews): ConversationTreePiecePreview[] {
  const text = previews?.text?.preview?.pieces
  const thumbnails = previews?.thumbnail?.preview?.pieces
  return (text ?? thumbnails ?? []).map((piece: ConversationTreePiecePreview, index: number) => ({
    ...piece,
    thumbnail_url: thumbnails?.[index]?.thumbnail_url ?? piece.thumbnail_url,
  }))
}

export interface TreeReadSnapshot {
  readonly nodes: ReadonlyMap<string, ConversationTreeNode>
  readonly conversations: ReadonlyMap<string, ConversationTreeEndpoint>
  readonly previews: ReadonlyMap<string, NodePreviews>
  readonly mainConversationId: string | null
  readonly revision: string | null
  readonly processed: number
  readonly total: number | null
  readonly complete: boolean
  readonly loading: boolean
  readonly refreshing: boolean
  readonly error?: string
  readonly needsRefresh: boolean
}

interface PreviewTask {
  readonly key: string
  readonly node: ConversationTreeNode
  readonly level: TreePreviewLevel
  priority: PreviewPriority
  running: boolean
}

interface PreviewBatch {
  readonly tasks: PreviewTask[]
  readonly priority: PreviewPriority
}

function messageKey(message: TreeMessageReference): string {
  return JSON.stringify([message.conversation_id, message.sequence])
}

function taskKey(node: ConversationTreeNode, level: TreePreviewLevel): string {
  return JSON.stringify([node.preview_key, level])
}

function wasAborted(error: unknown, signal: AbortSignal): boolean {
  return signal.aborted || (error instanceof Error && (error.name === 'AbortError' || error.name === 'CanceledError'))
}

/** Three independent read slots: topology, explicit interaction, and passive previews. */
export class TreeReadCoordinator {
  private snapshot: TreeReadSnapshot = {
    nodes: new Map(),
    conversations: new Map(),
    previews: new Map(),
    mainConversationId: null,
    revision: null,
    processed: 0,
    total: null,
    complete: false,
    loading: false,
    refreshing: false,
    needsRefresh: false,
  }

  private readonly listeners = new Set<() => void>()
  private readonly tasks = new Map<string, PreviewTask>()
  private readonly previewBatches = new Map<AbortController, PreviewBatch>()
  private visibleNodes: ConversationTreeNode[] = []
  private enabled = false
  private epoch = 0
  private started = false
  private cursor: string | null = null
  private revision: string | null = null
  private readonly seenCursors = new Set<string>()
  private replacing = true
  private replacementNodes = new Map<string, ConversationTreeNode>()
  private replacementConversations = new Map<string, ConversationTreeEndpoint>()
  private topologyController: AbortController | null = null
  private topologyTimer: ReturnType<typeof setTimeout> | null = null
  private interactionBusy = false
  private visibleBusy = false
  private activeConversationId: string | null
  private refreshKey: number
  private generationRefreshKey: number

  constructor(
    private readonly attackResultId: string,
    activeConversationId: string | null,
    refreshKey: number,
  ) {
    this.activeConversationId = activeConversationId
    this.refreshKey = refreshKey
    this.generationRefreshKey = refreshKey
  }

  getSnapshot = (): TreeReadSnapshot => this.snapshot

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  private publish(update: Partial<TreeReadSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...update }
    for (const listener of this.listeners) listener()
  }

  setActiveConversation(id: string | null): void {
    this.activeConversationId = id
  }

  setActive(active: boolean): void {
    if (active === this.enabled) return
    this.enabled = active
    if (!active) {
      this.cancelReads()
      this.publish({ loading: false })
      return
    }
    if (!this.started || (this.snapshot.complete && this.refreshKey !== this.generationRefreshKey)) {
      this.startGeneration()
    } else {
      this.loadPage()
    }
    this.setVisibleNodes(this.visibleNodes)
  }

  refresh(key: number): void {
    this.refreshKey = key
    // Finish a bounded snapshot before incorporating writes arriving during discovery.
    if (this.enabled && this.snapshot.complete && key !== this.generationRefreshKey) this.startGeneration()
  }

  retryTopology = (): void => {
    if (this.snapshot.needsRefresh) {
      this.startGeneration()
    } else {
      this.publish({ error: undefined })
      this.loadPage()
    }
  }

  private cancelReads(): void {
    this.epoch += 1
    this.topologyController?.abort()
    this.topologyController = null
    if (this.topologyTimer !== null) clearTimeout(this.topologyTimer)
    this.topologyTimer = null
    for (const controller of this.previewBatches.keys()) controller.abort()
    this.previewBatches.clear()
    this.tasks.clear()
    this.interactionBusy = false
    this.visibleBusy = false
    const previews = new Map(this.snapshot.previews)
    for (const [key, entry] of previews) {
      const next = { ...entry }
      for (const level of ['text', 'thumbnail', 'full'] as const) {
        const result = next[level]
        if (result?.loading) next[level] = { ...result, loading: false }
      }
      previews.set(key, next)
    }
    this.publish({ previews })
  }

  private startGeneration(): void {
    if (!this.enabled) return
    this.cancelReads()
    this.started = true
    this.generationRefreshKey = this.refreshKey
    this.cursor = null
    this.revision = null
    this.seenCursors.clear()
    this.replacing = true
    this.replacementNodes = new Map()
    this.replacementConversations = new Map()
    this.publish({
      complete: false,
      loading: true,
      refreshing: this.snapshot.revision !== null,
      error: undefined,
      needsRefresh: false,
    })
    this.loadPage()
    this.setVisibleNodes(this.visibleNodes)
  }

  private loadPage(): void {
    if (!this.enabled || this.topologyController || this.snapshot.complete || this.snapshot.error) return
    const controller = new AbortController()
    const epoch = this.epoch
    const cursor = this.cursor
    this.topologyController = controller
    this.publish({ loading: true })
    void attacksApi.getConversationTree(this.attackResultId, {
      limit: TOPOLOGY_PAGE_SIZE,
      ...(cursor ? { cursor } : {}),
      ...(!cursor && this.activeConversationId ? { prioritize_conversation_id: this.activeConversationId } : {}),
    }, controller.signal).then((page: ConversationTreePage) => {
      if (epoch !== this.epoch || controller.signal.aborted) return
      if (page.attack_result_id !== this.attackResultId) throw new Error('The tree response belongs to a different attack.')
      if (this.revision !== null && this.revision !== page.revision) {
        this.publish({ error: 'The tree snapshot changed. Refresh to continue loading branches.', needsRefresh: true })
        return
      }
      if (!page.complete && (!page.next_cursor || this.seenCursors.has(page.next_cursor))) {
        this.publish({ error: 'Tree pagination did not advance. Refresh to continue loading branches.', needsRefresh: true })
        return
      }
      this.revision = page.revision
      this.cursor = page.next_cursor
      if (page.next_cursor) this.seenCursors.add(page.next_cursor)
      const nodes = new Map(this.replacing ? this.replacementNodes : this.snapshot.nodes)
      const conversations = new Map(this.replacing ? this.replacementConversations : this.snapshot.conversations)
      for (const node of page.nodes) nodes.set(node.node_id, node)
      for (const endpoint of page.conversations) conversations.set(endpoint.conversation_id, endpoint)
      this.replacementNodes = nodes
      this.replacementConversations = conversations
      if (nodes.size > 0 || conversations.size > 0 || page.complete) {
        this.replacing = false
        this.publish({
          nodes,
          conversations,
          revision: page.revision,
          mainConversationId: page.main_conversation_id,
          refreshing: false,
          complete: page.complete,
          processed: page.processed_conversations,
          total: page.total_conversations,
        })
      } else {
        this.publish({ processed: page.processed_conversations, total: page.total_conversations })
      }
    }).catch((error: unknown) => {
      if (epoch !== this.epoch || wasAborted(error, controller.signal)) return
      const apiError = toApiError(error)
      this.publish({ error: apiError.detail, needsRefresh: apiError.status === 409 || apiError.status === 410 })
    }).finally(() => {
      if (epoch !== this.epoch || controller.signal.aborted) return
      this.topologyController = null
      this.publish({ loading: !this.snapshot.error && !this.snapshot.complete })
      if (this.snapshot.error) return
      this.topologyTimer = setTimeout(() => {
        this.topologyTimer = null
        if (this.snapshot.complete) {
          if (this.refreshKey !== this.generationRefreshKey) this.startGeneration()
        } else {
          this.loadPage()
        }
      }, PAGE_YIELD_MS)
    })
  }

  setVisibleNodes(nodes: ConversationTreeNode[]): void {
    this.visibleNodes = nodes
    const visibleKeys = new Set(nodes.map((node: ConversationTreeNode) => node.preview_key))
    for (const [key, task] of this.tasks) {
      if (!task.running && task.priority === 'visible' && !visibleKeys.has(task.node.preview_key)) this.tasks.delete(key)
    }
    if (!this.enabled) return
    for (const node of nodes) this.enqueue(node, 'text', 'visible', false)
    for (const node of nodes) {
      if (node.piece_types.some((type: string) => type === 'image_path' || type === 'video_path')) {
        this.enqueue(node, 'thumbnail', 'visible', false)
      }
    }
    this.pumpPreviews()
  }

  requestPreviews(nodes: ConversationTreeNode[], level: TreePreviewLevel = 'text', retry = false): void {
    if (!this.enabled) return
    for (const node of nodes) this.enqueue(node, level, 'interaction', retry)
    this.pumpPreviews()
  }

  cancelFullPreview(node: ConversationTreeNode): void {
    const key = taskKey(node, 'full')
    const task = this.tasks.get(key)
    if (!task) return
    this.tasks.delete(key)
    if (!task.running) return
    for (const [controller, batch] of this.previewBatches) {
      if (!batch.tasks.includes(task)) continue
      controller.abort()
      this.previewBatches.delete(controller)
      const previews = new Map(this.snapshot.previews)
      for (const interrupted of batch.tasks) {
        interrupted.running = false
        const entry = previews.get(interrupted.node.preview_key)
        previews.set(interrupted.node.preview_key, { ...entry, full: { ...entry?.full, loading: false } })
      }
      if (batch.priority === 'interaction') this.interactionBusy = false
      else this.visibleBusy = false
      this.publish({ previews })
      this.pumpPreviews()
      return
    }
  }

  private enqueue(node: ConversationTreeNode, level: TreePreviewLevel, priority: PreviewPriority, retry: boolean): void {
    const key = taskKey(node, level)
    const existing = this.tasks.get(key)
    if (existing) {
      if (priority === 'interaction') existing.priority = priority
      return
    }
    const result = this.snapshot.previews.get(node.preview_key)?.[level]
    if (!retry && (result?.preview || result?.error)) return
    this.tasks.set(key, { key, node, level, priority, running: false })
  }

  private pumpPreviews(): void {
    if (!this.enabled) return
    if (!this.interactionBusy) this.startPreviewBatch('interaction')
    if (!this.visibleBusy) this.startPreviewBatch('visible')
  }

  private startPreviewBatch(priority: PreviewPriority): void {
    const pending = [...this.tasks.values()].filter((task: PreviewTask) => !task.running && task.priority === priority)
    const first = pending.find((task: PreviewTask) => task.level === (priority === 'interaction' ? 'full' : 'text')) ?? pending[0]
    if (!first) return
    const batch = pending.filter((task: PreviewTask) => task.level === first.level).slice(0, PREVIEW_BATCH_SIZE)
    const controller = new AbortController()
    const epoch = this.epoch
    this.previewBatches.set(controller, { tasks: batch, priority })
    if (priority === 'interaction') this.interactionBusy = true
    else this.visibleBusy = true
    const previews = new Map(this.snapshot.previews)
    for (const task of batch) {
      task.running = true
      const entry = previews.get(task.node.preview_key)
      previews.set(task.node.preview_key, { ...entry, [task.level]: { ...entry?.[task.level], loading: true, error: undefined } })
    }
    this.publish({ previews })
    void attacksApi.getTreePreviews(
      this.attackResultId,
      batch.map((task: PreviewTask) => task.node.message),
      first.level,
      controller.signal,
    ).then((response) => {
      if (epoch !== this.epoch || controller.signal.aborted) return
      const byMessage = new Map(response.previews.map((preview: ConversationTreePreview) => [messageKey(preview.message), preview]))
      const next = new Map(this.snapshot.previews)
      for (const task of batch) {
        const preview = byMessage.get(messageKey(task.node.message))
        next.set(task.node.preview_key, {
          ...next.get(task.node.preview_key),
          [task.level]: preview
            ? { preview, loading: false }
            : { loading: false, error: 'This message preview is unavailable. Retry to load it again.' },
        })
      }
      this.publish({ previews: next })
    }).catch((error: unknown) => {
      if (epoch !== this.epoch || wasAborted(error, controller.signal)) return
      const detail = toApiError(error).detail
      const next = new Map(this.snapshot.previews)
      for (const task of batch) {
        const entry = next.get(task.node.preview_key)
        next.set(task.node.preview_key, { ...entry, [task.level]: { ...entry?.[task.level], loading: false, error: detail } })
      }
      this.publish({ previews: next })
    }).finally(() => {
      if (epoch !== this.epoch || controller.signal.aborted) return
      this.previewBatches.delete(controller)
      for (const task of batch) this.tasks.delete(task.key)
      if (priority === 'interaction') this.interactionBusy = false
      else this.visibleBusy = false
      this.pumpPreviews()
    })
  }
}
