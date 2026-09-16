import type { ConversationTreeEndpoint, ConversationTreeNode, ConversationTreePiecePreview } from '@/types'

export const TREE_NODE_WIDTH = 320
export const TREE_NODE_HEIGHT = 336
export const TREE_COLUMN_GAP = 48
export const TREE_ROW_GAP = 64
export const COMPACT_PIECE_COUNT = 3

export interface TreeIndex {
  readonly nodes: ReadonlyMap<string, ConversationTreeNode>
  readonly roots: string[]
  readonly children: ReadonlyMap<string, string[]>
  readonly endpoints: ReadonlyMap<string, ConversationTreeEndpoint[]>
  readonly conversationCounts: ReadonlyMap<string, number>
  readonly start: ReadonlyMap<string, number>
  readonly end: ReadonlyMap<string, number>
}

export interface TreePosition {
  readonly x: number
  readonly y: number
}

export interface LayoutNode {
  readonly id: string
  readonly parentId: string | null
  readonly sequence: number
  readonly height?: number
}

export interface TreeSequenceRow {
  readonly sequence: number
  readonly top: number
  readonly height: number
}

export function treeSequenceRows(nodes: LayoutNode[]): TreeSequenceRow[] {
  const heights = new Map<number, number>()
  for (const node of nodes) {
    heights.set(node.sequence, Math.max(heights.get(node.sequence) ?? 0, node.height ?? TREE_NODE_HEIGHT))
  }
  let top = 0
  return [...heights].sort(([a]: [number, number], [b]: [number, number]) => a - b)
    .map(([sequence, height]: [number, number]): TreeSequenceRow => {
      const row = { sequence, top, height }
      top += height + TREE_ROW_GAP
      return row
    })
}

export function treeNodeHeight(pieceCount: number): number {
  return 160 + 48 * Math.min(Math.max(pieceCount, 1), COMPACT_PIECE_COUNT)
}

export function indexTree(
  nodes: ReadonlyMap<string, ConversationTreeNode>,
  conversations: ReadonlyMap<string, ConversationTreeEndpoint>,
): TreeIndex {
  const children = new Map<string, string[]>()
  const endpoints = new Map<string, ConversationTreeEndpoint[]>()
  const roots: string[] = []
  for (const node of nodes.values()) {
    if (node.parent_node_id && nodes.has(node.parent_node_id)) {
      const siblings = children.get(node.parent_node_id) ?? []
      siblings.push(node.node_id)
      children.set(node.parent_node_id, siblings)
    } else {
      roots.push(node.node_id)
    }
  }
  roots.sort()
  for (const siblings of children.values()) siblings.sort()
  for (const endpoint of conversations.values()) {
    if (endpoint.node_id !== null) {
      const sameNode = endpoints.get(endpoint.node_id) ?? []
      sameNode.push(endpoint)
      endpoints.set(endpoint.node_id, sameNode)
    }
  }
  const conversationCounts = new Map<string, number>()
  const start = new Map<string, number>()
  const end = new Map<string, number>()
  const stack = [...roots].reverse().map((id: string) => ({ id, exit: false }))
  let ordinal = 0
  while (stack.length > 0) {
    const current = stack.pop()
    if (!current) break
    if (current.exit) {
      const descendants = children.get(current.id) ?? []
      let count = endpoints.get(current.id)?.length ?? 0
      for (const child of descendants) {
        count += conversationCounts.get(child) ?? 0
      }
      conversationCounts.set(current.id, count)
      end.set(current.id, ordinal)
    } else if (!start.has(current.id)) {
      start.set(current.id, ordinal++)
      stack.push({ id: current.id, exit: true })
      for (const id of [...(children.get(current.id) ?? [])].reverse()) stack.push({ id, exit: false })
    }
  }
  return { nodes, roots, children, endpoints, conversationCounts, start, end }
}

export function conversationPath(
  index: TreeIndex,
  conversations: ReadonlyMap<string, ConversationTreeEndpoint>,
  conversationId: string | null,
): Set<string> {
  const path = new Set<string>()
  if (!conversationId) return path
  let id = conversations.get(conversationId)?.node_id
  if (id === undefined) {
    let last: ConversationTreeNode | undefined
    for (const node of index.nodes.values()) {
      if (node.message.conversation_id === conversationId && (!last || node.message.sequence > last.message.sequence)) last = node
    }
    id = last?.node_id
  }
  while (id && !path.has(id)) {
    path.add(id)
    id = index.nodes.get(id)?.parent_node_id
  }
  return path
}

export function orderedTreeNodes(index: TreeIndex): ConversationTreeNode[] {
  const ordered: ConversationTreeNode[] = []
  const stack = [...index.roots].reverse()
  while (stack.length > 0) {
    const id = stack.pop()
    if (!id) break
    const node = index.nodes.get(id)
    if (!node) continue
    ordered.push(node)
    for (const child of [...(index.children.get(id) ?? [])].reverse()) stack.push(child)
  }
  return ordered
}

export function isInBranch(index: TreeIndex, nodeId: string, ancestorId: string): boolean {
  const start = index.start.get(nodeId)
  const ancestorStart = index.start.get(ancestorId)
  const ancestorEnd = index.end.get(ancestorId)
  return start !== undefined && ancestorStart !== undefined && ancestorEnd !== undefined
    && start >= ancestorStart && start < ancestorEnd
}

/** Cheap placement keeps new topology usable while the worker calculates its layout. */
export function reservePositions(
  nodes: LayoutNode[],
  previous: ReadonlyMap<string, TreePosition>,
  anchorId: string | null = null,
): Map<string, TreePosition> {
  const rows = new Map(treeSequenceRows(nodes).map((row: TreeSequenceRow) => [row.sequence, row]))
  const positions = new Map<string, TreePosition>()
  const rightEdges = new Map<number, number>()
  const rowY = (node: LayoutNode): number => {
    const row = rows.get(node.sequence)
    if (!row) throw new Error(`No sequence row for message ${node.id}.`)
    return row.top + (row.height - (node.height ?? TREE_NODE_HEIGHT)) / 2
  }
  const anchor = nodes.find((node: LayoutNode) => node.id === anchorId && previous.has(node.id))
    ?? nodes.find((node: LayoutNode) => previous.has(node.id))
  const anchorPosition = anchor ? previous.get(anchor.id) : undefined
  const offsetY = anchor && anchorPosition ? anchorPosition.y - rowY(anchor) : 0
  let previousRight = -TREE_COLUMN_GAP
  for (const node of nodes) {
    const position = previous.get(node.id)
    if (position) {
      positions.set(node.id, { x: position.x, y: rowY(node) + offsetY })
      const right = position.x + TREE_NODE_WIDTH
      previousRight = Math.max(previousRight, right)
      rightEdges.set(node.sequence, Math.max(rightEdges.get(node.sequence) ?? -TREE_COLUMN_GAP, right))
    }
  }
  for (const node of nodes) {
    if (positions.has(node.id)) continue
    const parent = node.parentId ? positions.get(node.parentId) : undefined
    // New cards stay outside accepted routes until the worker can route their connections.
    const x = Math.max(parent?.x ?? 0, previousRight + TREE_COLUMN_GAP,
      (rightEdges.get(node.sequence) ?? -TREE_COLUMN_GAP) + TREE_COLUMN_GAP)
    positions.set(node.id, { x, y: rowY(node) + offsetY })
    rightEdges.set(node.sequence, x + TREE_NODE_WIDTH)
  }
  return positions
}

export function mediaType(dataType: string): 'image' | 'audio' | 'video' | 'file' {
  if (dataType === 'image_path') return 'image'
  if (dataType === 'audio_path') return 'audio'
  if (dataType === 'video_path') return 'video'
  return 'file'
}

export function pieceLabel(dataType: string): string {
  if (dataType === 'text') return 'Text'
  if (dataType === 'image_path') return 'Image'
  if (dataType === 'audio_path') return 'Audio'
  if (dataType === 'video_path') return 'Video'
  return dataType.split('_').join(' ')
}

export function pieceHasError(piece: ConversationTreePiecePreview): boolean {
  return Boolean(piece.response_error && piece.response_error !== 'none')
}

export function roleLabel(role: string): string {
  return role.charAt(0).toUpperCase() + role.slice(1).split('_').join(' ')
}
