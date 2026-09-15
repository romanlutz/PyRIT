import type { ConversationTreeEndpoint, ConversationTreeNode, ConversationTreePiecePreview } from '@/types'

export const TREE_NODE_WIDTH = 320
export const TREE_NODE_HEIGHT = 336
export const TREE_COLUMN_GAP = 48
export const TREE_ROW_GAP = 64
export const COMPACT_PIECE_COUNT = 3
const AUTO_COLLAPSE_MESSAGES = 12

export interface TreeIndex {
  readonly nodes: ReadonlyMap<string, ConversationTreeNode>
  readonly roots: string[]
  readonly children: ReadonlyMap<string, string[]>
  readonly endpoints: ReadonlyMap<string, ConversationTreeEndpoint[]>
  readonly messageCounts: ReadonlyMap<string, number>
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
  readonly height?: number
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
  const messageCounts = new Map<string, number>()
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
      let messages = 0
      let count = endpoints.get(current.id)?.length ?? 0
      for (const child of descendants) {
        messages += 1 + (messageCounts.get(child) ?? 0)
        count += conversationCounts.get(child) ?? 0
      }
      messageCounts.set(current.id, messages)
      conversationCounts.set(current.id, count)
      end.set(current.id, ordinal)
    } else if (!start.has(current.id)) {
      start.set(current.id, ordinal++)
      stack.push({ id: current.id, exit: true })
      for (const id of [...(children.get(current.id) ?? [])].reverse()) stack.push({ id, exit: false })
    }
  }
  return { nodes, roots, children, endpoints, messageCounts, conversationCounts, start, end }
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

export function collapsedBranches(
  index: TreeIndex,
  activePath: ReadonlySet<string>,
  overrides: ReadonlyMap<string, boolean>,
): Set<string> {
  const collapsed = new Set<string>()
  for (const node of index.nodes.values()) {
    const override = overrides.get(node.node_id)
    const boundary = !node.parent_node_id || activePath.has(node.parent_node_id)
    if (override === true || (override === undefined && boundary && !activePath.has(node.node_id)
      && (index.messageCounts.get(node.node_id) ?? 0) >= AUTO_COLLAPSE_MESSAGES)) {
      collapsed.add(node.node_id)
    }
  }
  return collapsed
}

export function expandedTree(index: TreeIndex, collapsed: ReadonlySet<string>): ConversationTreeNode[] {
  const expanded: ConversationTreeNode[] = []
  const stack = [...index.roots].reverse()
  while (stack.length > 0) {
    const id = stack.pop()
    if (!id) break
    const node = index.nodes.get(id)
    if (!node) continue
    expanded.push(node)
    if (!collapsed.has(id)) {
      for (const child of [...(index.children.get(id) ?? [])].reverse()) stack.push(child)
    }
  }
  return expanded
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
): Map<string, TreePosition> {
  const positions = new Map<string, TreePosition>()
  const rightEdges = new Map<number, number>()
  const rowHeight = TREE_NODE_HEIGHT + TREE_ROW_GAP
  const occupy = (position: TreePosition): void => {
    const firstRow = Math.floor(position.y / rowHeight)
    const lastRow = Math.floor((position.y + TREE_NODE_HEIGHT) / rowHeight)
    for (let row = firstRow; row <= lastRow; row += 1) {
      rightEdges.set(row, Math.max(rightEdges.get(row) ?? -TREE_COLUMN_GAP, position.x + TREE_NODE_WIDTH))
    }
  }
  for (const node of nodes) {
    const position = previous.get(node.id)
    if (position) {
      positions.set(node.id, position)
      occupy(position)
    }
  }
  for (const node of nodes) {
    if (positions.has(node.id)) continue
    const parent = node.parentId ? positions.get(node.parentId) : undefined
    const y = parent ? parent.y + rowHeight : 0
    let x = parent?.x ?? 0
    for (let row = Math.floor(y / rowHeight); row <= Math.floor((y + TREE_NODE_HEIGHT) / rowHeight); row += 1) {
      x = Math.max(x, (rightEdges.get(row) ?? -TREE_COLUMN_GAP) + TREE_COLUMN_GAP)
    }
    const position = { x, y }
    positions.set(node.id, position)
    occupy(position)
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
