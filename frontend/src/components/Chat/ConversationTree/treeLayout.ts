import dagre from '@dagrejs/dagre'

import {
  TREE_COLUMN_GAP,
  TREE_NODE_HEIGHT,
  TREE_NODE_WIDTH,
  TREE_ROW_GAP,
  treeSequenceRows,
  type LayoutNode,
  type TreePosition,
  type TreeSequenceRow,
} from './treeGraph'
import { separateTreeRoutes } from './treeEdgeRouting'
import type { TreeLayoutResult } from './treeLayout.types'

type LayoutGraph = InstanceType<typeof dagre.graphlib.Graph>

interface RankedNode {
  readonly rank: number
  order?: number
}

function isRankedNode(value: unknown): value is RankedNode {
  return typeof value === 'object' && value !== null && 'rank' in value && typeof value.rank === 'number'
    && (!('order' in value) || value.order === undefined || typeof value.order === 'number')
}

function orderSequenceTree(graph: LayoutGraph, ordinals: ReadonlyMap<string, number>): void {
  const layers = new Map<number, string[]>()
  for (const id of graph.nodes()) {
    const label = graph.node(id)
    if (!isRankedNode(label)) throw new Error(`Dagre did not assign a rank to ${id}.`)
    const layer = layers.get(label.rank) ?? []
    layer.push(id)
    layers.set(label.rank, layer)
  }
  const order = new Map(ordinals)
  for (const [, layer] of [...layers].sort(([a]: [number, string[]], [b]: [number, string[]]) => b - a)) {
    for (const id of layer) {
      if (order.has(id)) continue
      // Dagre splits long connections into temporary nodes. Keep each in its destination's subtree.
      const successors = graph.successors(id) ?? []
      const ordinal = successors.length === 1 ? order.get(successors[0]) : undefined
      if (ordinal === undefined) throw new Error(`Dagre did not preserve a tree connection at ${id}.`)
      order.set(id, ordinal)
    }
    layer.sort((a: string, b: string) => (order.get(a) ?? 0) - (order.get(b) ?? 0))
    for (const [index, id] of layer.entries()) {
      const label = graph.node(id)
      if (!isRankedNode(label)) throw new Error(`Dagre did not preserve the rank of ${id}.`)
      label.order = index
    }
  }
}

export function layoutTree(nodes: LayoutNode[]): TreeLayoutResult {
  if (nodes.length === 0) return { positions: [], edgeRoutes: [] }
  const byId = new Map(nodes.map((node: LayoutNode) => [node.id, node]))
  for (const node of nodes) {
    if (!Number.isSafeInteger(node.sequence) || node.sequence < 0) {
      throw new Error(`Message ${node.id} has an invalid sequence number: ${node.sequence}.`)
    }
    const parent = node.parentId ? byId.get(node.parentId) : undefined
    if (parent && parent.sequence >= node.sequence) {
      throw new Error(`Message ${node.id} (sequence ${node.sequence}) must follow its parent ${parent.id} (sequence ${parent.sequence}).`)
    }
  }
  const rows = treeSequenceRows(nodes)
  const ranks = new Map(rows.map((row: TreeSequenceRow, rank: number) => [row.sequence, rank]))
  const rankOf = (node: LayoutNode): number => {
    const rank = ranks.get(node.sequence)
    if (rank === undefined) throw new Error(`No sequence row for message ${node.id}.`)
    return rank
  }
  const graph = new dagre.graphlib.Graph()
    .setGraph({ rankdir: 'TB', nodesep: TREE_COLUMN_GAP, ranksep: TREE_ROW_GAP, ranker: 'tight-tree' })
    .setDefaultEdgeLabel(() => ({}))
  for (const node of nodes) {
    graph.setNode(node.id, { width: TREE_NODE_WIDTH, height: node.height ?? TREE_NODE_HEIGHT })
  }
  const roots = nodes.filter((node: LayoutNode) => !node.parentId || !byId.has(node.parentId))
  const children = new Map<string, LayoutNode[]>()
  for (const node of nodes) {
    if (!node.parentId || !byId.has(node.parentId)) continue
    const siblings = children.get(node.parentId) ?? []
    siblings.push(node)
    children.set(node.parentId, siblings)
  }
  const ordinals = new Map<string, number>()
  const stack = [...roots].reverse()
  while (stack.length > 0) {
    const node = stack.pop()
    if (!node) break
    ordinals.set(node.id, ordinals.size)
    for (const child of [...(children.get(node.id) ?? [])].reverse()) stack.push(child)
  }
  const lastRootRank = Math.max(...roots.map(rankOf))
  const guides: string[] = []
  // A shared, invisible rank spine aligns later roots without a separate long edge per root.
  for (let rank = 0; rank <= lastRootRank; rank++) {
    let id = `sequence-guide-${rank}`
    while (graph.hasNode(id)) id += '-'
    guides.push(id)
    ordinals.set(id, -1)
    graph.setNode(id, { width: 0, height: 0 })
    if (rank > 0) graph.setEdge(guides[rank - 1], id, { minlen: 1 })
  }
  for (const node of nodes) {
    const parent = node.parentId ? byId.get(node.parentId) : undefined
    graph.setEdge(parent?.id ?? guides[rankOf(node)], node.id, {
      minlen: parent ? rankOf(node) - rankOf(parent) : 1,
    })
  }
  dagre.layout(graph, { customOrder: (ranked: LayoutGraph) => { orderSequenceTree(ranked, ordinals) } })
  const positions = new Map<string, TreePosition>()
  const dagreRowCenters = new Map<number, number>()
  for (const node of nodes) {
    const row = rows[rankOf(node)]
    const point = graph.node(node.id)
    dagreRowCenters.set(node.sequence, point.y)
    positions.set(node.id, {
      x: point.x - TREE_NODE_WIDTH / 2,
      y: row.top + (row.height - (node.height ?? TREE_NODE_HEIGHT)) / 2,
    })
  }
  const edgeRoutes: Array<[string, TreePosition[]]> = []
  for (const node of nodes) {
    const parent = node.parentId ? byId.get(node.parentId) : undefined
    if (!parent) continue
    const source = positions.get(parent.id)
    const target = positions.get(node.id)
    if (!source || !target) throw new Error(`Missing connection positions for message ${node.id}.`)
    const waypoints = new Map<number, number>(
      graph.edge(parent.id, node.id).points.map((point: TreePosition) => [point.y, point.x]),
    )
    let x = source.x + TREE_NODE_WIDTH / 2
    const points: TreePosition[] = [{ x, y: source.y + (parent.height ?? TREE_NODE_HEIGHT) }]
    for (let rank = rankOf(parent) + 1; rank <= rankOf(node); rank++) {
      const row = rows[rank]
      const rowCenter = dagreRowCenters.get(row.sequence)
      const nextX = rank === rankOf(node) ? target.x + TREE_NODE_WIDTH / 2
        : rowCenter === undefined ? undefined : waypoints.get(rowCenter)
      if (nextX === undefined) throw new Error(`Missing connection waypoint for sequence ${row.sequence}.`)
      const y = row.top - TREE_ROW_GAP / 2
      points.push({ x, y }, { x: nextX, y })
      x = nextX
    }
    points.push({ x, y: target.y })
    edgeRoutes.push([node.id, points])
  }
  return { positions: [...positions], edgeRoutes: separateTreeRoutes(edgeRoutes) }
}
