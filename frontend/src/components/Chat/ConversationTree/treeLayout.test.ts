import { installStructuredClone } from './__mocks__/structuredClone'
import { treeEdgeLanes } from './treeEdgeRouting'
import {
  TREE_NODE_HEIGHT,
  TREE_NODE_WIDTH,
  TREE_ROW_GAP,
  treeNodeHeight,
  type LayoutNode,
  type TreePosition,
} from './treeGraph'
import { layoutTree } from './treeLayout'
import { treeSequenceLanes } from './treeSequenceLanes'

interface Rectangle extends TreePosition {
  id: string
  width: number
  height: number
}

type Segment = readonly [TreePosition, TreePosition]

interface Connection {
  source: string
  target: string
  segments: Segment[]
}

interface TestNode extends Omit<LayoutNode, 'sequence'> {
  readonly sequence?: number
}

function withSequences(nodes: TestNode[]): LayoutNode[] {
  const sequences = new Map<string, number>()
  return nodes.map((node: TestNode): LayoutNode => {
    const sequence = node.sequence ?? (node.parentId ? (sequences.get(node.parentId) ?? -1) + 1 : 0)
    sequences.set(node.id, sequence)
    return { ...node, sequence }
  })
}

const UNEVEN_BRANCHES = withSequences([
  { id: 'root', parentId: null, height: 208 },
  { id: 'context', parentId: 'root', height: 208 },
  { id: 'prompt', parentId: 'context', height: 208 },
  { id: 'short-a', parentId: 'prompt', height: 208 },
  { id: 'media-branch', parentId: 'prompt', height: 208 },
  { id: 'media', parentId: 'media-branch', height: 304 },
  { id: 'short-b', parentId: 'prompt', height: 208 },
  { id: 'short-c', parentId: 'prompt', height: 208 },
  { id: 'deep-branch', parentId: 'prompt', height: 208 },
  { id: 'next-prompt', parentId: 'deep-branch', height: 208 },
  { id: 'next-a', parentId: 'next-prompt', height: 208 },
  { id: 'next-b', parentId: 'next-prompt', height: 208 },
  { id: 'next-c', parentId: 'next-prompt', height: 208 },
])

function overlaps(a: Rectangle, b: Rectangle): boolean {
  return a.x < b.x + b.width && a.x + a.width > b.x
    && a.y < b.y + b.height && a.y + a.height > b.y
}

function hitsCard([a, b]: Segment, card: Rectangle): boolean {
  return a.y === b.y
    ? a.y > card.y && a.y < card.y + card.height
      && Math.max(a.x, b.x) > card.x && Math.min(a.x, b.x) < card.x + card.width
    : a.x > card.x && a.x < card.x + card.width
      && Math.max(a.y, b.y) > card.y && Math.min(a.y, b.y) < card.y + card.height
}

function crosses(a: Segment, b: Segment): boolean {
  const horizontalA = a[0].y === a[1].y
  const horizontalB = b[0].y === b[1].y
  const min = (segment: Segment, axis: 'x' | 'y'): number => Math.min(segment[0][axis], segment[1][axis])
  const max = (segment: Segment, axis: 'x' | 'y'): number => Math.max(segment[0][axis], segment[1][axis])
  if (horizontalA === horizontalB) {
    const axis = horizontalA ? 'x' : 'y'
    const fixedAxis = horizontalA ? 'y' : 'x'
    return a[0][fixedAxis] === b[0][fixedAxis]
      && Math.max(min(a, axis), min(b, axis)) < Math.min(max(a, axis), max(b, axis))
  }
  const [horizontal, vertical] = horizontalA ? [a, b] : [b, a]
  return vertical[0].x > min(horizontal, 'x') && vertical[0].x < max(horizontal, 'x')
    && horizontal[0].y > min(vertical, 'y') && horizontal[0].y < max(vertical, 'y')
}

function checkGeometry(input: TestNode[]): void {
  const nodes = withSequences(input)
  const result = layoutTree(nodes)
  const positions = new Map(result.positions)
  const routes = new Map(result.edgeRoutes)
  const lanes = treeEdgeLanes(nodes, positions)
  const cards = nodes.map((node: LayoutNode): Rectangle => {
    const position = positions.get(node.id)
    if (!position) throw new Error(`No position for ${node.id}`)
    return { id: node.id, ...position, width: TREE_NODE_WIDTH, height: node.height ?? TREE_NODE_HEIGHT }
  })
  const byId = new Map(cards.map((card: Rectangle) => [card.id, card]))
  const rowCenters = new Map<number, number>()
  for (const node of nodes) {
    const card = byId.get(node.id)
    if (!card) throw new Error(`Missing card ${node.id}`)
    const center = card.y + card.height / 2
    expect(center).toBe(rowCenters.get(node.sequence) ?? center)
    rowCenters.set(node.sequence, center)
  }
  const centers = [...rowCenters].sort(([a]: [number, number], [b]: [number, number]) => a - b)
    .map(([, center]: [number, number]) => center)
  expect(centers).toEqual([...centers].sort((a: number, b: number) => a - b))
  const connections: Connection[] = []
  for (const node of nodes) {
    if (!node.parentId) continue
    const parent = byId.get(node.parentId)
    const child = byId.get(node.id)
    const centerY = lanes.get(node.id)
    if (!parent || !child || centerY === undefined) throw new Error(`Incomplete edge for ${node.id}`)
    expect(centerY - (parent.y + parent.height)).toBeGreaterThanOrEqual(TREE_ROW_GAP / 2)
    expect(child.y - centerY).toBeGreaterThanOrEqual(TREE_ROW_GAP / 2)
    const points = routes.get(node.id)
    if (!points) throw new Error(`Missing route for ${node.id}`)
    expect(points[0]).toEqual({ x: parent.x + parent.width / 2, y: parent.y + parent.height })
    expect(points[points.length - 1]).toEqual({ x: child.x + child.width / 2, y: child.y })
    for (const [index, point] of points.entries()) {
      if (index > 0) expect(point.x === points[index - 1].x || point.y === points[index - 1].y).toBe(true)
    }
    connections.push({
      source: parent.id,
      target: child.id,
      segments: points.slice(1).map((point: TreePosition, index: number): Segment => [points[index], point])
        .filter(([a, b]: Segment) => a.x !== b.x || a.y !== b.y),
    })
  }
  const collisions: string[] = []
  for (let index = 0; index < cards.length; index++) {
    for (const card of cards.slice(index + 1)) {
      if (overlaps(cards[index], card)) collisions.push(`Cards ${cards[index].id} / ${card.id}`)
    }
  }
  for (const connection of connections) {
    for (const card of cards) {
      if (card.id === connection.source || card.id === connection.target) continue
      if (connection.segments.some((segment: Segment) => hitsCard(segment, card))) {
        collisions.push(`Edge ${connection.source} -> ${connection.target} through ${card.id}`)
      }
    }
  }
  for (let index = 0; index < connections.length; index++) {
    const first = connections[index]
    for (const second of connections.slice(index + 1)) {
      if (first.source === second.source) continue
      if (first.segments.some((segment: Segment) => second.segments.some((other: Segment) => crosses(segment, other)))) {
        collisions.push(`Crossed ${first.source} -> ${first.target} / ${second.source} -> ${second.target}`)
      }
    }
  }
  expect(collisions).toEqual([])
}

describe('Conversation tree geometry', () => {
  beforeAll(installStructuredClone)
  beforeEach(() => jest.clearAllMocks())

  it('keeps short branches beside the first reply instead of pushing them below deeper branches', () => {
    const positions = new Map(layoutTree(UNEVEN_BRANCHES).positions)
    const siblings = UNEVEN_BRANCHES.filter((node: LayoutNode) => node.parentId === 'prompt')
    expect(new Set(siblings.map((node: LayoutNode) => positions.get(node.id)?.y)).size).toBe(1)
    checkGeometry(UNEVEN_BRANCHES)
  })

  it('reserves one clear routing lane for differently sized messages in the same row', () => {
    const nodes = withSequences([
      { id: 'root', parentId: null, height: 208 },
      { id: 'short-a', parentId: 'root', height: 208 },
      { id: 'tall', parentId: 'root', height: 304 },
      { id: 'short-b', parentId: 'root', height: 208 },
    ])
    const positions = new Map(layoutTree(nodes).positions)
    const lanes = treeEdgeLanes(nodes, positions)
    expect(new Set(nodes.slice(1).map((node: LayoutNode) => lanes.get(node.id))).size).toBe(1)
    checkGeometry(nodes)
  })

  it('keeps a neighboring subtree clear when a tall media node extends into its routing band', () => {
    checkGeometry([
      { id: 'root', parentId: null, height: 208 },
      { id: 'left', parentId: 'root', height: 208 },
      { id: 'middle', parentId: 'root', height: 304 },
      { id: 'right', parentId: 'root', height: 208 },
      { id: 'left-a', parentId: 'left', height: 304 },
      { id: 'left-b', parentId: 'left', height: 208 },
      { id: 'middle-a', parentId: 'middle', height: 208 },
      { id: 'right-a', parentId: 'right', height: 208 },
      { id: 'right-b', parentId: 'right', height: 304 },
    ])
  })

  it('handles independent roots without overlapping their branches', () => {
    checkGeometry([
      ...UNEVEN_BRANCHES,
      { id: 'other-root', parentId: null, height: 304 },
      { id: 'other-a', parentId: 'other-root', height: 208 },
      { id: 'other-b', parentId: 'other-root', height: 256 },
    ])
  })

  it('avoids card and edge collisions across varied deterministic tree shapes', () => {
    let seed = 1
    const random = (): number => {
      seed = (seed * 1664525 + 1013904223) >>> 0
      return seed / 0x100000000
    }
    for (let sample = 0; sample < 24; sample++) {
      checkGeometry(Array.from({ length: 64 }, (_: unknown, index: number): TestNode => ({
        id: String(index),
        parentId: index === 0 ? null : String(Math.floor(random() * index)),
        height: 208 + 48 * Math.floor(random() * 3),
      })))
    }
  })

  it('aligns actual sequences across unrelated roots, nonzero histories, gaps, and mixed heights', () => {
    const nodes: LayoutNode[] = [
      { id: 'first', parentId: null, sequence: 3, height: 208 },
      { id: 'first-reply', parentId: 'first', sequence: 8, height: 208 },
      { id: 'second', parentId: null, sequence: 8, height: 304 },
      { id: 'second-reply', parentId: 'second', sequence: 12, height: 256 },
      { id: 'third', parentId: null, sequence: 5, height: 208 },
      { id: 'third-reply', parentId: 'third', sequence: 12, height: 208 },
    ]
    const positions = new Map(layoutTree(nodes).positions)
    const reply = positions.get('first-reply')
    const second = positions.get('second')
    if (!reply || !second) throw new Error('The layout omitted messages')
    expect(positions.get('first')?.y).toBe(0)
    expect(reply.y + 208 / 2).toBe(second.y + 304 / 2)
    expect(second.y).toBe(2 * (208 + TREE_ROW_GAP))
    checkGeometry(nodes)
  })

  it('routes skipped sequences around occupied intermediate bands without crossing other branches', () => {
    checkGeometry([
      { id: 'root', parentId: null, sequence: 0, height: 208 },
      { id: 'early', parentId: 'root', sequence: 1, height: 304 },
      { id: 'early-a', parentId: 'early', sequence: 2, height: 304 },
      { id: 'early-b', parentId: 'early', sequence: 4, height: 208 },
      { id: 'middle', parentId: 'root', sequence: 2, height: 208 },
      { id: 'middle-reply', parentId: 'middle', sequence: 5, height: 304 },
      { id: 'late', parentId: 'root', sequence: 5, height: 304 },
      { id: 'late-reply', parentId: 'late', sequence: 7, height: 208 },
      { id: 'other', parentId: null, sequence: 1, height: 208 },
      { id: 'other-reply', parentId: 'other', sequence: 4, height: 304 },
    ])
  })

  it('keeps numeric sequence alignment and clear routes across varied sparse forests', () => {
    let seed = 41
    const random = (): number => {
      seed = (seed * 1664525 + 1013904223) >>> 0
      return seed / 0x100000000
    }
    for (let sample = 0; sample < 24; sample++) {
      const nodes: LayoutNode[] = []
      for (let index = 0; index < 64; index++) {
        const parent = index === 0 || random() < 0.1 ? undefined : nodes[Math.floor(random() * index)]
        nodes.push({
          id: String(index),
          parentId: parent?.id ?? null,
          sequence: parent ? parent.sequence + 1 + Math.floor(random() * 5) : Math.floor(random() * 6),
          height: 208 + 48 * Math.floor(random() * 3),
        })
      }
      checkGeometry(nodes)
    }
  })

  it('reports non-increasing stored sequences instead of silently substituting graph depth', () => {
    expect(() => layoutTree([
      { id: 'root', parentId: null, sequence: 4 },
      { id: 'child', parentId: 'root', sequence: 4 },
    ])).toThrow('Message child (sequence 4) must follow its parent root (sequence 4).')
  })

  it('does not allocate empty bands for large numeric gaps', () => {
    const result = layoutTree([
      { id: 'root', parentId: null, sequence: 500_000, height: 208 },
      { id: 'child', parentId: 'root', sequence: 1_000_000, height: 208 },
    ])
    expect(new Map(result.positions).get('child')?.y).toBe(208 + TREE_ROW_GAP)
    expect(result.edgeRoutes).toHaveLength(1)
  })

  it('preserves subtree order when later roots and descendants share a sequence band', () => {
    const nodes: LayoutNode[] = [
      { id: 'root', parentId: null, sequence: 2, height: 208 },
      { id: 'left', parentId: 'root', sequence: 3, height: 208 },
      { id: 'left-reply', parentId: 'left', sequence: 7, height: 304 },
      { id: 'right', parentId: 'root', sequence: 7, height: 208 },
      { id: 'other-root', parentId: null, sequence: 7, height: 208 },
    ]
    const positions = new Map(layoutTree(nodes).positions)
    const row = ['left-reply', 'right', 'other-root'].map((id: string) => positions.get(id)?.x)
    expect(row).toEqual([...row].sort((a: number | undefined, b: number | undefined) => (a ?? 0) - (b ?? 0)))
    checkGeometry(nodes)
  })

  it('reserves clear sequence bands for the four-message Crescendo image-editing shape', () => {
    const nodes: LayoutNode[] = [
      { id: 'seed-prompt', parentId: null, sequence: 0, height: treeNodeHeight(3) },
      { id: 'first-image', parentId: 'seed-prompt', sequence: 1, height: treeNodeHeight(1) },
      { id: 'edit-prompt', parentId: 'first-image', sequence: 2, height: treeNodeHeight(2) },
      { id: 'edited-image', parentId: 'edit-prompt', sequence: 3, height: treeNodeHeight(1) },
    ]
    const result = layoutTree(nodes)
    const positions = new Map(result.positions)
    expect(nodes.map((node: LayoutNode) => node.height)).toEqual([304, 208, 256, 208])
    expect(result.positions.map(([, position]: [string, TreePosition]) => position.y)).toEqual([0, 368, 640, 960])
    expect(result.edgeRoutes.map(([id]: [string, TreePosition[]]) => id))
      .toEqual(['first-image', 'edit-prompt', 'edited-image'])
    expect(treeSequenceLanes(nodes, positions)).toEqual([
      { sequence: 0, top: -16, bottom: 320 },
      { sequence: 1, top: 352, bottom: 592 },
      { sequence: 2, top: 624, bottom: 912 },
      { sequence: 3, top: 944, bottom: 1184 },
    ])
    checkGeometry(nodes)
  })
})
