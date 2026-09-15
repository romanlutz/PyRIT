import { installStructuredClone } from './__mocks__/structuredClone'
import { treeEdgeLanes } from './treeEdgeRouting'
import {
  TREE_NODE_HEIGHT,
  TREE_NODE_WIDTH,
  TREE_ROW_GAP,
  type LayoutNode,
  type TreePosition,
} from './treeGraph'
import { layoutTree } from './treeLayout'

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

const UNEVEN_BRANCHES: LayoutNode[] = [
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
]

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

function checkGeometry(nodes: LayoutNode[]): void {
  const positions = new Map(layoutTree(nodes))
  const lanes = treeEdgeLanes(nodes, positions)
  const cards = nodes.map((node: LayoutNode): Rectangle => {
    const position = positions.get(node.id)
    if (!position) throw new Error(`No position for ${node.id}`)
    return { id: node.id, ...position, width: TREE_NODE_WIDTH, height: node.height ?? TREE_NODE_HEIGHT }
  })
  const byId = new Map(cards.map((card: Rectangle) => [card.id, card]))
  const connections: Connection[] = []
  for (const node of nodes) {
    if (!node.parentId) continue
    const parent = byId.get(node.parentId)
    const child = byId.get(node.id)
    const centerY = lanes.get(node.id)
    if (!parent || !child || centerY === undefined) throw new Error(`Incomplete edge for ${node.id}`)
    expect(centerY - (parent.y + parent.height)).toBeGreaterThanOrEqual(TREE_ROW_GAP / 2)
    expect(child.y - centerY).toBeGreaterThanOrEqual(TREE_ROW_GAP / 2)
    const points = [
      { x: parent.x + parent.width / 2, y: parent.y + parent.height },
      { x: parent.x + parent.width / 2, y: centerY },
      { x: child.x + child.width / 2, y: centerY },
      { x: child.x + child.width / 2, y: child.y },
    ]
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
    const positions = new Map(layoutTree(UNEVEN_BRANCHES))
    const siblings = UNEVEN_BRANCHES.filter((node: LayoutNode) => node.parentId === 'prompt')
    expect(new Set(siblings.map((node: LayoutNode) => positions.get(node.id)?.y)).size).toBe(1)
    checkGeometry(UNEVEN_BRANCHES)
  })

  it('reserves one clear routing lane for differently sized messages in the same row', () => {
    const nodes: LayoutNode[] = [
      { id: 'root', parentId: null, height: 208 },
      { id: 'short-a', parentId: 'root', height: 208 },
      { id: 'tall', parentId: 'root', height: 304 },
      { id: 'short-b', parentId: 'root', height: 208 },
    ]
    const positions = new Map(layoutTree(nodes))
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
      checkGeometry(Array.from({ length: 64 }, (_: unknown, index: number): LayoutNode => ({
        id: String(index),
        parentId: index === 0 ? null : String(Math.floor(random() * index)),
        height: 208 + 48 * Math.floor(random() * 3),
      })))
    }
  })
})
