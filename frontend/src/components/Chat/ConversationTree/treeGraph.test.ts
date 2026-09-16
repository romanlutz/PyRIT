import type { ConversationTreeEndpoint } from '@/types'

import { treeNode } from './__fixtures__/treeFixtures'
import { installStructuredClone } from './__mocks__/structuredClone'
import { conversationPath, orderedTreeNodes, indexTree, isInBranch, reservePositions, TREE_NODE_HEIGHT, TREE_NODE_WIDTH, TREE_ROW_GAP, type LayoutNode } from './treeGraph'
import { layoutTree } from './treeLayout'
import { visibleTreeNodes } from './useTreeViewport'

describe('tree graph utilities', () => {
  beforeEach(() => { jest.clearAllMocks(); installStructuredClone() })

  it('should retain all 1,000 messages and all exact-prefix endpoints without quadratic membership lists', () => {
    const nodes = Array.from({ length: 1_000 }, (_: unknown, index: number) => treeNode(`node-${index}`, {
      parent_node_id: index === 0 ? null : `node-${index - 1}`,
      message: { conversation_id: 'main', sequence: index },
      piece_count: 4,
      piece_types: ['text', 'image_path', 'audio_path', 'video_path'],
    }))
    const endpoints: ConversationTreeEndpoint[] = nodes.map((node, index: number) => ({ conversation_id: `conversation-${index}`, node_id: node.node_id }))
    const nodeMap = new Map(nodes.map((node) => [node.node_id, node]))
    const conversations = new Map(endpoints.map((endpoint: ConversationTreeEndpoint) => [endpoint.conversation_id, endpoint]))
    const index = indexTree(nodeMap, conversations)
    const path = conversationPath(index, conversations, 'conversation-999')
    expect(orderedTreeNodes(index)).toEqual(nodes)
    expect(index.conversationCounts.get('node-0')).toBe(1_000)
    expect(index.endpoints.get('node-0')).toEqual([endpoints[0]])
    expect(path.size).toBe(1_000)
    expect(isInBranch(index, 'node-999', 'node-1')).toBe(true)
    expect(isInBranch(index, 'node-0', 'node-1')).toBe(false)
    expect(index.conversationCounts.get('node-10')).toBe(990)
  })

  it('should use separate roots and count descendants without inventing an attack message', () => {
    const first = treeNode('first')
    const child = treeNode('child', { parent_node_id: 'first' })
    const second = treeNode('second')
    const index = indexTree(new Map([first, child, second].map((node) => [node.node_id, node])), new Map())
    expect(index.roots).toEqual(['first', 'second'])
    expect(orderedTreeNodes(index)).toEqual([first, child, second])
    expect(index.children.get('first')).toEqual(['child'])
    expect(isInBranch(index, 'second', 'first')).toBe(false)
  })

  it('should lay out whole message nodes with reserved dimensions', () => {
    const result = new Map(layoutTree([
      { id: 'root', parentId: null, sequence: 0 },
      { id: 'left', parentId: 'root', sequence: 1 },
      { id: 'right', parentId: 'root', sequence: 1 },
    ]).positions)
    const root = result.get('root')
    const left = result.get('left')
    const right = result.get('right')
    expect(root).toBeDefined()
    expect(left).toBeDefined()
    expect(right).toBeDefined()
    if (!root || !left || !right) throw new Error('Layout omitted a node')
    expect(left.y - root.y).toBeGreaterThan(TREE_NODE_HEIGHT)
    expect(Math.abs(left.x - right.x)).toBeGreaterThan(TREE_NODE_WIDTH)
  })

  it('should request only nodes intersecting the actual panned and zoomed viewport', () => {
    const first = treeNode('first')
    const second = treeNode('second')
    const far = treeNode('far')
    const positions = new Map([
      ['first', { x: 0, y: 0 }],
      ['second', { x: 400, y: 0 }],
      ['far', { x: 1_000, y: 800 }],
    ])
    expect(visibleTreeNodes([first, second, far], positions, { x: 0, y: 0, zoom: 1 }, { width: 350, height: 400 })).toEqual([first])
    expect(visibleTreeNodes([first, second, far], positions, { x: -800, y: 0, zoom: 2 }, { width: 640, height: 700 })).toEqual([second])
  })

  it('should not cover existing selectable nodes with temporary positions for a new page', () => {
    const previous = new Map([['old-root', { x: 0, y: 0 }]])
    const positions = reservePositions([
      { id: 'new-root', parentId: null, sequence: 0 }, { id: 'old-root', parentId: null, sequence: 0 },
    ], previous)
    expect(positions.get('old-root')).toEqual(previous.get('old-root'))
    expect(positions.get('new-root')?.x).toBeGreaterThan(TREE_NODE_WIDTH)
  })

  it('should align provisional sequence rows and preserve a focused anchor as earlier bands arrive', () => {
    const nodes: LayoutNode[] = [
      { id: 'earlier', parentId: null, sequence: 3, height: 304 },
      { id: 'old-root', parentId: null, sequence: 7, height: 208 },
      { id: 'new-peer', parentId: 'earlier', sequence: 7, height: 304 },
      { id: 'later', parentId: 'old-root', sequence: 15, height: 208 },
    ]
    const previous = new Map([['old-root', { x: 120, y: 100 }]])
    const positions = reservePositions(nodes, previous, 'old-root')
    expect(positions.get('old-root')).toEqual({ x: 120, y: 100 })
    expect(positions.get('new-peer')?.y).toBe(52)
    expect(positions.get('new-peer')?.x).toBeGreaterThan(120 + TREE_NODE_WIDTH)
    expect(positions.get('earlier')?.y).toBe(52 - 304 - TREE_ROW_GAP)
    expect(positions.get('later')?.y).toBe(52 + 304 + TREE_ROW_GAP)
  })

  it('should lay out a thousand stored sequence numbers without substituting their ordinal', () => {
    const nodes = Array.from({ length: 1_000 }, (_: unknown, index: number): LayoutNode => ({
      id: `node-${index}`, parentId: index === 0 ? null : `node-${index - 1}`, sequence: 7 + index * 3, height: 208,
    }))
    const result = layoutTree(nodes)
    expect(result.positions).toHaveLength(1_000)
    expect(result.edgeRoutes).toHaveLength(999)
    expect(new Map(result.positions).get('node-999')?.y).toBe(999 * (208 + TREE_ROW_GAP))
  })
})
