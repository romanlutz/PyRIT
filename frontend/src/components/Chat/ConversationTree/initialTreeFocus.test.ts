import type { ConversationTreeNode } from '@/types'

import { treeNodeHeight, type TreePosition } from './treeGraph'
import { layoutTree } from './treeLayout'
import { initialTreeFocus } from './useTreeViewport'
import { installStructuredClone } from './__mocks__/structuredClone'

installStructuredClone()

function node(id: string, parent: string | null, pieceCount = 1): ConversationTreeNode {
  return {
    node_id: id,
    parent_node_id: parent,
    message: { conversation_id: 'conversation', sequence: parent ? 1 : 0 },
    role: parent ? 'assistant' : 'user',
    piece_count: pieceCount,
    piece_types: Array.from({ length: pieceCount }, () => 'text'),
    preview_key: id,
    created_at: '2026-01-01T00:00:00Z',
  }
}

describe('Readable initial tree focus', () => {
  const nodes = [node('parent', null), node('current', 'parent')]
  const path = new Set(['current', 'parent'])

  it('keeps the current message and its parent when both fit readably', () => {
    const positions = new Map<string, TreePosition>([
      ['parent', { x: 0, y: 0 }],
      ['current', { x: 0, y: 300 }],
    ])
    expect(initialTreeFocus(nodes, positions, path, { width: 1280, height: 720 })).toEqual(['current', 'parent'])
  })

  it('focuses the current message instead of shrinking a wide branch to unreadable text', () => {
    const positions = new Map<string, TreePosition>([
      ['parent', { x: 800, y: 0 }],
      ['current', { x: 0, y: 300 }],
    ])
    expect(initialTreeFocus(nodes, positions, path, { width: 330, height: 560 })).toEqual(['current'])
  })

  it('reserves bounded heights based on the number of visible pieces', () => {
    expect(treeNodeHeight(1)).toBe(208)
    expect(treeNodeHeight(2)).toBe(256)
    expect(treeNodeHeight(3)).toBe(304)
    expect(treeNodeHeight(50)).toBe(304)
  })

  it('passes actual node heights to the layout engine', () => {
    const positions = new Map(layoutTree([
      { id: 'parent', parentId: null, height: 208 },
      { id: 'current', parentId: 'parent', height: 304 },
    ]))
    const parent = positions.get('parent')
    const current = positions.get('current')
    expect(parent).toBeDefined()
    expect(current).toBeDefined()
    expect((current?.y ?? 0) - (parent?.y ?? 0)).toBe(272)
  })
})
