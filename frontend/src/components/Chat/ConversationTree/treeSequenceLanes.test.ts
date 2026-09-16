import { reservePositions, TREE_ROW_GAP, type LayoutNode } from './treeGraph'
import { treeSequenceLanes, visibleSequenceLanes, type SequenceLane } from './treeSequenceLanes'

describe('sequence lane geometry', () => {
  beforeEach(() => jest.clearAllMocks())

  it('should bound the tallest complete message in each stored sequence, not each graph depth', () => {
    const nodes: LayoutNode[] = [
      { id: 'root', parentId: null, sequence: 4, height: 208 },
      { id: 'reply', parentId: 'root', sequence: 10, height: 208 },
      { id: 'other-root', parentId: null, sequence: 10, height: 304 },
      { id: 'other-reply', parentId: 'other-root', sequence: 20, height: 256 },
    ]
    const positions = reservePositions(nodes, new Map())
    const lanes = treeSequenceLanes(nodes, positions)
    expect(lanes).toEqual([
      { sequence: 4, top: -16, bottom: 224 },
      { sequence: 10, top: 256, bottom: 592 },
      { sequence: 20, top: 624, bottom: 912 },
    ])
    expect(lanes[1].top - lanes[0].bottom).toBe(TREE_ROW_GAP / 2)
    expect(positions.get('reply')?.y).toBe(320)
    expect(positions.get('other-root')?.y).toBe(272)
  })

  it('should cull thousands of offscreen bands with the current pan and zoom', () => {
    const lanes = Array.from({ length: 2_000 }, (_: unknown, index: number): SequenceLane => ({
      sequence: 100 + index * 5, top: index * 400, bottom: index * 400 + 336,
    }))
    expect(visibleSequenceLanes(lanes, { x: 0, y: -400_000, zoom: 2 }, { width: 800, height: 800 }))
      .toEqual([lanes[500], lanes[501]])
    expect(visibleSequenceLanes(lanes, { x: 900, y: 0, zoom: 1 }, { width: 0, height: 0 })).toEqual([])
    expect(visibleSequenceLanes(lanes, { x: 0, y: 2_000, zoom: 1 }, { width: 800, height: 800 })).toEqual([])
  })

  it('should derive bands from translated accepted positions and omit absent messages', () => {
    const lanes = treeSequenceLanes([
      { id: 'first', parentId: null, sequence: 8, height: 208 },
      { id: 'missing', parentId: null, sequence: 15, height: 208 },
    ], new Map([['first', { x: -200, y: -500 }]]))
    expect(lanes).toEqual([{ sequence: 8, top: -516, bottom: -276 }])
  })
})
