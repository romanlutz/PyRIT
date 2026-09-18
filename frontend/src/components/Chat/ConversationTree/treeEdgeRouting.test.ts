import { separateTreeRoutes, simplifyTreeRoute, treeEdgeLanes, treeEdgePath } from './treeEdgeRouting'

describe('conversation connection routing', () => {
  beforeEach(() => jest.clearAllMocks())

  it('should share a clear gap above the tallest message of the actual sequence', () => {
    const lanes = treeEdgeLanes([
      { id: 'short', parentId: null, sequence: 8, height: 208 },
      { id: 'tall', parentId: null, sequence: 8, height: 304 },
      { id: 'later', parentId: null, sequence: 15, height: 208 },
      { id: 'missing', parentId: null, sequence: 8 },
    ], new Map([
      ['short', { x: 0, y: 100 }], ['tall', { x: 400, y: 52 }], ['later', { x: 0, y: 600 }],
    ]))
    expect([...lanes]).toEqual([['short', 20], ['tall', 20], ['later', 568]])
  })

  it('should remove only redundant waypoints, retaining each turn around intermediate bands', () => {
    expect(simplifyTreeRoute([
      { x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 10 }, { x: 0, y: 20 },
      { x: 10, y: 20 }, { x: 20, y: 20 }, { x: 20, y: 60 }, { x: 40, y: 60 }, { x: 40, y: 80 },
    ])).toEqual([
      { x: 0, y: 0 }, { x: 0, y: 20 }, { x: 20, y: 20 },
      { x: 20, y: 60 }, { x: 40, y: 60 }, { x: 40, y: 80 },
    ])
  })

  it('should round orthogonal corners inside their clear routing gaps', () => {
    expect(treeEdgePath([
      { x: 0, y: 0 }, { x: 0, y: 40 }, { x: 80, y: 40 }, { x: 80, y: 100 },
    ])).toBe('M 0 0 L 0 32 Q 0 40 8 40 L 72 40 Q 80 40 80 48 L 80 100')
    expect(treeEdgePath([{ x: 10, y: 0 }, { x: 10, y: 200 }])).toBe('M 10 0 L 10 200')
    expect(() => treeEdgePath([{ x: 10, y: 0 }])).toThrow('at least two distinct points')
  })

  it.each([1, -1])('should round along slanted segments without exaggerating their direction (%s)', (direction: number) => {
    expect(treeEdgePath([
      { x: 0, y: 0 }, { x: direction * 6, y: 8 }, { x: direction * 26, y: 8 },
    ])).toBe(`M 0 0 L ${direction * 3} 4 Q ${direction * 6} 8 ${direction * 11} 8 L ${direction * 26} 8`)
  })

  it('should keep the radius inside short routing segments', () => {
    expect(treeEdgePath([
      { x: 0, y: 0 }, { x: 0, y: 4 }, { x: 4, y: 4 }, { x: 4, y: 8 },
    ])).toBe('M 0 0 L 0 2 Q 0 4 2 4 L 2 4 Q 4 4 4 6 L 4 8')
  })

  it('should stagger unrelated rightward connections instead of overlapping their horizontal runs', () => {
    const routes = new Map(separateTreeRoutes([
      ['left', [{ x: 0, y: 0 }, { x: 0, y: 100 }, { x: 100, y: 100 }, { x: 100, y: 200 }]],
      ['right', [{ x: 50, y: 0 }, { x: 50, y: 100 }, { x: 150, y: 100 }, { x: 150, y: 200 }]],
    ]))
    const left = routes.get('left'), right = routes.get('right')
    if (!left || !right) throw new Error('Missing separated connection')
    expect(left[1].y).toBeGreaterThan(right[1].y)
    expect(right[1].y).toBeGreaterThan(84)
    expect(left[1].y).toBeLessThan(116)
  })

  it('should reverse the staggering order for leftward connections', () => {
    const routes = new Map(separateTreeRoutes([
      ['left', [{ x: 100, y: 0 }, { x: 100, y: 100 }, { x: 0, y: 100 }, { x: 0, y: 200 }]],
      ['right', [{ x: 150, y: 0 }, { x: 150, y: 100 }, { x: 50, y: 100 }, { x: 50, y: 200 }]],
    ]))
    const left = routes.get('left'), right = routes.get('right')
    if (!left || !right) throw new Error('Missing separated connection')
    expect(left[1].y).toBeLessThan(right[1].y)
  })
})
