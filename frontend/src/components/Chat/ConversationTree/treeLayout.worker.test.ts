import { layoutTree } from './treeLayout'
import type { TreeLayoutRequest } from './treeLayout.types'

jest.mock('./treeLayout', () => ({ layoutTree: jest.fn() }))

describe('tree layout worker', () => {
  const original = self.onmessage

  beforeEach(async () => {
    jest.clearAllMocks()
    jest.spyOn(self, 'postMessage').mockImplementation(() => {})
    await jest.isolateModulesAsync(async () => { await import('./treeLayout.worker') })
  })

  afterEach(() => {
    self.onmessage = original
    jest.restoreAllMocks()
  })

  it('should return positions with the request identity', () => {
    jest.mocked(layoutTree).mockReturnValue([['root', { x: 10, y: 20 }]])
    const request: TreeLayoutRequest = { requestId: 5, nodes: [{ id: 'root', parentId: null }] }
    self.onmessage?.(new MessageEvent<TreeLayoutRequest>('message', { data: request }))
    expect(self.postMessage).toHaveBeenCalledWith({ requestId: 5, positions: [['root', { x: 10, y: 20 }]] })
  })

  it('should return a recoverable error instead of leaving a layout request pending', () => {
    jest.mocked(layoutTree).mockImplementation(() => { throw new Error('Layout unavailable') })
    self.onmessage?.(new MessageEvent<TreeLayoutRequest>('message', { data: { requestId: 6, nodes: [] } }))
    expect(self.postMessage).toHaveBeenCalledWith({ requestId: 6, error: 'Layout unavailable' })
  })
})
