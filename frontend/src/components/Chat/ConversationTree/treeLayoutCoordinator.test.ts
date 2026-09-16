import type { TreePosition } from './treeGraph'
import type { TreeLayoutReply, TreeLayoutRequest, TreeLayoutWorkerPort } from './treeLayout.types'
import { TreeLayoutCoordinator } from './treeLayoutCoordinator'

class ControlledWorker implements TreeLayoutWorkerPort {
  onmessage: ((event: MessageEvent<TreeLayoutReply>) => void) | null = null
  onerror: ((event: ErrorEvent) => void) | null = null
  postMessage = jest.fn<void, [TreeLayoutRequest]>()
  terminate = jest.fn()

  reply(requestId: number, positions: Array<[string, TreePosition]>, edgeRoutes: Array<[string, TreePosition[]]> = []): void {
    this.onmessage?.(new MessageEvent<TreeLayoutReply>('message', { data: { requestId, positions, edgeRoutes } }))
  }
}

describe('TreeLayoutCoordinator', () => {
  let worker: ControlledWorker
  let layout: TreeLayoutCoordinator

  beforeEach(() => {
    jest.clearAllMocks()
    jest.useFakeTimers()
    worker = new ControlledWorker()
    layout = new TreeLayoutCoordinator(() => worker)
  })

  afterEach(() => {
    layout.setActive(false)
    jest.useRealTimers()
  })

  it('should coalesce pages without delaying the first useful layout indefinitely', () => {
    layout.setActive(true)
    layout.setGraph([{ id: 'root', parentId: null, sequence: 0 }], 'root')
    jest.advanceTimersByTime(16)
    layout.setGraph([{ id: 'root', parentId: null, sequence: 0 }, { id: 'child', parentId: 'root', sequence: 1 }], 'root')
    jest.advanceTimersByTime(16)
    expect(worker.postMessage).toHaveBeenCalledTimes(1)
    expect(worker.postMessage.mock.calls[0][0].nodes).toHaveLength(2)
    expect(layout.getSnapshot().positions.size).toBe(2)
    expect(layout.getSnapshot().arrangedNodeIds.size).toBe(0)
  })

  it('should discard stale worker replies and keep the active anchor at the same position', () => {
    layout.setActive(true)
    layout.setGraph([{ id: 'root', parentId: null, sequence: 0 }], 'root')
    jest.advanceTimersByTime(32)
    const first = worker.postMessage.mock.calls[0][0].requestId
    layout.setGraph([{ id: 'root', parentId: null, sequence: 0 }, { id: 'child', parentId: 'root', sequence: 1 }], 'root')
    worker.reply(first, [['root', { x: 999, y: 999 }]])
    expect(layout.getSnapshot().positions.get('root')).toEqual({ x: 0, y: 0 })
    jest.advanceTimersByTime(32)
    const current = worker.postMessage.mock.calls[1][0].requestId
    worker.reply(current, [['root', { x: 100, y: 50 }], ['child', { x: 200, y: 500 }]])
    expect(layout.getSnapshot().positions.get('root')).toEqual({ x: 0, y: 0 })
    expect(layout.getSnapshot().positions.get('child')).toEqual({ x: 100, y: 450 })
    expect(layout.getSnapshot().ready).toBe(true)
    expect([...layout.getSnapshot().arrangedNodeIds]).toEqual(['root', 'child'])
  })

  it('keeps existing routed branches but does not connect provisional nodes through them', () => {
    layout.setActive(true)
    layout.setGraph([{ id: 'root', parentId: null, sequence: 0 }], 'root')
    jest.advanceTimersByTime(32)
    worker.reply(worker.postMessage.mock.calls[0][0].requestId, [['root', { x: 0, y: 0 }]])
    layout.setGraph([{ id: 'root', parentId: null, sequence: 0 }, { id: 'new-child', parentId: 'root', sequence: 1 }], 'root')
    expect(layout.getSnapshot().positions.has('new-child')).toBe(true)
    expect([...layout.getSnapshot().arrangedNodeIds]).toEqual(['root'])
    jest.advanceTimersByTime(32)
    worker.reply(worker.postMessage.mock.calls[1][0].requestId, [
      ['root', { x: 0, y: 0 }],
      ['new-child', { x: 0, y: 400 }],
    ])
    expect([...layout.getSnapshot().arrangedNodeIds]).toEqual(['root', 'new-child'])
  })

  it('should suspend worker work while hidden and preserve accepted positions on return', () => {
    layout.setGraph([{ id: 'root', parentId: null, sequence: 0 }], 'root')
    jest.advanceTimersByTime(100)
    expect(worker.postMessage).not.toHaveBeenCalled()
    layout.setActive(true)
    jest.advanceTimersByTime(32)
    worker.reply(worker.postMessage.mock.calls[0][0].requestId, [['root', { x: 20, y: 30 }]])
    const positions = layout.getSnapshot().positions
    layout.setActive(false)
    expect(worker.terminate).toHaveBeenCalledTimes(1)
    worker.reply(1, [['root', { x: 900, y: 900 }]])
    layout.setActive(true)
    jest.advanceTimersByTime(100)
    expect(layout.getSnapshot().positions).toBe(positions)
    expect(worker.postMessage).toHaveBeenCalledTimes(1)
  })

  it('should report worker failures locally and permit a targeted retry', () => {
    layout.setGraph([{ id: 'root', parentId: null, sequence: 0 }], null)
    layout.setActive(true)
    jest.advanceTimersByTime(32)
    worker.onerror?.(new ErrorEvent('error', { message: 'Worker failed' }))
    expect(layout.getSnapshot()).toMatchObject({ error: 'Worker failed', busy: false })
    expect(layout.getSnapshot().positions.size).toBe(1)
    layout.retry()
    jest.advanceTimersByTime(32)
    expect(worker.postMessage).toHaveBeenCalledTimes(2)
  })

  it('should retain selectable placeholders if the browser cannot create a worker', () => {
    layout = new TreeLayoutCoordinator(() => { throw new Error('Workers unavailable') })
    layout.setGraph([{ id: 'root', parentId: null, sequence: 0 }], null)
    layout.setActive(true)
    jest.advanceTimersByTime(32)
    expect(layout.getSnapshot().error).toBe('Workers unavailable')
    expect(layout.getSnapshot().positions.has('root')).toBe(true)
  })

  it('should translate accepted routes with the anchor and withhold them when an intermediate band arrives', () => {
    const root = { id: 'root', parentId: null, sequence: 5 }
    const child = { id: 'child', parentId: 'root', sequence: 9 }
    layout.setGraph([root, child], 'child')
    layout.setActive(true)
    jest.advanceTimersByTime(32)
    const points = [{ x: 260, y: 386 }, { x: 260, y: 418 }, { x: 360, y: 418 }, { x: 360, y: 450 }]
    worker.reply(worker.postMessage.mock.calls[0][0].requestId, [
      ['root', { x: 100, y: 50 }], ['child', { x: 200, y: 450 }],
    ], [['child', points]])
    expect(layout.getSnapshot().positions.get('child')).toEqual({ x: 0, y: 400 })
    expect(layout.getSnapshot().edgeRoutes.get('child')).toEqual(
      points.map((point: TreePosition) => ({ x: point.x - 200, y: point.y - 50 })),
    )

    layout.setGraph([root, { id: 'new-root', parentId: null, sequence: 7 }, child], 'child')
    expect(layout.getSnapshot().positions.get('child')).toEqual({ x: 0, y: 400 })
    expect(layout.getSnapshot().positions.get('root')?.y).toBe(-400)
    expect(layout.getSnapshot().edgeRoutes.size).toBe(0)
    expect(layout.getSnapshot().arrangedNodeIds.size).toBe(0)
    jest.advanceTimersByTime(32)
    expect(worker.postMessage.mock.calls[1][0].nodes.map((node) => node.sequence)).toEqual([5, 7, 9])
    worker.reply(worker.postMessage.mock.calls[1][0].requestId, [
      ['root', { x: 100, y: 0 }], ['new-root', { x: 500, y: 400 }], ['child', { x: 100, y: 800 }],
    ], [['child', [{ x: 260, y: 336 }, { x: 260, y: 800 }]]])
    expect(layout.getSnapshot().positions.get('child')).toEqual({ x: 0, y: 400 })
    expect(layout.getSnapshot().edgeRoutes.get('child')).toEqual([{ x: 160, y: -64 }, { x: 160, y: 400 }])
    expect(layout.getSnapshot().arrangedNodeIds.size).toBe(3)
  })
})
