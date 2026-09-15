import { attacksApi } from '@/services/api'
import type { ConversationTreePage, ConversationTreePreviewResponse, TreeMessageReference } from '@/types'

import { deferred, treeNode, treePage, treePiece, treePreview } from './__fixtures__/treeFixtures'
import { PREVIEW_BATCH_SIZE, TreeReadCoordinator } from './treeReadCoordinator'

jest.mock('@/services/api', () => ({
  attacksApi: { getConversationTree: jest.fn(), getTreePreviews: jest.fn() },
}))

const getTree = jest.mocked(attacksApi.getConversationTree)
const getPreviews = jest.mocked(attacksApi.getTreePreviews)

describe('TreeReadCoordinator', () => {
  let reads: TreeReadCoordinator
  const root = treeNode('root')
  const response = treeNode('response', { parent_node_id: root.node_id, role: 'assistant', message: { conversation_id: 'main', sequence: 1 } })

  beforeEach(() => {
    jest.clearAllMocks()
    jest.useFakeTimers()
    getTree.mockReset().mockResolvedValue(treePage([root]))
    getPreviews.mockReset().mockImplementation(async (_attack: string, messages: TreeMessageReference[]) => ({
      previews: messages.map((message: TreeMessageReference) => treePreview(message)),
    }))
    reads = new TreeReadCoordinator('attack', 'main', 0)
  })

  afterEach(() => {
    reads.setActive(false)
    jest.useRealTimers()
  })

  it('should publish the first topology page without awaiting later structure or previews', async () => {
    const laterPage = deferred<ConversationTreePage>()
    const text = deferred<ConversationTreePreviewResponse>()
    getTree.mockResolvedValueOnce(treePage([root], { complete: false, next_cursor: 'page-2', total_conversations: 8 }))
      .mockReturnValueOnce(laterPage.promise)
    getPreviews.mockReturnValueOnce(text.promise)
    reads.setActive(true)
    await jest.advanceTimersByTimeAsync(0)
    reads.setVisibleNodes([root])
    await jest.advanceTimersByTimeAsync(16)

    expect(reads.getSnapshot()).toMatchObject({ processed: 1, total: 8, complete: false, loading: true })
    expect(reads.getSnapshot().nodes.get('root')).toEqual(root)
    expect(getTree).toHaveBeenNthCalledWith(1, 'attack', { limit: 100, prioritize_conversation_id: 'main' }, expect.any(AbortSignal))
    expect(getTree).toHaveBeenNthCalledWith(2, 'attack', { limit: 100, cursor: 'page-2' }, expect.any(AbortSignal))
    expect(reads.getSnapshot().previews.get(root.preview_key)?.text?.loading).toBe(true)
  })

  it('should reserve an interaction slot while both topology and visible text are pending', async () => {
    const page = deferred<ConversationTreePage>()
    const text = deferred<ConversationTreePreviewResponse>()
    const full = deferred<ConversationTreePreviewResponse>()
    getTree.mockReturnValue(page.promise)
    getPreviews.mockReturnValueOnce(text.promise).mockReturnValueOnce(full.promise)
    reads.setActive(true)
    reads.setVisibleNodes([root])
    reads.requestPreviews([root], 'full')

    expect(getTree).toHaveBeenCalledTimes(1)
    expect(getPreviews.mock.calls.map((call) => call[2])).toEqual(['text', 'full'])
    expect(reads.getSnapshot().previews.get(root.preview_key)?.full?.loading).toBe(true)
  })

  it('should batch previews at 64 and never start every promise at once', async () => {
    const firstBatch = deferred<ConversationTreePreviewResponse>()
    const nodes = Array.from({ length: 150 }, (_: unknown, sequence: number) => treeNode(`node-${sequence}`, {
      message: { conversation_id: 'main', sequence },
    }))
    getPreviews.mockReturnValueOnce(firstBatch.promise)
    reads.setActive(true)
    reads.setVisibleNodes(nodes)
    expect(getPreviews).toHaveBeenCalledTimes(1)
    expect(getPreviews.mock.calls[0][1]).toHaveLength(PREVIEW_BATCH_SIZE)
    firstBatch.resolve({ previews: nodes.slice(0, PREVIEW_BATCH_SIZE).map((node) => treePreview(node.message)) })
    await jest.advanceTimersByTimeAsync(0)
    expect(getPreviews.mock.calls.map((call) => call[1].length)).toEqual([64, 64, 22])
  })

  it('should deduplicate in-flight and cached previews by stable preview key', async () => {
    const text = deferred<ConversationTreePreviewResponse>()
    const alias = treeNode('alias', { preview_key: root.preview_key, message: { conversation_id: 'copy', sequence: 0 } })
    getPreviews.mockReturnValueOnce(text.promise)
    reads.setActive(true)
    reads.setVisibleNodes([root, alias])
    reads.requestPreviews([alias])
    reads.setVisibleNodes([alias, root])
    expect(getPreviews).toHaveBeenCalledTimes(1)
    expect(getPreviews.mock.calls[0][1]).toEqual([root.message])
    text.resolve({ previews: [treePreview(root.message)] })
    await jest.advanceTimersByTimeAsync(0)
    reads.setVisibleNodes([])
    reads.setVisibleNodes([alias])
    expect(getPreviews).toHaveBeenCalledTimes(1)
  })

  it('should drop queued offscreen work and never automatically request originals', async () => {
    const text = deferred<ConversationTreePreviewResponse>()
    const nodes = Array.from({ length: 100 }, (_: unknown, sequence: number) => treeNode(`image-${sequence}`, {
      piece_types: ['image_path', 'audio_path', 'video_path'],
      piece_count: 3,
      message: { conversation_id: 'main', sequence },
    }))
    getPreviews.mockReturnValueOnce(text.promise)
    reads.setActive(true)
    reads.setVisibleNodes(nodes)
    reads.setVisibleNodes([])
    text.resolve({ previews: nodes.slice(0, PREVIEW_BATCH_SIZE).map((node) => treePreview(node.message)) })
    await jest.advanceTimersByTimeAsync(100)
    expect(getPreviews).toHaveBeenCalledTimes(1)
    expect(getPreviews.mock.calls.some((call) => call[2] === 'full')).toBe(false)
  })

  it('should load visible thumbnails after useful text without waiting for all topology', async () => {
    const later = deferred<ConversationTreePage>()
    const image = treeNode('image', { piece_types: ['image_path'] })
    getTree.mockResolvedValueOnce(treePage([image], { complete: false, next_cursor: 'later' })).mockReturnValueOnce(later.promise)
    reads.setActive(true)
    reads.setVisibleNodes([image])
    await jest.advanceTimersByTimeAsync(32)
    expect(getPreviews.mock.calls.map((call) => call[2])).toEqual(['text', 'thumbnail'])
    expect(reads.getSnapshot().complete).toBe(false)
  })

  it('should retain topology on page failure and retry only the failed page', async () => {
    getTree.mockResolvedValueOnce(treePage([root], { complete: false, next_cursor: 'later', total_conversations: 2 }))
      .mockRejectedValueOnce(new Error('Temporary page failure'))
      .mockResolvedValueOnce(treePage([response], { processed_conversations: 2, total_conversations: 2 }))
    reads.setActive(true)
    await jest.advanceTimersByTimeAsync(32)
    expect(reads.getSnapshot().nodes.size).toBe(1)
    expect(reads.getSnapshot()).toMatchObject({ complete: false, error: 'Temporary page failure', loading: false })
    reads.retryTopology()
    await jest.advanceTimersByTimeAsync(32)
    expect(getTree.mock.calls[2][1]).toEqual({ cursor: 'later', limit: 100 })
    expect(reads.getSnapshot().nodes.size).toBe(2)
    expect(reads.getSnapshot().complete).toBe(true)
    await jest.advanceTimersByTimeAsync(10_000)
    expect(getTree).toHaveBeenCalledTimes(3)
  })

  it('should retain successful previews and retry only the failed message preview', async () => {
    getPreviews.mockRejectedValueOnce(new Error('Preview offline'))
    reads.setActive(true)
    reads.setVisibleNodes([root])
    await jest.advanceTimersByTimeAsync(0)
    expect(reads.getSnapshot().nodes.size).toBe(1)
    expect(reads.getSnapshot().previews.get(root.preview_key)?.text?.error).toBe('Preview offline')
    reads.setVisibleNodes([root])
    expect(getPreviews).toHaveBeenCalledTimes(1)
    reads.requestPreviews([root], 'text', true)
    await jest.advanceTimersByTimeAsync(0)
    expect(reads.getSnapshot().previews.get(root.preview_key)?.text?.preview?.pieces[0].text).toBe('A quiet garden')
    expect(getTree).toHaveBeenCalledTimes(1)
  })

  it('should report missing previews instead of marking them loaded', async () => {
    getPreviews.mockResolvedValueOnce({ previews: [] })
    reads.setActive(true)
    reads.requestPreviews([root])
    await jest.advanceTimersByTimeAsync(0)
    expect(reads.getSnapshot().previews.get(root.preview_key)?.text?.error).toMatch(/unavailable/i)
  })

  it('should require explicit refresh on an expired cursor and retain the old graph until replacement is useful', async () => {
    const replacement = deferred<ConversationTreePage>()
    getTree.mockResolvedValueOnce(treePage([root], { complete: false, next_cursor: 'expired' }))
      .mockRejectedValueOnce({ isAxiosError: true, response: { status: 410, data: { detail: 'The cursor has expired' } } })
      .mockReturnValueOnce(replacement.promise)
    reads.setActive(true)
    await jest.advanceTimersByTimeAsync(32)
    expect(reads.getSnapshot()).toMatchObject({ needsRefresh: true, complete: false })
    expect(getTree).toHaveBeenCalledTimes(2)
    reads.retryTopology()
    expect(reads.getSnapshot().refreshing).toBe(true)
    expect(reads.getSnapshot().nodes.has(root.node_id)).toBe(true)
    replacement.resolve(treePage([response], { revision: 'revision-2' }))
    await jest.advanceTimersByTimeAsync(0)
    expect(reads.getSnapshot().nodes.has(root.node_id)).toBe(false)
    expect(reads.getSnapshot().nodes.has(response.node_id)).toBe(true)
    expect(reads.getSnapshot()).toMatchObject({ revision: 'revision-2', refreshing: false, complete: true })
  })

  it('should reject mixed revisions and non-advancing cursors without an endless restart', async () => {
    getTree.mockResolvedValueOnce(treePage([root], { complete: false, next_cursor: 'later' }))
      .mockResolvedValueOnce(treePage([response], { revision: 'changed' }))
    reads.setActive(true)
    await jest.advanceTimersByTimeAsync(1_000)
    expect(reads.getSnapshot()).toMatchObject({ needsRefresh: true, complete: false })
    expect(reads.getSnapshot().nodes.size).toBe(1)
    expect(getTree).toHaveBeenCalledTimes(2)
    getTree.mockResolvedValueOnce(treePage([root], { complete: false, next_cursor: 'repeat' }))
      .mockResolvedValueOnce(treePage([response], { complete: false, next_cursor: 'repeat' }))
    reads.retryTopology()
    await jest.advanceTimersByTimeAsync(1_000)
    expect(reads.getSnapshot().error).toMatch(/did not advance/i)
    expect(getTree).toHaveBeenCalledTimes(4)
  })

  it('should finish its generation before coalescing refresh keys from simultaneous appends', async () => {
    const later = deferred<ConversationTreePage>()
    getTree.mockResolvedValueOnce(treePage([root], { complete: false, next_cursor: 'later' }))
      .mockReturnValueOnce(later.promise)
      .mockResolvedValueOnce(treePage([root, response], { revision: 'revision-2' }))
    reads.setActive(true)
    await jest.advanceTimersByTimeAsync(32)
    reads.refresh(1)
    reads.refresh(2)
    reads.refresh(3)
    expect(getTree).toHaveBeenCalledTimes(2)
    later.resolve(treePage([response]))
    await jest.advanceTimersByTimeAsync(64)
    expect(getTree).toHaveBeenCalledTimes(3)
    expect(reads.getSnapshot().revision).toBe('revision-2')
  })

  it('should cancel tree reads when hidden, ignore late results, and resume the same page', async () => {
    const stale = deferred<ConversationTreePage>()
    const stalePreview = deferred<ConversationTreePreviewResponse>()
    getTree.mockResolvedValueOnce(treePage([root], { complete: false, next_cursor: 'resume' }))
      .mockReturnValueOnce(stale.promise)
      .mockResolvedValueOnce(treePage([response]))
    getPreviews.mockReturnValueOnce(stalePreview.promise)
    reads.setActive(true)
    reads.setVisibleNodes([root])
    await jest.advanceTimersByTimeAsync(32)
    reads.setActive(false)
    expect(getTree.mock.calls[1][2]?.aborted).toBe(true)
    expect(getPreviews.mock.calls[0][3]?.aborted).toBe(true)
    stale.resolve(treePage([treeNode('obsolete')], { revision: 'obsolete' }))
    stalePreview.resolve({ previews: [treePreview(root.message, [treePiece({ text: 'Obsolete preview' })])] })
    await jest.advanceTimersByTimeAsync(0)
    expect(reads.getSnapshot().nodes.has('obsolete')).toBe(false)
    expect(reads.getSnapshot().previews.get(root.preview_key)?.text?.preview).toBeUndefined()
    expect(reads.getSnapshot().error).toBeUndefined()
    reads.setActive(true)
    await jest.advanceTimersByTimeAsync(32)
    expect(getTree.mock.calls[2][1]).toEqual({ cursor: 'resume', limit: 100 })
    expect(reads.getSnapshot().nodes.has(response.node_id)).toBe(true)
  })

  it('should not refetch a completed snapshot on a hide/show cycle', async () => {
    reads.setActive(true)
    await jest.advanceTimersByTimeAsync(32)
    reads.setActive(false)
    reads.setActive(true)
    await jest.advanceTimersByTimeAsync(100)
    expect(getTree).toHaveBeenCalledTimes(1)
  })

  it('should free the explicit interaction slot when a media dialog closes', async () => {
    const obsolete = deferred<ConversationTreePreviewResponse>()
    getPreviews.mockReturnValueOnce(obsolete.promise)
    reads.setActive(true)
    reads.requestPreviews([root], 'full')
    reads.cancelFullPreview(root)
    expect(getPreviews.mock.calls[0][3]?.aborted).toBe(true)
    reads.requestPreviews([response], 'full')
    expect(getPreviews).toHaveBeenCalledTimes(2)
    obsolete.resolve({ previews: [treePreview(root.message)] })
    await jest.advanceTimersByTimeAsync(0)
    expect(reads.getSnapshot().previews.get(root.preview_key)?.full?.preview).toBeUndefined()
    expect(reads.getSnapshot().previews.get(response.preview_key)?.full?.preview).toBeDefined()
  })

  it('should merge stable nodes and retain explicit prefix and empty endpoints', async () => {
    getTree.mockResolvedValueOnce(treePage([root], {
      complete: false, next_cursor: 'more', conversations: [{ conversation_id: 'prefix', node_id: root.node_id }],
    })).mockResolvedValueOnce(treePage([root, response], {
      conversations: [{ conversation_id: 'main', node_id: response.node_id }, { conversation_id: 'empty', node_id: null }],
    }))
    reads.setActive(true)
    await jest.advanceTimersByTimeAsync(32)
    expect(reads.getSnapshot().nodes.size).toBe(2)
    expect([...reads.getSnapshot().conversations.keys()]).toEqual(['prefix', 'main', 'empty'])
  })

  it('should reject topology responses for a different attack', async () => {
    getTree.mockResolvedValueOnce(treePage([root], { attack_result_id: 'different-attack' }))
    reads.setActive(true)
    await jest.advanceTimersByTimeAsync(0)
    expect(reads.getSnapshot().nodes.size).toBe(0)
    expect(reads.getSnapshot().error).toMatch(/different attack/i)
  })
})
