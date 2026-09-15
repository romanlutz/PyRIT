import React, { useEffect, useRef } from 'react'

import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { Viewport } from '@xyflow/react'

import { attacksApi } from '@/services/api'
import type { ConversationTreePage, ConversationTreePreviewResponse, TreeMessageReference } from '@/types'

import { deferred, treeNode, treePage, treePiece, treePreview } from './__fixtures__/treeFixtures'
import ConversationTree from './ConversationTree'
import ConversationTreeNode, { type MessageFlowNode } from './ConversationTreeNode'

jest.mock('@/services/api', () => ({
  attacksApi: { getConversationTree: jest.fn(), getTreePreviews: jest.fn() },
}))

jest.mock('./useTreeViewport', () => ({
  ...jest.requireActual<typeof import('./useTreeViewport')>('./useTreeViewport'),
  useTreePaneSize: () => ({ width: 1200, height: 800 }),
}))

interface MockFlowProps {
  nodes: MessageFlowNode[]
  onInit: () => void
  onMoveEnd: (event: null, viewport: Viewport) => void
}

const mockGraph = jest.fn()
let mockViewport: Viewport = { x: 0, y: 0, zoom: 1 }
const mockFlow = {
  fitView: jest.fn(async () => true),
  zoomIn: jest.fn(async () => true),
  zoomOut: jest.fn(async () => true),
  getViewport: jest.fn(() => mockViewport),
}

function MockReactFlow(props: MockFlowProps) {
  const initialized = useRef(false)
  mockGraph(props)
  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true
      props.onInit()
    }
  }, [props])
  return (
    <div aria-label="Read-only conversation message graph">
      <button onClick={() => { mockViewport = { x: 80, y: 40, zoom: 1.25 }; props.onMoveEnd(null, mockViewport) }}>Pan graph</button>
      {props.nodes.map((node: MessageFlowNode) => <ConversationTreeNode key={node.id} data={node.data} />)}
    </div>
  )
}

jest.mock('@xyflow/react', () => ({
  ReactFlowProvider: ({ children }: { children: React.ReactNode }) => children,
  ReactFlow: MockReactFlow,
  useReactFlow: () => mockFlow,
  Handle: () => null,
  Position: { Top: 'top', Bottom: 'bottom' },
}))

const getTree = jest.mocked(attacksApi.getConversationTree)
const getPreviews = jest.mocked(attacksApi.getTreePreviews)
const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
)

describe('ConversationTree', () => {
  const root = treeNode('root')
  const response = treeNode('response', { parent_node_id: 'root', role: 'assistant', message: { conversation_id: 'main', sequence: 1 } })
  const defaultProps = { attackResultId: 'attack', activeConversationId: 'main', onSelectConversation: jest.fn() }

  beforeEach(() => {
    jest.clearAllMocks()
    mockViewport = { x: 0, y: 0, zoom: 1 }
    getTree.mockReset().mockResolvedValue(treePage([root, response]))
    getPreviews.mockReset().mockImplementation(async (_attack: string, messages: TreeMessageReference[]) => ({
      previews: messages.map((message: TreeMessageReference) => treePreview(message)),
    }))
  })

  afterEach(() => { jest.restoreAllMocks() })

  it('should render and select useful nodes before later pages, text, or media resolve', async () => {
    const user = userEvent.setup()
    const later = deferred<ConversationTreePage>()
    const text = deferred<ConversationTreePreviewResponse>()
    getTree.mockResolvedValueOnce(treePage([root], {
      complete: false, next_cursor: 'later', processed_conversations: 1, total_conversations: 12,
    })).mockReturnValueOnce(later.promise)
    getPreviews.mockReturnValue(text.promise)
    render(<TestWrapper><ConversationTree {...defaultProps} /></TestWrapper>)
    expect(await screen.findByRole('article', { name: /user message 0, 1 pieces/i })).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('Loading more branches; 1 of 12 conversations')
    expect(screen.queryByText(/no messages are stored/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/no response stored/i)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Open conversation main' }))
    expect(defaultProps.onSelectConversation).toHaveBeenCalledWith('main')
  })

  it('should group all message pieces in one node with bounded progressive overflow', async () => {
    const user = userEvent.setup()
    const multipart = treeNode('multipart', {
      piece_count: 20,
      piece_types: Array.from({ length: 20 }, () => 'image_path'),
    })
    const pending = deferred<ConversationTreePreviewResponse>()
    getTree.mockResolvedValueOnce(treePage([multipart]))
    getPreviews.mockReturnValue(pending.promise)
    render(<TestWrapper><ConversationTree {...defaultProps} /></TestWrapper>)
    const node = await screen.findByRole('article', { name: /20 pieces/i })
    expect(screen.getAllByRole('article')).toHaveLength(1)
    expect(within(node).getAllByRole('button', { name: /open image/i })).toHaveLength(3)
    await user.click(within(node).getByRole('button', { name: 'View all 20 pieces' }))
    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getAllByRole('listitem')).toHaveLength(16)
    await user.click(within(dialog).getByRole('button', { name: /show more pieces/i }))
    expect(within(dialog).getAllByRole('listitem')).toHaveLength(20)
    expect(getPreviews.mock.calls.some((call) => call[2] === 'full')).toBe(false)
  })

  it('should not pick an arbitrary descendant of a shared message and should preserve prefix endpoints', async () => {
    const user = userEvent.setup()
    const later = deferred<ConversationTreePage>()
    getTree.mockResolvedValueOnce(treePage([root, response], {
      conversations: [
        { conversation_id: 'prefix', node_id: 'root' },
        { conversation_id: 'main', node_id: 'response' },
        { conversation_id: 'empty', node_id: null },
      ],
      complete: false, next_cursor: 'later', processed_conversations: 3, total_conversations: 4,
    })).mockReturnValueOnce(later.promise)
    render(<TestWrapper><ConversationTree {...defaultProps} /></TestWrapper>)
    await user.click(await screen.findByRole('button', { name: /choose conversation for user message 0/i }))
    expect(defaultProps.onSelectConversation).not.toHaveBeenCalled()
    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText(/more conversations are still loading/i)).toBeInTheDocument()
    expect(within(dialog).getByText(/ends at this message/i)).toBeInTheDocument()
    expect(within(dialog).queryByRole('button', { name: 'Open conversation empty' })).not.toBeInTheDocument()
    await user.type(within(dialog).getByRole('textbox', { name: /search conversations/i }), 'prefix')
    await user.click(within(dialog).getByRole('button', { name: 'Open conversation prefix' }))
    expect(defaultProps.onSelectConversation).toHaveBeenCalledWith('prefix')
  })

  it('should expose empty conversations separately without creating fake message nodes', async () => {
    const user = userEvent.setup()
    getTree.mockResolvedValueOnce(treePage([], {
      conversations: [{ conversation_id: 'main', node_id: null }, { conversation_id: 'empty', node_id: null }],
      processed_conversations: 2, total_conversations: 2,
    }))
    render(<TestWrapper><ConversationTree {...defaultProps} /></TestWrapper>)
    const empty = await screen.findByRole('region', { name: /empty conversations/i })
    expect(screen.queryAllByRole('article')).toHaveLength(0)
    await user.click(within(empty).getByRole('button', { name: 'Open empty conversation empty' }))
    expect(defaultProps.onSelectConversation).toHaveBeenCalledWith('empty')
  })

  it('should keep the graph on page errors and offer a targeted retry', async () => {
    const user = userEvent.setup()
    getTree.mockResolvedValueOnce(treePage([root], { complete: false, next_cursor: 'retry' }))
      .mockRejectedValueOnce(new Error('Branch page unavailable'))
      .mockResolvedValueOnce(treePage([response]))
    render(<TestWrapper><ConversationTree {...defaultProps} /></TestWrapper>)
    expect(await screen.findByText('Branch page unavailable')).toBeInTheDocument()
    expect(screen.getByRole('article', { name: /user message/i })).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('Partial tree')
    await user.click(screen.getByRole('button', { name: /retry loading branches/i }))
    expect(await screen.findByRole('article', { name: /assistant message/i })).toBeInTheDocument()
    expect(getTree.mock.calls[2][1]).toEqual({ cursor: 'retry', limit: 100 })
  })

  it('should use thumbnails only and prioritize explicit full media before remaining structure finishes', async () => {
    const user = userEvent.setup()
    const later = deferred<ConversationTreePage>()
    const full = deferred<ConversationTreePreviewResponse>()
    const image = treeNode('image', { piece_types: ['image_path'] })
    getTree.mockResolvedValueOnce(treePage([image], { complete: false, next_cursor: 'later' })).mockReturnValueOnce(later.promise)
    getPreviews.mockImplementation(async (_attack: string, messages: TreeMessageReference[], level) => {
      if (level === 'full') return full.promise
      return {
        previews: messages.map((message: TreeMessageReference) => treePreview(message, [treePiece({
          data_type: 'image_path', text: null, filename: 'garden.png',
          thumbnail_url: level === 'thumbnail' ? 'https://example.test/thumb.png' : null,
        })])),
      }
    })
    render(<TestWrapper><ConversationTree {...defaultProps} /></TestWrapper>)
    fireEvent.load(await screen.findByAltText('garden.png thumbnail'))
    expect(screen.getByRole('img', { name: /garden.png thumbnail/i })).toHaveAttribute('src', 'https://example.test/thumb.png')
    expect(getPreviews.mock.calls.some((call) => call[2] === 'full')).toBe(false)
    const opener = screen.getByRole('button', { name: /open image 1/i })
    await user.click(opener)
    expect(getPreviews.mock.calls.some((call) => call[2] === 'full')).toBe(true)
    expect(screen.getByText(/loading full image/i)).toBeInTheDocument()
    expect(screen.getByRole('status', { hidden: true })).toHaveTextContent('Loading more branches')
    await act(async () => {
      full.resolve({ previews: [treePreview(image.message, [treePiece({
        data_type: 'image_path', text: null, filename: 'garden.png', media_url: 'https://example.test/original.png',
      })])] })
    })
    fireEvent.load(await screen.findByAltText('garden.png'))
    expect(screen.getByRole('img', { name: 'garden.png' })).toHaveAttribute('src', 'https://example.test/original.png')
    await user.click(screen.getByRole('button', { name: /close media/i }))
    await waitFor(() => { expect(opener).toHaveFocus() })
    expect(defaultProps.onSelectConversation).not.toHaveBeenCalled()
  })

  it('should fit only once, preserve viewport across previews and hide/show, and disable graph editing', async () => {
    const user = userEvent.setup()
    const text = deferred<ConversationTreePreviewResponse>()
    getPreviews.mockReturnValueOnce(text.promise)
    const { rerender } = render(<TestWrapper><ConversationTree {...defaultProps} /></TestWrapper>)
    jest.spyOn(screen.getByTestId('conversation-tree-pane'), 'getBoundingClientRect')
      .mockReturnValue(new DOMRect(0, 0, 1200, 800))
    await waitFor(() => { expect(mockFlow.fitView).toHaveBeenCalledTimes(1) })
    await user.click(screen.getByRole('button', { name: 'Pan graph' }))
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('125%')
    await act(async () => { text.resolve({ previews: [treePreview(root.message)] }) })
    expect(mockFlow.fitView).toHaveBeenCalledTimes(1)
    rerender(<TestWrapper><ConversationTree {...defaultProps} active={false} /></TestWrapper>)
    expect(screen.queryByRole('region', { name: 'Conversation tree' })).not.toBeInTheDocument()
    rerender(<TestWrapper><ConversationTree {...defaultProps} active /></TestWrapper>)
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('125%')
    expect(getTree).toHaveBeenCalledTimes(1)
    expect(mockFlow.fitView).toHaveBeenCalledTimes(1)
    await user.click(screen.getByRole('button', { name: /fit to view/i }))
    expect(mockFlow.fitView).toHaveBeenCalledTimes(2)
    expect(mockGraph).toHaveBeenLastCalledWith(expect.objectContaining({
      nodesDraggable: false, nodesConnectable: false, edgesReconnectable: false,
      elementsSelectable: false, deleteKeyCode: null, selectionOnDrag: false, onlyRenderVisibleElements: true,
    }))
  })

  it('should keep every loaded branch expanded across toggles and retain conversation search', async () => {
    const user = userEvent.setup()
    const branch = Array.from({ length: 14 }, (_: unknown, index: number) => treeNode(`other-${index}`, {
      parent_node_id: index === 0 ? 'root' : `other-${index - 1}`,
      message: { conversation_id: 'other', sequence: index + 1 },
    }))
    getTree.mockResolvedValueOnce(treePage([root, response, ...branch], {
      conversations: [{ conversation_id: 'main', node_id: 'response' }, { conversation_id: 'other', node_id: 'other-13' }],
      total_conversations: 2, processed_conversations: 2,
    }))
    const { rerender } = render(<TestWrapper><ConversationTree {...defaultProps} /></TestWrapper>)
    expect(await screen.findByTestId('tree-message-other-13')).toBeInTheDocument()
    expect(screen.getAllByRole('article')).toHaveLength(16)
    expect(screen.queryByRole('button', { name: /(?:expand|collapse) branch/i })).not.toBeInTheDocument()
    const opener = screen.getByRole('button', { name: /conversations \(2\)/i })
    await user.click(opener)
    const dialog = screen.getByRole('dialog')
    await user.type(within(dialog).getByRole('textbox', { name: /search conversations/i }), 'other')
    expect(within(dialog).getByRole('button', { name: 'Open conversation other' })).toBeInTheDocument()
    await waitFor(() => { expect(screen.queryByRole('region', { name: 'Conversation tree' })).not.toBeInTheDocument() })
    await user.click(within(dialog).getByRole('button', { name: /close conversation chooser/i }))
    expect(await screen.findByRole('region', { name: 'Conversation tree' })).toBeInTheDocument()
    await waitFor(() => { expect(opener).toHaveFocus() })
    expect(screen.getByTestId('tree-message-other-13')).toBeInTheDocument()
    rerender(<TestWrapper><ConversationTree {...defaultProps} active={false} /></TestWrapper>)
    rerender(<TestWrapper><ConversationTree {...defaultProps} active /></TestWrapper>)
    expect(screen.getAllByRole('article')).toHaveLength(16)
    expect(screen.getByTestId('tree-message-other-13')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /(?:expand|collapse) branch/i })).not.toBeInTheDocument()
    expect(mockGraph).toHaveBeenLastCalledWith(expect.objectContaining({ onlyRenderVisibleElements: true }))
  })

  it('should abort obsolete attack reads and ignore their eventual responses', async () => {
    const first = deferred<ConversationTreePage>()
    getTree.mockReturnValueOnce(first.promise).mockResolvedValueOnce(treePage([response], { attack_result_id: 'next' }))
    const { rerender } = render(<TestWrapper><ConversationTree {...defaultProps} /></TestWrapper>)
    await waitFor(() => { expect(getTree).toHaveBeenCalledTimes(1) })
    rerender(<TestWrapper><ConversationTree {...defaultProps} attackResultId="next" /></TestWrapper>)
    expect(getTree.mock.calls[0][2]?.aborted).toBe(true)
    expect(await screen.findByRole('article', { name: /assistant message/i })).toBeInTheDocument()
    await act(async () => { first.resolve(treePage([root])) })
    expect(screen.queryByRole('article', { name: /user message/i })).not.toBeInTheDocument()
  })

  it('should keep newly arriving messages expanded without branch controls', async () => {
    const later = deferred<ConversationTreePage>()
    const branch = Array.from({ length: 14 }, (_: unknown, index: number) => treeNode(`growing-${index}`, {
      parent_node_id: index === 0 ? 'root' : `growing-${index - 1}`,
      message: { conversation_id: 'other', sequence: index + 1 },
    }))
    getTree.mockResolvedValueOnce(treePage([root, response, branch[0], branch[1]], {
      complete: false, next_cursor: 'later',
    })).mockReturnValueOnce(later.promise)
    render(<TestWrapper><ConversationTree {...defaultProps} /></TestWrapper>)
    expect(await screen.findByTestId('tree-message-growing-1')).toBeInTheDocument()
    await act(async () => {
      later.resolve(treePage(branch.slice(2), {
        conversations: [{ conversation_id: 'other', node_id: 'growing-13' }],
      }))
    })
    expect(await screen.findByTestId('tree-message-growing-13')).toBeInTheDocument()
    expect(screen.getAllByRole('article')).toHaveLength(16)
    expect(screen.queryByRole('button', { name: /(?:expand|collapse) branch/i })).not.toBeInTheDocument()
  })

  it('should defer all tree requests when initially inactive and abort reads on hiding', async () => {
    const pending = deferred<ConversationTreePage>()
    getTree.mockReturnValueOnce(pending.promise)
    const { rerender } = render(<TestWrapper><ConversationTree {...defaultProps} active={false} /></TestWrapper>)
    expect(getTree).not.toHaveBeenCalled()
    rerender(<TestWrapper><ConversationTree {...defaultProps} active /></TestWrapper>)
    await waitFor(() => { expect(getTree).toHaveBeenCalledTimes(1) })
    rerender(<TestWrapper><ConversationTree {...defaultProps} active={false} refreshKey={1} /></TestWrapper>)
    expect(getTree.mock.calls[0][2]?.aborted).toBe(true)
    expect(getTree).toHaveBeenCalledTimes(1)
  })
})
