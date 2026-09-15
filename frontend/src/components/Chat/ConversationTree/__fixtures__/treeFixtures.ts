import type {
  ConversationTreeNode,
  ConversationTreePage,
  ConversationTreePiecePreview,
  ConversationTreePreview,
  TreeMessageReference,
} from '@/types'

export function treeNode(id: string, overrides: Partial<ConversationTreeNode> = {}): ConversationTreeNode {
  return {
    node_id: id,
    parent_node_id: null,
    message: { conversation_id: 'main', sequence: 0 },
    role: 'user',
    piece_types: ['text'],
    piece_count: 1,
    preview_key: `preview-${id}`,
    created_at: '2026-09-14T12:00:00Z',
    ...overrides,
  }
}

export function treePage(nodes: ConversationTreeNode[], overrides: Partial<ConversationTreePage> = {}): ConversationTreePage {
  return {
    attack_result_id: 'attack',
    main_conversation_id: 'main',
    revision: 'revision-1',
    nodes,
    conversations: [{ conversation_id: 'main', node_id: nodes[nodes.length - 1]?.node_id ?? null }],
    processed_conversations: 1,
    total_conversations: 1,
    next_cursor: null,
    complete: true,
    ...overrides,
  }
}

export function treePiece(overrides: Partial<ConversationTreePiecePreview> = {}): ConversationTreePiecePreview {
  return {
    piece_id: 'piece',
    data_type: 'text',
    text: 'A quiet garden',
    truncated: false,
    media_url: null,
    thumbnail_url: null,
    mime_type: null,
    filename: null,
    response_error: 'none',
    ...overrides,
  }
}

export function treePreview(message: TreeMessageReference, pieces: ConversationTreePiecePreview[] = [treePiece()]): ConversationTreePreview {
  return { message, pieces }
}

export function deferred<T>() {
  let resolvePromise: ((value: T) => void) | undefined
  let rejectPromise: ((error: unknown) => void) | undefined
  const promise = new Promise<T>((resolve, reject) => { resolvePromise = resolve; rejectPromise = reject })
  return {
    promise,
    resolve: (value: T): void => {
      if (!resolvePromise) throw new Error('Deferred promise has not been initialized')
      resolvePromise(value)
    },
    reject: (error: unknown): void => {
      if (!rejectPromise) throw new Error('Deferred promise has not been initialized')
      rejectPromise(error)
    },
  }
}
