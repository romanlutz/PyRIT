import { useEffect, useMemo, useState } from 'react'

import { Badge, Button, Dialog, DialogBody, DialogContent, DialogSurface, DialogTitle, Input, MessageBar, MessageBarBody, Text } from '@fluentui/react-components'
import { DismissRegular, SearchRegular } from '@fluentui/react-icons'

import type { ConversationTreeEndpoint, ConversationTreeNode } from '@/types'

import { useConversationChooserStyles } from './ConversationChooser.styles'
import { isInBranch, pieceLabel, roleLabel, type TreeIndex } from './treeGraph'
import { previewPieces, type TreeReadCoordinator, type TreeReadSnapshot } from './treeReadCoordinator'
import { useTreeDialogFocus } from './useTreeDialogFocus'

const CONVERSATIONS_PER_PAGE = 32

export type ConversationChooserScope =
  | { readonly kind: 'all' }
  | { readonly kind: 'empty' }
  | { readonly kind: 'branch'; readonly nodeId: string; readonly endpointsOnly: boolean }

interface ConversationChooserProps {
  scope: ConversationChooserScope
  index: TreeIndex
  snapshot: TreeReadSnapshot
  reads: TreeReadCoordinator
  activeConversationId: string | null
  opener: HTMLElement | null
  onSelect: (id: string) => void
  onClose: () => void
}

function conversationContext(endpoint: ConversationTreeEndpoint, snapshot: TreeReadSnapshot): string {
  if (endpoint.node_id === null) return 'Empty conversation'
  const node = snapshot.nodes.get(endpoint.node_id)
  if (!node) return 'Message structure is still loading'
  const previews = snapshot.previews.get(node.preview_key)
  const text = previewPieces(previews).map((piece) => piece.text || piece.filename || pieceLabel(piece.data_type)).join(' ')
  if (text) return `${roleLabel(node.role)}: ${text}`
  if (previews?.text?.error) return 'Preview unavailable. The conversation can still be opened.'
  return `${roleLabel(node.role)}: ${node.piece_types.map(pieceLabel).join(', ')}. Preview not loaded.`
}

export default function ConversationChooser({
  scope, index, snapshot, reads, activeConversationId, opener, onSelect, onClose,
}: ConversationChooserProps) {
  const styles = useConversationChooserStyles()
  const [query, setQuery] = useState('')
  const [limit, setLimit] = useState(CONVERSATIONS_PER_PAGE)
  useTreeDialogFocus(opener)
  const results = useMemo(() => {
    const search = query.trim().toLocaleLowerCase()
    return [...snapshot.conversations.values()].filter((endpoint: ConversationTreeEndpoint) => {
      if (scope.kind === 'empty' && endpoint.node_id !== null) return false
      if (scope.kind === 'branch') {
        if (endpoint.node_id === null) return false
        if (scope.endpointsOnly ? endpoint.node_id !== scope.nodeId : !isInBranch(index, endpoint.node_id, scope.nodeId)) return false
      }
      return !search || `${endpoint.conversation_id} ${conversationContext(endpoint, snapshot)}`.toLocaleLowerCase().includes(search)
    }).sort((left: ConversationTreeEndpoint, right: ConversationTreeEndpoint) => {
      const priority = (endpoint: ConversationTreeEndpoint): number =>
        endpoint.conversation_id === activeConversationId ? 0 : endpoint.conversation_id === snapshot.mainConversationId ? 1 : 2
      return priority(left) - priority(right) || left.conversation_id.localeCompare(right.conversation_id)
    })
  }, [query, scope, index, snapshot, activeConversationId])
  const shown = useMemo(() => results.slice(0, limit), [results, limit])

  useEffect(() => {
    const nodes = shown.map((endpoint: ConversationTreeEndpoint) => endpoint.node_id ? snapshot.nodes.get(endpoint.node_id) : undefined)
      .filter((node: ConversationTreeNode | undefined): node is ConversationTreeNode => node !== undefined)
    reads.requestPreviews(nodes)
  }, [shown, snapshot.nodes, reads])

  return (
    <Dialog open onOpenChange={(_, data) => { if (!data.open) onClose() }}>
      <DialogSurface className={styles.surface}>
        <DialogBody>
          <DialogTitle action={<Button appearance="subtle" icon={<DismissRegular />} className={styles.action} aria-label="Close conversation chooser" onClick={onClose} />}>
            {scope.kind === 'empty' ? 'Empty conversations' : 'Choose a conversation'}
          </DialogTitle>
          <DialogContent className={styles.content}>
            <Input
              autoFocus
              aria-label="Search conversations"
              contentBefore={<SearchRegular />}
              placeholder="Search IDs or loaded preview text"
              value={query}
              onChange={(_, data) => { setQuery(data.value); setLimit(CONVERSATIONS_PER_PAGE) }}
            />
            {!snapshot.complete && <MessageBar intent="info"><MessageBarBody>More conversations are still loading. These are the known options, not a complete list.</MessageBarBody></MessageBar>}
            <Text size={200}>Search includes conversation IDs and previews already loaded.</Text>
            <div className={styles.results}>
              <ul className={styles.list} aria-label="Matching conversations">
                {shown.map((endpoint: ConversationTreeEndpoint) => (
                  <li key={endpoint.conversation_id}>
                    <Button
                      appearance="subtle"
                      className={styles.row}
                      title={endpoint.conversation_id}
                      aria-label={`Open conversation ${endpoint.conversation_id}`}
                      onClick={() => { onSelect(endpoint.conversation_id) }}
                    >
                      <span className={styles.heading}>
                        <Text className={styles.id}>{endpoint.conversation_id.slice(0, 12)}</Text>
                        {endpoint.conversation_id === activeConversationId && <Badge appearance="tint" size="small">Current</Badge>}
                        {endpoint.conversation_id === snapshot.mainConversationId && <Badge appearance="outline" size="small">Main</Badge>}
                        {scope.kind === 'branch' && endpoint.node_id === scope.nodeId && <Text size={200}>Ends at this message</Text>}
                      </span>
                      <Text size={200} className={styles.context}>{conversationContext(endpoint, snapshot)}</Text>
                    </Button>
                  </li>
                ))}
              </ul>
              {results.length === 0 && <Text>{snapshot.complete ? 'No matching conversations.' : 'No matching conversations discovered yet.'}</Text>}
              {shown.length < results.length && <Button className={styles.more} onClick={() => { setLimit((previous: number) => previous + CONVERSATIONS_PER_PAGE) }}>Show more conversations ({results.length - shown.length} remaining)</Button>}
            </div>
            <Text size={200}>{shown.length} of {results.length} known matches shown</Text>
          </DialogContent>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  )
}
