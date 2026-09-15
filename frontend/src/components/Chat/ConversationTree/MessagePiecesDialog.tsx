import { useState } from 'react'

import { Button, Dialog, DialogBody, DialogContent, DialogSurface, DialogTitle, MessageBar, MessageBarBody, Text, useRestoreFocusTarget } from '@fluentui/react-components'
import { DismissRegular } from '@fluentui/react-icons'

import type { ConversationTreeNode } from '@/types'

import { useMessagePiecesDialogStyles } from './MessagePiecesDialog.styles'
import { mediaType, pieceHasError, pieceLabel, roleLabel } from './treeGraph'
import { previewPieces, type NodePreviews } from './treeReadCoordinator'
import { useTreeDialogFocus } from './useTreeDialogFocus'

const PIECES_PER_PAGE = 16

interface MessagePiecesDialogProps {
  node: ConversationTreeNode
  previews?: NodePreviews
  opener: HTMLElement | null
  onClose: () => void
  onMedia: (nodeId: string, index: number, opener: HTMLElement) => void
  onRetry: (node: ConversationTreeNode) => void
}

export default function MessagePiecesDialog({ node, previews, opener, onClose, onMedia, onRetry }: MessagePiecesDialogProps) {
  const styles = useMessagePiecesDialogStyles()
  const restoreFocusTargetAttributes = useRestoreFocusTarget()
  const [limit, setLimit] = useState(PIECES_PER_PAGE)
  useTreeDialogFocus(opener)
  const pieces = previewPieces(previews)
  const visibleCount = Math.min(limit, node.piece_count)

  return (
    <Dialog open onOpenChange={(_, data) => { if (!data.open) onClose() }}>
      <DialogSurface className={styles.surface}>
        <DialogBody>
          <DialogTitle action={<Button autoFocus appearance="subtle" icon={<DismissRegular />} className={styles.action} aria-label="Close message details" onClick={onClose} />}>
            {roleLabel(node.role)} message: {node.piece_count} pieces
          </DialogTitle>
          <DialogContent className={styles.content}>
            <Text size={200}>Message previews are shortened. Open a conversation for the full text.</Text>
            {previews?.text?.error && <MessageBar intent="error"><MessageBarBody>{previews.text.error}</MessageBarBody><Button className={styles.action} onClick={() => { onRetry(node) }}>Retry preview</Button></MessageBar>}
            <ol className={styles.list}>
              {Array.from({ length: visibleCount }, (_: unknown, index: number) => {
                const piece = pieces[index]
                const type = piece?.data_type ?? node.piece_types[index] ?? node.piece_types[0] ?? 'file'
                const label = pieceLabel(type)
                return (
                  <li key={`${node.node_id}:${index}`} className={styles.piece}>
                    <Text weight="semibold">{piece?.filename ?? label}</Text>
                    {piece?.text && <p>{piece.text}{piece.truncated ? '...' : ''}</p>}
                    {!piece && <p>Preview not loaded</p>}
                    {piece && pieceHasError(piece) && <MessageBar intent="warning"><MessageBarBody>{piece.response_error}</MessageBarBody></MessageBar>}
                    {(mediaType(type) !== 'file' || type.endsWith('_path')) && (
                      <Button {...restoreFocusTargetAttributes} className={styles.action} onClick={(event) => { onMedia(node.node_id, index, event.currentTarget) }}>
                        Open {label.toLowerCase()} {index + 1}
                      </Button>
                    )}
                  </li>
                )
              })}
            </ol>
            {visibleCount < node.piece_count && <Button className={styles.more} onClick={() => { setLimit((previous: number) => previous + PIECES_PER_PAGE) }}>Show more pieces ({node.piece_count - visibleCount} remaining)</Button>}
          </DialogContent>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  )
}
