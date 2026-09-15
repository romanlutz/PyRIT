import { useState } from 'react'

import {
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  Link,
  MessageBar,
  MessageBarBody,
  Spinner,
  Text,
} from '@fluentui/react-components'
import { ArrowSyncRegular, DismissRegular } from '@fluentui/react-icons'

import { ImageWithSpinner, MediaWithFallback } from '@/components/Chat/MediaPreview'
import type { ConversationTreeNode } from '@/types'

import { useMediaLightboxStyles } from './MediaLightbox.styles'
import { mediaType, pieceHasError, pieceLabel } from './treeGraph'
import type { PreviewResult } from './treeReadCoordinator'
import { useTreeDialogFocus } from './useTreeDialogFocus'

interface MediaLightboxProps {
  node: ConversationTreeNode
  pieceIndex: number
  result?: PreviewResult
  opener: HTMLElement | null
  onClose: () => void
  onRetry: () => void
}

export default function MediaLightbox({ node, pieceIndex, result, opener, onClose, onRetry }: MediaLightboxProps) {
  const styles = useMediaLightboxStyles()
  const [attempt, setAttempt] = useState(0)
  useTreeDialogFocus(opener)
  const piece = result?.preview?.pieces[pieceIndex]
  const dataType = piece?.data_type ?? node.piece_types[pieceIndex] ?? 'file'
  const type = mediaType(dataType)
  const title = piece?.filename ?? `${pieceLabel(dataType)} ${pieceIndex + 1}`
  const loading = !result || result.loading

  return (
    <Dialog open onOpenChange={(_, data) => { if (!data.open) onClose() }}>
      <DialogSurface className={styles.surface} aria-label={title} data-testid="tree-media-lightbox">
        <DialogBody className={styles.body}>
          <DialogTitle
            className={styles.title}
            action={<Button autoFocus appearance="subtle" icon={<DismissRegular />} className={styles.action} aria-label="Close media" onClick={onClose} />}
          >
            {title}
          </DialogTitle>
          <DialogContent className={styles.content}>
            {loading && <Spinner size="small" label={`Loading full ${type}`} />}
            {!loading && result?.error && <MessageBar intent="error"><MessageBarBody>{result.error}</MessageBarBody></MessageBar>}
            {!loading && !result?.error && !piece?.media_url && (
              <MessageBar intent="warning"><MessageBarBody>This media is unavailable. Retry, or open its conversation to inspect the stored message.</MessageBarBody></MessageBar>
            )}
            {!loading && piece && pieceHasError(piece) && (
              <MessageBar intent="warning"><MessageBarBody>Stored error: {piece.response_error}</MessageBarBody></MessageBar>
            )}
            {!loading && !result?.error && piece?.media_url && (
              type === 'image'
                ? <ImageWithSpinner
                    key={`${piece.media_url}:${attempt}`}
                    src={piece.media_url}
                    alt={title}
                    className={styles.image}
                    hiddenClassName={styles.hiddenImage}
                    containerClassName={styles.imageContainer}
                    spinnerClassName={styles.spinner}
                  />
                : type === 'audio' || type === 'video'
                  ? <MediaWithFallback key={`${piece.media_url}:${attempt}`} type={type} src={piece.media_url} preload="metadata" stopOnUnmount className={styles.player} />
                  : <Link href={piece.media_url} target="_blank" rel="noopener noreferrer">Open {title}</Link>
            )}
            {!loading && piece?.mime_type && <Text size={200}>{piece.mime_type}</Text>}
          </DialogContent>
          <DialogActions>
            <Button
              icon={<ArrowSyncRegular />}
              disabled={loading}
              className={styles.action}
              onClick={() => { setAttempt((previous: number) => previous + 1); onRetry() }}
            >
              Reload media
            </Button>
          </DialogActions>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  )
}
