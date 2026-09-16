import { memo } from 'react'

import { Badge, Button, Skeleton, SkeletonItem, Text, mergeClasses, useRestoreFocusTarget } from '@fluentui/react-components'
import {
  ArrowExpandRegular,
  ArrowSyncRegular,
  ChatMultipleRegular,
} from '@fluentui/react-icons'
import { Handle, Position, type Node } from '@xyflow/react'

import { ImageWithSpinner } from '@/components/Chat/MediaPreview'
import type { ConversationTreeNode as TreeNode } from '@/types'

import { useConversationTreeNodeStyles } from './ConversationTreeNode.styles'
import { COMPACT_PIECE_COUNT, mediaType, pieceHasError, pieceLabel, roleLabel } from './treeGraph'
import { previewPieces, type NodePreviews } from './treeReadCoordinator'

export interface ConversationTreeNodeData extends Record<string, unknown> {
  readonly message: TreeNode
  readonly previews?: NodePreviews
  readonly currentPath: boolean
  readonly currentEndpoint: boolean
  readonly mainEndpoint: boolean
  readonly endpointIds: string[]
  readonly hasChildren: boolean
  readonly conversationCount: number
  readonly complete: boolean
  readonly onChoose: (nodeId: string, opener: HTMLElement, endpointsOnly?: boolean) => void
  readonly onSelect: (conversationId: string) => void
  readonly onMedia: (nodeId: string, pieceIndex: number, opener: HTMLElement) => void
  readonly onPieces: (nodeId: string, opener: HTMLElement) => void
  readonly onRetry: (node: TreeNode) => void
  readonly onFocus: (nodeId: string) => void
}

export type MessageFlowNode = Node<ConversationTreeNodeData, 'message'>

interface ConversationTreeNodeProps {
  data: ConversationTreeNodeData
}

function ConversationTreeNode({ data }: ConversationTreeNodeProps) {
  const styles = useConversationTreeNodeStyles()
  const restoreFocusTargetAttributes = useRestoreFocusTarget()
  const node = data.message
  const pieces = previewPieces(data.previews)
  const count = Math.min(node.piece_count, COMPACT_PIECE_COUNT)
  const error = data.previews?.text?.error ?? data.previews?.thumbnail?.error
  const role = roleLabel(node.role)

  return (
    <article
      aria-label={`${role} message ${node.message.sequence}, ${node.piece_count} pieces`}
      data-testid={`tree-message-${node.node_id}`}
      data-sequence={node.message.sequence}
      data-parent-node-id={node.parent_node_id ?? undefined}
      className={mergeClasses(styles.root, node.piece_count === 1 && styles.singlePiece, node.piece_count === 2 && styles.twoPieces, node.role === 'user' && styles.user, data.currentPath && styles.currentPath, 'nodrag nopan')}
      onFocus={() => { data.onFocus(node.node_id) }}
    >
      <Handle type="target" position={Position.Top} isConnectable={false} className={styles.handle} />
      <div className={styles.header}>
        <div className={styles.title}>
          <Text weight="semibold">{role}</Text>
          <Text size={200} className={styles.detail}>{node.piece_count} {node.piece_count === 1 ? 'piece' : 'pieces'}</Text>
        </div>
        <Button
          {...restoreFocusTargetAttributes}
          appearance="subtle"
          className={styles.button}
          icon={<ChatMultipleRegular />}
          aria-label={`Choose conversation for ${role} message ${node.message.sequence}`}
          title={`${data.conversationCount}${data.complete ? '' : '+'} known conversations`}
          onClick={(event) => { event.stopPropagation(); data.onChoose(node.node_id, event.currentTarget) }}
        />
      </div>
      <div className={styles.markers}>
        {data.currentPath && <Badge appearance="tint" size="small">{data.currentEndpoint ? 'Current conversation' : 'Current path'}</Badge>}
        {data.mainEndpoint && <Badge appearance="outline" size="small">Main</Badge>}
      </div>
      <div className={styles.pieces}>
        {Array.from({ length: count }, (_: unknown, index: number) => {
          const piece = pieces[index]
          const type = piece?.data_type ?? node.piece_types[index] ?? node.piece_types[0] ?? 'file'
          const label = pieceLabel(type)
          const media = mediaType(type)
          return (
            <div className={styles.piece} key={`${node.node_id}:${index}`} data-testid={`tree-piece-${node.node_id}-${index}`}>
              {piece?.thumbnail_url && (media === 'image' || media === 'video') && (
                <ImageWithSpinner
                  key={piece.thumbnail_url}
                  src={piece.thumbnail_url}
                  alt={`${piece.filename ?? label} thumbnail`}
                  className={styles.thumbnail}
                  hiddenClassName={styles.hiddenImage}
                  containerClassName={styles.thumbnailContainer}
                  spinnerClassName={styles.spinner}
                />
              )}
              <div className={styles.pieceText}>
                {piece
                  ? <Text size={200} className={pieceHasError(piece) ? styles.error : undefined}>
                      {pieceHasError(piece) ? `Stored error: ${piece.response_error}. ` : ''}
                      {piece.text || piece.filename || label}{piece.truncated ? '...' : ''}
                    </Text>
                  : <>
                      <Text size={200}>{label}</Text>
                      <Skeleton animation="pulse" aria-label={`${label} preview not loaded`} className={styles.skeleton}>
                        <SkeletonItem size={8} />
                      </Skeleton>
                    </>
                }
              </div>
              {media !== 'file' && (
                <Button
                  {...restoreFocusTargetAttributes}
                  appearance="subtle"
                  size="small"
                  icon={<ArrowExpandRegular />}
                  className={styles.button}
                  aria-label={`Open ${media} ${index + 1}${piece?.filename ? `: ${piece.filename}` : ''}`}
                  onPointerDown={(event) => { event.stopPropagation() }}
                  onClick={(event) => { event.stopPropagation(); data.onMedia(node.node_id, index, event.currentTarget) }}
                />
              )}
            </div>
          )
        })}
      </div>
      {error && (
        <div className={styles.retryRow}>
          <Text size={200} className={mergeClasses(styles.pieceText, styles.error)} title={error}>Preview unavailable</Text>
          <Button
            {...restoreFocusTargetAttributes}
            appearance="subtle"
            size="small"
            icon={<ArrowSyncRegular />}
            className={styles.button}
            aria-label={`Retry previews for ${role} message ${node.message.sequence}`}
            onClick={() => { data.onRetry(node) }}
          />
        </div>
      )}
      <Button
        appearance="subtle"
        size="small"
        className={styles.textButton}
        onClick={(event) => { data.onPieces(node.node_id, event.currentTarget) }}
      >
        {node.piece_count > COMPACT_PIECE_COUNT ? `View all ${node.piece_count} pieces` : 'Message details'}
      </Button>
      {data.endpointIds.length > 0 && <div className={styles.footer}>
        {data.endpointIds.length === 1 && (
          <Button
            appearance="secondary"
            size="small"
            className={styles.endpoint}
            title={data.endpointIds[0]}
            aria-label={`Open conversation ${data.endpointIds[0]}`}
            onClick={() => { data.onSelect(data.endpointIds[0]) }}
          >
            Open {data.endpointIds[0].slice(0, 8)}
          </Button>
        )}
        {data.endpointIds.length > 1 && (
          <Button
            {...restoreFocusTargetAttributes}
            size="small"
            className={styles.endpoint}
            onClick={(event) => { data.onChoose(node.node_id, event.currentTarget, true) }}
          >
            {data.endpointIds.length} end here
          </Button>
        )}
      </div>}
      {!data.complete && <Text size={100} className={styles.detail}>More branches may appear</Text>}
      {data.complete && !data.hasChildren && node.role === 'user' && (
        <Text size={100} className={styles.detail}>No response stored</Text>
      )}
      <Handle type="source" position={Position.Bottom} isConnectable={false} className={styles.handle} />
    </article>
  )
}

export default memo(ConversationTreeNode)
