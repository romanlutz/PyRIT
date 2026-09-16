import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button, MessageBar, MessageBarBody, Text, mergeClasses, useRestoreFocusTarget } from '@fluentui/react-components'
import { AddRegular, ChatMultipleRegular, SubtractRegular } from '@fluentui/react-icons'
import { ReactFlow, ReactFlowProvider, useReactFlow, type Edge, type EdgeTypes, type NodeTypes, type Viewport } from '@xyflow/react'
import { ErrorBoundary } from 'react-error-boundary'
import '@xyflow/react/dist/style.css'

import { toApiError } from '@/services/errors'
import type { ConversationTreeEndpoint, ConversationTreeNode as TreeNode } from '@/types'

import ConversationChooser, { type ConversationChooserScope } from './ConversationChooser'
import { useConversationTreeStyles } from './ConversationTree.styles'
import ConversationTreeNode, { type MessageFlowNode } from './ConversationTreeNode'
import ConversationTreeEdge from './ConversationTreeEdge'
import MediaLightbox from './MediaLightbox'
import MessagePiecesDialog from './MessagePiecesDialog'
import SequenceLanes from './SequenceLanes'
import { treeSequenceLanes } from './treeSequenceLanes'
import {
  TREE_NODE_WIDTH,
  conversationPath,
  indexTree,
  orderedTreeNodes,
  reservePositions,
  treeNodeHeight,
  type LayoutNode,
  type TreePosition,
} from './treeGraph'
import { useConversationTreeData } from './useConversationTreeData'
import { useTreeLayout } from './useTreeLayout'
import { initialTreeFocus, useTreePaneSize, visibleTreeNodes } from './useTreeViewport'

const NODE_TYPES: NodeTypes = { message: ConversationTreeNode }
const EDGE_TYPES: EdgeTypes = { tree: ConversationTreeEdge }
const EMPTY_CONVERSATION_BUTTONS = 3
const MIN_ZOOM = 0.2
const MAX_ZOOM = 2

interface ConversationTreeProps {
  attackResultId: string
  activeConversationId: string | null
  onSelectConversation: (id: string) => void
  active?: boolean
  refreshKey?: number
}

interface NodeDialog {
  readonly nodeId: string
  readonly opener: HTMLElement
}

interface MediaDialog extends NodeDialog {
  readonly pieceIndex: number
}

interface ChooserDialog {
  readonly scope: ConversationChooserScope
  readonly opener: HTMLElement
}

function ConversationTreePane({
  attackResultId,
  activeConversationId,
  onSelectConversation,
  active = true,
  refreshKey = 0,
}: ConversationTreeProps) {
  const styles = useConversationTreeStyles()
  const restoreFocusTargetAttributes = useRestoreFocusTarget()
  const { reads, snapshot } = useConversationTreeData(attackResultId, activeConversationId, active, refreshKey)
  const [focusedId, setFocusedId] = useState<string | null>(null)
  const [chooser, setChooser] = useState<ChooserDialog | null>(null)
  const [media, setMedia] = useState<MediaDialog | null>(null)
  const [details, setDetails] = useState<NodeDialog | null>(null)
  const [viewport, setViewport] = useState<Viewport>({ x: 0, y: 0, zoom: 1 })
  const [flowReady, setFlowReady] = useState(false)
  const [navigationError, setNavigationError] = useState<string | null>(null)
  const [previousActive, setPreviousActive] = useState(active)
  if (previousActive !== active) {
    setPreviousActive(active)
    if (!active) {
      setChooser(null)
      setMedia(null)
      setDetails(null)
    }
  }
  const paneRef = useRef<HTMLDivElement>(null)
  const fitted = useRef(false)
  const flow = useReactFlow<MessageFlowNode>()
  const size = useTreePaneSize(paneRef, active)
  const index = useMemo(() => indexTree(snapshot.nodes, snapshot.conversations), [snapshot.nodes, snapshot.conversations])
  const currentPath = useMemo(
    () => conversationPath(index, snapshot.conversations, activeConversationId ?? snapshot.mainConversationId),
    [index, snapshot.conversations, snapshot.mainConversationId, activeConversationId],
  )
  const orderedNodes = useMemo(() => orderedTreeNodes(index), [index])
  const nodeIds = useMemo(() => new Set(orderedNodes.map((node: TreeNode) => node.node_id)), [orderedNodes])
  const layoutNodes = useMemo(() => orderedNodes.map((node: TreeNode): LayoutNode => ({
    id: node.node_id, parentId: node.parent_node_id, sequence: node.message.sequence, height: treeNodeHeight(node.piece_count),
  })), [orderedNodes])
  const anchorId = focusedId && nodeIds.has(focusedId)
    ? focusedId
    : [...currentPath].find((id: string) => nodeIds.has(id)) ?? orderedNodes[0]?.node_id ?? null
  const layout = useTreeLayout(layoutNodes, anchorId, active)
  const positions = useMemo(() => reservePositions(layoutNodes, layout.positions, anchorId), [layoutNodes, layout.positions, anchorId])
  const lanes = useMemo(() => treeSequenceLanes(layoutNodes, positions), [layoutNodes, positions])
  const visible = useMemo(
    () => visibleTreeNodes(orderedNodes, positions, viewport, size),
    [orderedNodes, positions, viewport, size],
  )

  useEffect(() => { reads.setVisibleNodes(active ? visible : []) }, [reads, active, visible])

  const reportNavigationError = useCallback((error: unknown) => { setNavigationError(toApiError(error).detail) }, [])
  useEffect(() => {
    if (!active || !flowReady || fitted.current || orderedNodes.length === 0 || !size.width || !size.height || (!layout.ready && !layout.error)) return
    fitted.current = true
    const nodes = initialTreeFocus(orderedNodes, positions, currentPath, size).map((id: string) => ({ id }))
    void flow.fitView({ nodes, maxZoom: 1, minZoom: MIN_ZOOM, padding: 0.12 })
      .then(() => { setViewport(flow.getViewport()) }).catch(reportNavigationError)
  }, [active, flowReady, orderedNodes, positions, size, currentPath, flow, layout.ready, layout.error, reportNavigationError])

  const choose = useCallback((nodeId: string, opener: HTMLElement, endpointsOnly = false): void => {
    setChooser({ scope: { kind: 'branch', nodeId, endpointsOnly }, opener })
  }, [])
  const select = useCallback((id: string): void => {
    setChooser(null)
    onSelectConversation(id)
  }, [onSelectConversation])
  const openMedia = useCallback((nodeId: string, pieceIndex: number, opener: HTMLElement): void => {
    const node = snapshot.nodes.get(nodeId)
    if (!node) {
      setNavigationError('This message is no longer in the loaded snapshot. Select another message.')
      return
    }
    setMedia({ nodeId, pieceIndex, opener })
    reads.requestPreviews([node], 'full')
  }, [snapshot.nodes, reads])
  const openPieces = useCallback((nodeId: string, opener: HTMLElement): void => {
    const node = snapshot.nodes.get(nodeId)
    if (!node) {
      setNavigationError('This message is no longer in the loaded snapshot. Select another message.')
      return
    }
    setDetails({ nodeId, opener })
    reads.requestPreviews([node])
  }, [snapshot.nodes, reads])
  const retryPreviews = useCallback((node: TreeNode): void => {
    reads.requestPreviews([node], 'text', true)
    if (snapshot.previews.get(node.preview_key)?.thumbnail?.error) reads.requestPreviews([node], 'thumbnail', true)
  }, [reads, snapshot.previews])
  const focus = useCallback((id: string): void => { setFocusedId(id) }, [])
  const nodes = useMemo(() => orderedNodes.map((node: TreeNode): MessageFlowNode => ({
    id: node.node_id,
    type: 'message',
    position: positions.get(node.node_id) ?? { x: 0, y: 0 },
    width: TREE_NODE_WIDTH,
    height: treeNodeHeight(node.piece_count),
    draggable: false,
    connectable: false,
    selectable: false,
    focusable: false,
    data: {
      message: node,
      previews: snapshot.previews.get(node.preview_key),
      currentPath: currentPath.has(node.node_id),
      currentEndpoint: snapshot.conversations.get(activeConversationId ?? '')?.node_id === node.node_id,
      mainEndpoint: snapshot.conversations.get(snapshot.mainConversationId ?? '')?.node_id === node.node_id,
      endpointIds: (index.endpoints.get(node.node_id) ?? []).map((endpoint: ConversationTreeEndpoint) => endpoint.conversation_id),
      hasChildren: (index.children.get(node.node_id)?.length ?? 0) > 0,
      conversationCount: index.conversationCounts.get(node.node_id) ?? 0,
      complete: snapshot.complete,
      onChoose: choose,
      onSelect: select,
      onMedia: openMedia,
      onPieces: openPieces,
      onRetry: retryPreviews,
      onFocus: focus,
    },
  })), [orderedNodes, positions, snapshot.previews, snapshot.conversations, snapshot.mainConversationId, snapshot.complete,
    currentPath, activeConversationId, index, choose, select, openMedia, openPieces, retryPreviews, focus])
  const edges = useMemo(() => orderedNodes.flatMap((node: TreeNode): Edge<{ points: TreePosition[] }>[] => {
    const points = layout.edgeRoutes.get(node.node_id)
    return node.parent_node_id && nodeIds.has(node.parent_node_id)
      && layout.arrangedNodeIds.has(node.parent_node_id) && layout.arrangedNodeIds.has(node.node_id) && points
      ? [{
          id: `${node.parent_node_id}:${node.node_id}`,
          source: node.parent_node_id,
          target: node.node_id,
          type: 'tree',
          data: { points },
          selectable: false,
          focusable: false,
          className: currentPath.has(node.node_id) ? styles.pathEdge : undefined,
        }]
      : []
  }), [orderedNodes, nodeIds, currentPath, styles.pathEdge, layout.arrangedNodeIds, layout.edgeRoutes])
  const emptyConversations = [...snapshot.conversations.values()].filter((endpoint: ConversationTreeEndpoint) => endpoint.node_id === null)
  const mediaNode = media ? snapshot.nodes.get(media.nodeId) : undefined
  const detailsNode = details ? snapshot.nodes.get(details.nodeId) : undefined
  const progress = snapshot.total === null ? '' : `; ${snapshot.processed} of ${snapshot.total} conversations`
  const status = snapshot.refreshing ? 'Refreshing branches; showing the previous snapshot'
    : snapshot.error ? `Partial tree${progress}`
      : snapshot.complete ? `All ${snapshot.total ?? snapshot.processed} conversations loaded`
        : snapshot.nodes.size > 0 ? `Loading more branches${progress}` : `Loading branches${progress}`

  return (
    <section className={mergeClasses(styles.root, !active && styles.inactive)} aria-label="Conversation tree" hidden={!active} data-testid="conversation-tree" data-layout-pending={layout.busy}>
      <div className={styles.toolbar}>
        <Button {...restoreFocusTargetAttributes} icon={<ChatMultipleRegular />} className={styles.textButton} onClick={(event) => { setChooser({ scope: { kind: 'all' }, opener: event.currentTarget }) }}>
          Conversations ({snapshot.conversations.size}{snapshot.complete ? '' : '+'})
        </Button>
        <div className={styles.controls} role="group" aria-label="Tree view controls">
          <Button appearance="subtle" icon={<SubtractRegular />} className={styles.button} aria-label="Zoom out" onClick={() => { void flow.zoomOut().then(() => { setViewport(flow.getViewport()) }).catch(reportNavigationError) }} />
          <Text size={200} aria-label="Zoom level">{Math.round(viewport.zoom * 100)}%</Text>
          <Button appearance="subtle" icon={<AddRegular />} className={styles.button} aria-label="Zoom in" onClick={() => { void flow.zoomIn().then(() => { setViewport(flow.getViewport()) }).catch(reportNavigationError) }} />
          <Button className={styles.textButton} disabled={nodes.length === 0} onClick={() => { void flow.fitView({ maxZoom: 1, minZoom: MIN_ZOOM, padding: 0.2 }).then(() => { setViewport(flow.getViewport()) }).catch(reportNavigationError) }}>Fit to view</Button>
        </div>
      </div>
      <div className={styles.status} role="status" aria-live="polite" aria-atomic="true">{status}{layout.busy ? '; arranging branches' : ''}</div>
      {snapshot.error && <MessageBar intent="error"><MessageBarBody>{snapshot.error}</MessageBarBody><Button className={styles.textButton} onClick={reads.retryTopology}>{snapshot.needsRefresh ? 'Refresh tree' : 'Retry loading branches'}</Button></MessageBar>}
      {layout.error && <MessageBar intent="warning"><MessageBarBody>Branches could not be arranged. Known messages remain available. {layout.error}</MessageBarBody><Button className={styles.textButton} onClick={layout.retry}>Retry layout</Button></MessageBar>}
      {navigationError && <MessageBar intent="error"><MessageBarBody>{navigationError}</MessageBarBody></MessageBar>}
      {emptyConversations.length > 0 && (
        <section className={styles.emptyConversations} aria-label="Empty conversations">
          <Text size={200}>Empty conversations</Text>
          {emptyConversations.slice(0, EMPTY_CONVERSATION_BUTTONS).map((endpoint: ConversationTreeEndpoint) => (
            <Button
              key={endpoint.conversation_id}
              size="small"
              className={styles.textButton}
              aria-label={`Open empty conversation ${endpoint.conversation_id}`}
              title={endpoint.conversation_id}
              onClick={() => { select(endpoint.conversation_id) }}
            >
              {endpoint.conversation_id.slice(0, 8)}{endpoint.conversation_id === activeConversationId ? ' (Current)' : ''}{endpoint.conversation_id === snapshot.mainConversationId ? ' (Main)' : ''}
            </Button>
          ))}
          {emptyConversations.length > EMPTY_CONVERSATION_BUTTONS && <Button {...restoreFocusTargetAttributes} className={styles.textButton} onClick={(event) => { setChooser({ scope: { kind: 'empty' }, opener: event.currentTarget }) }}>View all {emptyConversations.length}</Button>}
        </section>
      )}
      {snapshot.complete && snapshot.nodes.size === 0 && <Text className={styles.empty}>No messages are stored in these conversations. Open a conversation to begin.</Text>}
      <div ref={paneRef} className={styles.pane} data-testid="conversation-tree-pane">
        <ReactFlow<MessageFlowNode>
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          nodesDraggable={false}
          nodesConnectable={false}
          nodesFocusable={false}
          edgesFocusable={false}
          edgesReconnectable={false}
          elementsSelectable={false}
          connectOnClick={false}
          deleteKeyCode={null}
          selectionKeyCode={null}
          selectionOnDrag={false}
          selectNodesOnDrag={false}
          zoomOnDoubleClick={false}
          onlyRenderVisibleElements
          minZoom={MIN_ZOOM}
          maxZoom={MAX_ZOOM}
          onInit={() => { setFlowReady(true) }}
          onMoveEnd={(_, next: Viewport) => { setViewport(next) }}
          aria-label="Read-only conversation message graph"
        >
          <SequenceLanes lanes={lanes} size={size} />
        </ReactFlow>
      </div>
      {active && chooser && <ConversationChooser scope={chooser.scope} opener={chooser.opener} index={index} snapshot={snapshot} reads={reads} activeConversationId={activeConversationId} onSelect={select} onClose={() => { setChooser(null) }} />}
      {active && details && detailsNode && <MessagePiecesDialog key={details.nodeId} node={detailsNode} previews={snapshot.previews.get(detailsNode.preview_key)} opener={details.opener} onClose={() => { setDetails(null) }} onMedia={openMedia} onRetry={retryPreviews} />}
      {active && media && mediaNode && <MediaLightbox key={`${media.nodeId}:${media.pieceIndex}`} node={mediaNode} pieceIndex={media.pieceIndex} result={snapshot.previews.get(mediaNode.preview_key)?.full} opener={media.opener} onClose={() => { setMedia(null); setDetails(null); reads.cancelFullPreview(mediaNode) }} onRetry={() => { reads.requestPreviews([mediaNode], 'full', true) }} />}
    </section>
  )
}

/** Keeping this mounted preserves the viewport; hiding it suspends only tree-owned reads. */
export default function ConversationTree(props: ConversationTreeProps) {
  return (
    <ErrorBoundary key={props.attackResultId} fallback={<MessageBar intent="error"><MessageBarBody>The conversation tree could not be displayed. Return to chat to continue working.</MessageBarBody></MessageBar>}>
      <ReactFlowProvider>
        <ConversationTreePane {...props} />
      </ReactFlowProvider>
    </ErrorBoundary>
  )
}
