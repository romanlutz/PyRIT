import { useEffect, useState, type RefObject } from 'react'

import type { Viewport } from '@xyflow/react'

import type { ConversationTreeNode } from '@/types'

import { TREE_NODE_WIDTH, treeNodeHeight, type TreePosition } from './treeGraph'

const INITIAL_FOCUS_PADDING_FACTOR = 1.24
const MINIMUM_INITIAL_READABLE_ZOOM = 0.8

export interface TreePaneSize {
  readonly width: number
  readonly height: number
}

export function initialTreeFocus(
  nodes: ConversationTreeNode[],
  positions: ReadonlyMap<string, TreePosition>,
  currentPath: ReadonlySet<string>,
  size: TreePaneSize,
): string[] {
  const byId = new Map(nodes.map((node: ConversationTreeNode) => [node.node_id, node]))
  const path = [...currentPath].filter((id: string) => byId.has(id)).slice(0, 2)
  const candidates = path.length ? path : nodes.slice(0, 1).map((node: ConversationTreeNode) => node.node_id)
  if (candidates.length < 2) return candidates
  let left = Infinity
  let top = Infinity
  let right = -Infinity
  let bottom = -Infinity
  for (const id of candidates) {
    const node = byId.get(id)
    const position = positions.get(id)
    if (!node || !position) return candidates.slice(0, 1)
    left = Math.min(left, position.x)
    top = Math.min(top, position.y)
    right = Math.max(right, position.x + TREE_NODE_WIDTH)
    bottom = Math.max(bottom, position.y + treeNodeHeight(node.piece_count))
  }
  const zoom = Math.min(
    size.width / ((right - left) * INITIAL_FOCUS_PADDING_FACTOR),
    size.height / ((bottom - top) * INITIAL_FOCUS_PADDING_FACTOR),
  )
  return zoom >= MINIMUM_INITIAL_READABLE_ZOOM ? candidates : candidates.slice(0, 1)
}

export function visibleTreeNodes(
  nodes: ConversationTreeNode[],
  positions: ReadonlyMap<string, TreePosition>,
  viewport: Viewport,
  size: TreePaneSize,
): ConversationTreeNode[] {
  const left = -viewport.x / viewport.zoom
  const top = -viewport.y / viewport.zoom
  const right = left + size.width / viewport.zoom
  const bottom = top + size.height / viewport.zoom
  return nodes.filter((node: ConversationTreeNode) => {
    const position = positions.get(node.node_id)
    return position !== undefined
      && position.x <= right && position.x + TREE_NODE_WIDTH >= left
      && position.y <= bottom && position.y + treeNodeHeight(node.piece_count) >= top
  })
}

export function useTreePaneSize(ref: RefObject<HTMLDivElement | null>, active: boolean): TreePaneSize {
  const [size, setSize] = useState<TreePaneSize>({ width: 0, height: 0 })
  useEffect(() => {
    const pane = ref.current
    if (!pane || !active) return
    const measure = (): void => {
      const { width, height } = pane.getBoundingClientRect()
      setSize((previous: TreePaneSize) => previous.width === width && previous.height === height ? previous : { width, height })
    }
    const observer = new ResizeObserver(measure)
    observer.observe(pane)
    const frame = requestAnimationFrame(measure)
    return () => { cancelAnimationFrame(frame); observer.disconnect() }
  }, [ref, active])
  return size
}
