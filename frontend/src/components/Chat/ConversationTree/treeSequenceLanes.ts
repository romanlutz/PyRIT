import type { Viewport } from '@xyflow/react'

import { TREE_NODE_HEIGHT, TREE_ROW_GAP, type LayoutNode, type TreePosition } from './treeGraph'
import type { TreePaneSize } from './useTreeViewport'

const SEQUENCE_LANE_PADDING = TREE_ROW_GAP / 4

export interface SequenceLane {
  readonly sequence: number
  readonly top: number
  readonly bottom: number
}

export function treeSequenceLanes(
  nodes: LayoutNode[],
  positions: ReadonlyMap<string, TreePosition>,
): SequenceLane[] {
  const lanes = new Map<number, SequenceLane>()
  for (const node of nodes) {
    const position = positions.get(node.id)
    if (!position) continue
    const previous = lanes.get(node.sequence)
    lanes.set(node.sequence, {
      sequence: node.sequence,
      top: Math.min(previous?.top ?? Infinity, position.y - SEQUENCE_LANE_PADDING),
      bottom: Math.max(previous?.bottom ?? -Infinity,
        position.y + (node.height ?? TREE_NODE_HEIGHT) + SEQUENCE_LANE_PADDING),
    })
  }
  return [...lanes.values()].sort((a: SequenceLane, b: SequenceLane) => a.sequence - b.sequence)
}

export function visibleSequenceLanes(lanes: SequenceLane[], viewport: Viewport, size: TreePaneSize): SequenceLane[] {
  if (size.width <= 0 || size.height <= 0) return []
  const top = -viewport.y / viewport.zoom
  const bottom = top + size.height / viewport.zoom
  let start = 0
  let end = lanes.length
  while (start < end) {
    const middle = Math.floor((start + end) / 2)
    if (lanes[middle].bottom < top) start = middle + 1
    else end = middle
  }
  const visible: SequenceLane[] = []
  for (let index = start; index < lanes.length && lanes[index].top <= bottom; index++) {
    visible.push(lanes[index])
  }
  return visible
}
