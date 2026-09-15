import { TREE_NODE_HEIGHT, TREE_ROW_GAP, type LayoutNode, type TreePosition } from './treeGraph'

/** Route every connection into a row above its tallest message, not through a neighboring card. */
export function treeEdgeLanes(
  nodes: LayoutNode[],
  positions: ReadonlyMap<string, TreePosition>,
): Map<string, number> {
  const rowLanes = new Map<number, number>()
  const rows = new Map<string, number>()
  for (const node of nodes) {
    const position = positions.get(node.id)
    if (!position) continue
    const row = position.y + (node.height ?? TREE_NODE_HEIGHT) / 2
    rows.set(node.id, row)
    const lane = position.y - TREE_ROW_GAP / 2
    rowLanes.set(row, Math.min(rowLanes.get(row) ?? lane, lane))
  }
  const lanes = new Map<string, number>()
  for (const [id, row] of rows) {
    const lane = rowLanes.get(row)
    if (lane !== undefined) lanes.set(id, lane)
  }
  return lanes
}
