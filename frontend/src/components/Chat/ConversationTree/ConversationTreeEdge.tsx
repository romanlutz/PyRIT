import { BaseEdge, type Edge, type EdgeProps } from '@xyflow/react'

import { treeEdgePath } from './treeEdgeRouting'
import type { TreePosition } from './treeGraph'

type RoutedEdge = Edge<{ points: TreePosition[] }, 'tree'>

export default function ConversationTreeEdge({
  id, sourceX, sourceY, targetX, targetY, data, style,
}: EdgeProps<RoutedEdge>) {
  if (!data) throw new Error(`Conversation connection ${id} has no accepted route.`)
  const bends = data.points.slice(1, -1)
  if (bends.length > 1) {
    // Measured handles can differ fractionally from the layout; keep their adjoining legs vertical.
    bends[0] = { ...bends[0], x: sourceX }
    bends[bends.length - 1] = { ...bends[bends.length - 1], x: targetX }
  }
  const path = treeEdgePath([
    { x: sourceX, y: sourceY },
    ...bends,
    { x: targetX, y: targetY },
  ])
  return <BaseEdge id={id} path={path} style={style} interactionWidth={0} />
}
