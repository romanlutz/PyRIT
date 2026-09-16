import { BaseEdge, type Edge, type EdgeProps } from '@xyflow/react'

import { treeEdgePath } from './treeEdgeRouting'
import type { TreePosition } from './treeGraph'

type RoutedEdge = Edge<{ points: TreePosition[] }, 'tree'>

export default function ConversationTreeEdge({
  id, sourceX, sourceY, targetX, targetY, data, style,
}: EdgeProps<RoutedEdge>) {
  if (!data) throw new Error(`Conversation connection ${id} has no accepted route.`)
  const path = treeEdgePath([
    { x: sourceX, y: sourceY },
    ...data.points.slice(1, -1),
    { x: targetX, y: targetY },
  ])
  return <BaseEdge id={id} path={path} style={style} interactionWidth={0} />
}
