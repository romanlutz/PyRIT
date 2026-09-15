import { BaseEdge, getSmoothStepPath, type Edge, type EdgeProps } from '@xyflow/react'

type RoutedEdge = Edge<{ centerY: number }, 'tree'>

export default function ConversationTreeEdge({
  id, sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition, data, style,
}: EdgeProps<RoutedEdge>) {
  const [path] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    centerY: data?.centerY,
  })
  return <BaseEdge id={id} path={path} style={style} interactionWidth={0} />
}
