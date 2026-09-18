import { render, screen } from '@testing-library/react'
import { Position } from '@xyflow/react'

import ConversationTreeEdge from './ConversationTreeEdge'

jest.mock('@xyflow/react', () => ({
  BaseEdge: ({ id, path, interactionWidth }: { id: string; path: string; interactionWidth: number }) => (
    <svg><path data-testid={id} d={path} data-interaction-width={interactionWidth} /></svg>
  ),
  Position: { Top: 'top', Bottom: 'bottom' },
}))

describe('ConversationTreeEdge', () => {
  beforeEach(() => jest.clearAllMocks())

  it('should render the accepted intermediate routing points between the actual message handles', () => {
    render(
      <ConversationTreeEdge
        id="route"
        source="source"
        target="target"
        sourceX={0}
        sourceY={2}
        sourcePosition={Position.Bottom}
        targetX={80}
        targetY={102}
        targetPosition={Position.Top}
        data={{ points: [{ x: 0, y: 0 }, { x: 0, y: 40 }, { x: 80, y: 40 }, { x: 80, y: 100 }] }}
      />,
    )
    expect(screen.getByTestId('route')).toHaveAttribute('d', 'M 0 2 L 0 32 Q 0 40 8 40 L 72 40 Q 80 40 80 48 L 80 102')
    expect(screen.getByTestId('route')).toHaveAttribute('data-interaction-width', '0')
  })

  it.each([
    [0.0001, 80.0002, 80],
    [-0.0001, 79.9998, 80],
    [0.0001, -79.9998, -80],
    [-0.0001, -80.0002, -80],
  ])('should keep endpoint legs vertical from x=%s to x=%s', (
    sourceX: number, targetX: number, plannedTargetX: number,
  ) => {
    const points = [
      { x: 0, y: 0 }, { x: 0, y: 40 }, { x: plannedTargetX, y: 40 }, { x: plannedTargetX, y: 100 },
    ]
    const originalPoints = points.map((point: { x: number; y: number }) => ({ ...point }))
    const direction = Math.sign(targetX - sourceX)
    render(
      <ConversationTreeEdge
        id="route"
        source="source"
        target="target"
        sourceX={sourceX}
        sourceY={2}
        sourcePosition={Position.Bottom}
        targetX={targetX}
        targetY={102}
        targetPosition={Position.Top}
        data={{ points }}
      />,
    )
    expect(screen.getByTestId('route')).toHaveAttribute('d',
      `M ${sourceX} 2 L ${sourceX} 32 Q ${sourceX} 40 ${sourceX + direction * 8} 40 `
      + `L ${targetX - direction * 8} 40 Q ${targetX} 40 ${targetX} 48 L ${targetX} 102`,
    )
    expect(points).toEqual(originalPoints)
  })

  it('should preserve intermediate corridors while aligning endpoint legs', () => {
    render(
      <ConversationTreeEdge
        id="route"
        source="source"
        target="target"
        sourceX={0.25}
        sourceY={2}
        sourcePosition={Position.Bottom}
        targetX={160.5}
        targetY={302}
        targetPosition={Position.Top}
        data={{ points: [
          { x: 0, y: 0 }, { x: 0, y: 40 }, { x: 80, y: 40 },
          { x: 80, y: 240 }, { x: 160, y: 240 }, { x: 160, y: 300 },
        ] }}
      />,
    )
    expect(screen.getByTestId('route')).toHaveAttribute('d',
      'M 0.25 2 L 0.25 32 Q 0.25 40 8.25 40 L 72 40 Q 80 40 80 48 '
      + 'L 80 232 Q 80 240 88 240 L 152.5 240 Q 160.5 240 160.5 248 L 160.5 302',
    )
  })

  it('should keep a straight connection attached to both measured handles', () => {
    render(
      <ConversationTreeEdge
        id="route"
        source="source"
        target="target"
        sourceX={0.0001}
        sourceY={2}
        sourcePosition={Position.Bottom}
        targetX={0.0002}
        targetY={102}
        targetPosition={Position.Top}
        data={{ points: [{ x: 0, y: 0 }, { x: 0, y: 100 }] }}
      />,
    )
    expect(screen.getByTestId('route')).toHaveAttribute('d', 'M 0.0001 2 L 0.0002 102')
  })
})
