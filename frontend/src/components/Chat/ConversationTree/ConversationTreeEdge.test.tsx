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
})
