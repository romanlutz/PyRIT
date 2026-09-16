import React from 'react'

import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { render, screen } from '@testing-library/react'
import type { Viewport } from '@xyflow/react'

import SequenceLanes from './SequenceLanes'
import type { SequenceLane } from './treeSequenceLanes'

let mockViewport: Viewport = { x: 0, y: 0, zoom: 1 }

jest.mock('@xyflow/react', () => ({
  useViewport: () => mockViewport,
  ViewportPortal: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="flow-viewport" style={{ transform: `translate(${mockViewport.x}px, ${mockViewport.y}px) scale(${mockViewport.zoom})` }}>
      {children}
    </div>
  ),
}))

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
)

describe('SequenceLanes', () => {
  const lanes: SequenceLane[] = [
    { sequence: 4, top: -16, bottom: 224 },
    { sequence: 10, top: 256, bottom: 592 },
    { sequence: 20, top: 624, bottom: 912 },
  ]
  const size = { width: 800, height: 600 }

  beforeEach(() => {
    jest.clearAllMocks()
    mockViewport = { x: 0, y: 0, zoom: 1 }
  })

  it('should render only visible stored sequence labels with transparent dashed boundaries', () => {
    render(<TestWrapper><SequenceLanes lanes={lanes} size={size} /></TestWrapper>)
    expect(screen.getByText('Sequence 4')).toBeInTheDocument()
    expect(screen.getByText('Sequence 10')).toBeInTheDocument()
    expect(screen.getByText('Sequence 10')).toHaveStyle({ transform: 'scale(1)' })
    expect(screen.queryByText('Sequence 20')).not.toBeInTheDocument()
    expect(screen.queryByText('Sequence 0')).not.toBeInTheDocument()
    expect(screen.getByTestId('tree-sequence-lanes')).toHaveAttribute('aria-hidden', 'true')
    expect(screen.getByTestId('tree-sequence-lanes')).toHaveStyle({ pointerEvents: 'none' })
    const lane = screen.getByTestId('tree-sequence-lane-10')
    expect(lane).toHaveStyle({ top: '256px', height: '336px' })
    expect(getComputedStyle(lane).borderTopStyle).toBe('dashed')
    expect(getComputedStyle(lane).borderBottomStyle).toBe('dashed')
    expect(getComputedStyle(lane).pointerEvents).toBe('none')
    expect(getComputedStyle(lane).backgroundColor).toBe('rgba(0, 0, 0, 0)')
    expect(screen.queryAllByRole('button')).toHaveLength(0)
  })

  it('should share the message viewport transform while keeping labels at its visible left edge', () => {
    const { rerender } = render(<TestWrapper><SequenceLanes lanes={lanes} size={size} /></TestWrapper>)
    mockViewport = { x: -160, y: -600, zoom: 2 }
    rerender(<TestWrapper><SequenceLanes lanes={lanes} size={size} /></TestWrapper>)
    expect(screen.getByTestId('flow-viewport')).toHaveStyle({ transform: 'translate(-160px, -600px) scale(2)' })
    expect(screen.getByText('Sequence 10')).toHaveStyle({ transform: 'scale(0.5)' })
    expect(screen.queryByText('Sequence 4')).not.toBeInTheDocument()
    expect(screen.getByText('Sequence 10')).toBeInTheDocument()
    expect(screen.queryByText('Sequence 20')).not.toBeInTheDocument()
    expect(screen.getByTestId('tree-sequence-lane-10')).toHaveStyle({
      left: '80px', top: '256px', height: '336px', width: '400px',
    })

    mockViewport = { x: 100, y: -300, zoom: 0.5 }
    rerender(<TestWrapper><SequenceLanes lanes={lanes} size={size} /></TestWrapper>)
    expect(screen.queryByText('Sequence 10')).not.toBeInTheDocument()
    expect(screen.getByTestId('tree-sequence-lane-20')).toHaveStyle({
      left: '-200px', top: '624px', height: '288px', width: '1600px',
    })
    expect(screen.getByText('Sequence 20')).toHaveStyle({ transform: 'scale(2)' })
  })
})
