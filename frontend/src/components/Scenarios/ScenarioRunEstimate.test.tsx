import type { ReactNode } from 'react'

import { render, screen } from '@testing-library/react'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'

import type { ScenarioRunEstimateState } from '@/types'

import {
  ScenarioRunEstimateDetails,
  ScenarioRunEstimateSummary,
} from './ScenarioRunEstimate'

function TestWrapper({ children }: { children: ReactNode }) {
  return <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
}

describe('ScenarioRunEstimate', () => {
  it('renders the authoritative backend total and ordered structured formula terms', () => {
    const state: ScenarioRunEstimateState = {
      status: 'available',
      estimate: {
        scope: 'request',
        total: 8,
        additiveComponents: [
          { id: 'objectives', label: 'Objectives', value: 5 },
          { id: 'baseline', label: 'Baseline attacks', value: 0 },
        ],
        multiplicativeFactors: [
          { id: 'techniques', label: 'Techniques', value: 1 },
          { id: 'templates', label: 'Jailbreak templates', value: 2 },
        ],
        formula: '(5 objectives + 0 baseline) * 1 technique * 2 templates',
        caveat: 'The backend total is authoritative.',
      },
    }

    render(
      <TestWrapper>
        <ScenarioRunEstimateDetails state={state} />
      </TestWrapper>,
    )

    expect(screen.getByText('8 attacks')).toBeInTheDocument()
    expect(screen.queryByText('10 attacks')).not.toBeInTheDocument()
    expect(screen.getAllByRole('term').map((term) => term.textContent)).toEqual([
      'Objectives',
      'Baseline attacks',
      'Techniques',
      'Jailbreak templates',
    ])
    expect(screen.getByText('(5 objectives + 0 baseline) * 1 technique * 2 templates')).toBeInTheDocument()
    expect(screen.getByText('The backend total is authoritative.')).toBeInTheDocument()
  })

  it('supports loading, conditional, unavailable, and default-only labels', () => {
    const loading: ScenarioRunEstimateState = { status: 'loading', scope: 'request' }
    const { rerender } = render(
      <TestWrapper>
        <ScenarioRunEstimateDetails state={loading} />
      </TestWrapper>,
    )
    expect(screen.getByText('Loading backend run estimate...')).toBeInTheDocument()

    const conditional: ScenarioRunEstimateState = {
      status: 'conditional',
      estimate: {
        scope: 'default',
        additiveComponents: [],
        multiplicativeFactors: [],
      },
    }
    rerender(
      <TestWrapper>
        <ScenarioRunEstimateDetails state={conditional} />
      </TestWrapper>,
    )
    expect(screen.getByText('Conditional estimate')).toBeInTheDocument()
    expect(screen.getByText('Default configuration')).toBeInTheDocument()
    expect(screen.getByText('No formula supplied by the backend.')).toBeInTheDocument()
    expect(screen.getByText('No additional caveat supplied by the backend.')).toBeInTheDocument()

    const unavailable: ScenarioRunEstimateState = {
      status: 'unavailable',
      scope: 'request',
      label: 'Estimate endpoint unavailable',
      caveat: 'Launch remains available.',
    }
    rerender(
      <TestWrapper>
        <ScenarioRunEstimateSummary state={unavailable} />
        <ScenarioRunEstimateDetails state={unavailable} />
      </TestWrapper>,
    )
    expect(screen.getAllByText('Estimate unavailable')).toHaveLength(2)
    expect(screen.getByText('Estimate endpoint unavailable')).toBeInTheDocument()
    expect(screen.getByText('Launch remains available.')).toBeInTheDocument()
  })
})
