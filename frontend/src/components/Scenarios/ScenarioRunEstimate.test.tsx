import type { ReactNode } from 'react'

import { render, screen } from '@testing-library/react'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'

import type { ScenarioDefaultRunSizeEstimate, ScenarioRunEstimateState } from '@/types'

import {
  ScenarioRunEstimateDetails,
  ScenarioRunEstimateSummary,
} from './ScenarioRunEstimate'
import { mapScenarioRunEstimate } from './scenarioRunEstimateAdapter'

function TestWrapper({ children }: { children: ReactNode }) {
  return <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
}

const EXACT_ESTIMATE: ScenarioDefaultRunSizeEstimate = {
  version: 1,
  status: 'exact',
  total_attack_count: 8,
  minimum_attack_count: null,
  maximum_attack_count: null,
  condition: null,
  components: [
    {
      label: 'Prompt sending',
      count: 8,
      factors: [
        { label: 'selected seed groups', count: 4 },
        { label: 'jailbreak templates', count: 2 },
        { label: 'techniques', count: 1 },
        { label: 'attempts', count: 1 },
      ],
      is_baseline: false,
      note: 'One planned attack per selected objective and template.',
    },
    {
      label: 'Baseline attack',
      // Deliberately differs from the authoritative total when added to the
      // first component so this test detects accidental client-side summing.
      count: 2,
      factors: [],
      is_baseline: true,
      note: 'Fixture component used to guard the authoritative total.',
    },
  ],
  datasets: [
    {
      name: 'harmbench',
      kind: 'dataset',
      logical_seed_group_count: 4,
      selected_seed_group_count: 4,
      configured_caps: [
        {
          label: 'Jailbreak templates',
          count: 2,
          configured_on: 'configuration',
          dataset_name: null,
        },
      ],
      selection_note: 'Four compatible objective groups selected.',
    },
  ],
  note: 'The planned total is authoritative.',
  retries_included: false,
}

describe('ScenarioRunEstimate', () => {
  it('renders the authoritative total, ordered factors, dataset counts, caps, and notes', () => {
    const state = mapScenarioRunEstimate(EXACT_ESTIMATE, 'request')

    render(
      <TestWrapper>
        <ScenarioRunEstimateDetails state={state} />
      </TestWrapper>,
    )

    expect(screen.getByText('8 planned attacks')).toBeInTheDocument()
    expect(screen.queryByText('10 planned attacks')).not.toBeInTheDocument()
    expect(screen.getByText('Prompt sending')).toBeInTheDocument()
    expect(screen.getByText('Baseline attack')).toBeInTheDocument()
    expect(screen.getByText('Baseline')).toBeInTheDocument()
    expect(screen.getByText('× 4 selected seed groups')).toBeInTheDocument()
    expect(screen.getByText('× 2 jailbreak templates')).toBeInTheDocument()
    expect(screen.getByText('harmbench')).toBeInTheDocument()
    expect(screen.getByText('Jailbreak templates: 2 (configuration)')).toBeInTheDocument()
    expect(screen.getByText('Four compatible objective groups selected.')).toBeInTheDocument()
    expect(screen.getByText(
      'Prompt sending: 4 selected seed groups × 2 jailbreak templates × 1 techniques × 1 attempts = 8 + Baseline attack: 2; planned total = 8',
    )).toBeInTheDocument()
    expect(screen.getByText('The planned total is authoritative.')).toBeInTheDocument()
    expect(screen.getByText('How this count is calculated')).toBeInTheDocument()
    expect(screen.getByText('Retries are not included. Estimate schema v1.')).toBeInTheDocument()
    expect(screen.queryByText(
      /Run size calculated|Final count set at launch|Backend estimate|Conditional estimate|Backend formula/i,
    )).not.toBeInTheDocument()
  })

  it('renders bounded and upper-only conditional estimates from structured bounds', () => {
    const bounded = mapScenarioRunEstimate({
      ...EXACT_ESTIMATE,
      status: 'conditional',
      total_attack_count: null,
      minimum_attack_count: 12,
      maximum_attack_count: 20,
      condition: 'target_capabilities',
    }, 'default')
    const { rerender } = render(
      <TestWrapper>
        <ScenarioRunEstimateDetails state={bounded} />
      </TestWrapper>,
    )

    expect(screen.getByText('12–20 planned attacks')).toBeInTheDocument()
    expect(screen.queryByText(/Run size calculated|Final count set at launch/i)).not.toBeInTheDocument()

    const upperOnly = mapScenarioRunEstimate({
      ...EXACT_ESTIMATE,
      status: 'conditional',
      total_attack_count: null,
      minimum_attack_count: null,
      maximum_attack_count: 20,
      condition: 'launch_configuration',
    }, 'request')
    rerender(
      <TestWrapper>
        <ScenarioRunEstimateSummary state={upperOnly} />
      </TestWrapper>,
    )
    expect(screen.getByText('Up to 20 planned attacks')).toBeInTheDocument()
    expect(screen.getByText('Up to 20 planned attacks')).toBeInTheDocument()
  })

  it('renders an equal bounded estimate as one number', () => {
    const conditional = mapScenarioRunEstimate({
      ...EXACT_ESTIMATE,
      status: 'conditional',
      total_attack_count: null,
      minimum_attack_count: 12,
      maximum_attack_count: 12,
      condition: 'launch_configuration',
    }, 'default')

    render(
      <TestWrapper>
        <ScenarioRunEstimateSummary state={conditional} />
      </TestWrapper>,
    )

    expect(screen.getByText('12 planned attacks')).toBeInTheDocument()
    expect(screen.queryByText('12–12 planned attacks')).not.toBeInTheDocument()
  })

  it('supports loading, unknown conditional, unavailable, and stale states', () => {
    const loading: ScenarioRunEstimateState = { status: 'loading', scope: 'request' }
    const { rerender } = render(
      <TestWrapper>
        <ScenarioRunEstimateDetails state={loading} />
      </TestWrapper>,
    )
    expect(screen.getByText('Calculating planned attacks...')).toBeInTheDocument()

    const conditional = mapScenarioRunEstimate({
      ...EXACT_ESTIMATE,
      status: 'conditional',
      total_attack_count: null,
      minimum_attack_count: null,
      maximum_attack_count: null,
      condition: null,
      components: [],
      datasets: [],
      note: null,
    }, 'default')
    rerender(
      <TestWrapper>
        <ScenarioRunEstimateDetails state={conditional} />
      </TestWrapper>,
    )
    expect(screen.getByText('Select targets to calculate')).toBeInTheDocument()
    expect(screen.queryByText(/Run size calculated|Final count set at launch/i)).not.toBeInTheDocument()
    expect(screen.getByText(
      'Component breakdown unavailable; final count is set at launch',
    )).toBeInTheDocument()
    expect(screen.getByText('A component breakdown isn’t available.')).toBeInTheDocument()
    expect(screen.getByText('Dataset population details aren’t available.')).toBeInTheDocument()

    const configuredConditional = mapScenarioRunEstimate({
      ...EXACT_ESTIMATE,
      status: 'conditional',
      total_attack_count: null,
      minimum_attack_count: null,
      maximum_attack_count: null,
      condition: 'launch_configuration',
      components: [],
      datasets: [],
    }, 'request')
    rerender(
      <TestWrapper>
        <ScenarioRunEstimateSummary state={configuredConditional} />
      </TestWrapper>,
    )
    expect(screen.getByText('Run size is confirmed at launch.')).toBeInTheDocument()
    expect(screen.queryByText('Select targets to calculate')).not.toBeInTheDocument()

    const unavailable = mapScenarioRunEstimate({
      ...EXACT_ESTIMATE,
      status: 'unavailable',
      total_attack_count: null,
      minimum_attack_count: null,
      maximum_attack_count: null,
      condition: null,
      components: [],
      datasets: [],
      note: 'Target capability is not available.',
    }, 'request')
    rerender(
      <TestWrapper>
        <ScenarioRunEstimateSummary state={unavailable} />
        <ScenarioRunEstimateDetails state={unavailable} />
      </TestWrapper>,
    )
    expect(screen.getAllByText('Estimate unavailable')).toHaveLength(2)
    expect(screen.getByText('Configured run size unavailable')).toBeInTheDocument()
    expect(screen.getByText('Target capability is not available.')).toBeInTheDocument()

    const exact = mapScenarioRunEstimate(EXACT_ESTIMATE, 'request')
    if (exact.status !== 'available') {
      throw new Error('Expected exact estimate to map to an available state.')
    }
    const stale: ScenarioRunEstimateState = {
      status: 'stale',
      estimate: exact.estimate,
      label: 'Showing the last successful estimate.',
      error: 'Preview service timed out.',
    }
    rerender(
      <TestWrapper>
        <ScenarioRunEstimateDetails state={stale} />
      </TestWrapper>,
    )
    expect(screen.getByText('Previous estimate')).toBeInTheDocument()
    expect(screen.getByText('8 planned attacks')).toBeInTheDocument()
    expect(screen.getByText('Showing the last successful estimate.')).toBeInTheDocument()
    expect(screen.getByText('Preview service timed out.')).toBeInTheDocument()
  })
})
