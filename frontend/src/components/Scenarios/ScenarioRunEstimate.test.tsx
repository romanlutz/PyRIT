import type { ReactNode } from 'react'

import { render, screen, within } from '@testing-library/react'
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

function makeEstimate(
  overrides: Partial<ScenarioDefaultRunSizeEstimate> = {},
): ScenarioDefaultRunSizeEstimate {
  return {
    version: 1,
    status: 'exact',
    total_attack_count: 16,
    minimum_attack_count: null,
    maximum_attack_count: null,
    condition: null,
    components: [
      {
        label: 'Default technique sweep',
        count: 16,
        factors: [
          { label: 'selected logical seed groups', count: 4 },
          { label: 'default concrete techniques', count: 4 },
        ],
        is_baseline: false,
        note: null,
      },
    ],
    datasets: [
      {
        name: 'harmbench',
        kind: 'dataset',
        logical_seed_group_count: 400,
        selected_seed_group_count: 4,
        configured_caps: [],
        selection_note: 'The default selection uses 4 of 400 logical seed groups.',
      },
    ],
    adaptive_details: null,
    note: null,
    retries_included: false,
    ...overrides,
  }
}

function renderDetails(estimate: ScenarioDefaultRunSizeEstimate): void {
  render(
    <TestWrapper>
      <ScenarioRunEstimateDetails state={mapScenarioRunEstimate(estimate, 'request')} />
    </TestWrapper>,
  )
}

describe('ScenarioRunEstimate', () => {
  it('renders an exact technique-by-objective equation with a complete accessible sentence', () => {
    renderDetails(makeEstimate())

    const equation = screen.getByRole('group', {
      name: '4 techniques multiplied by 4 objectives equals 16 planned attacks.',
    })
    expect(within(equation).getAllByText('4')).toHaveLength(2)
    expect(within(equation).getByText('techniques')).toBeInTheDocument()
    expect(within(equation).getByText('objectives')).toBeInTheDocument()
    expect(within(equation).getByText('16')).toBeInTheDocument()
    expect(within(equation).getByText('planned attacks')).toBeInTheDocument()
    expect(screen.getByText('4 objectives from harmbench · 400 available')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Run calculation' })).toBeInTheDocument()
  })

  it('renders heterogeneous compatibility as truthful per-technique additive terms', () => {
    renderDetails(makeEstimate({
      total_attack_count: 6,
      components: [
        {
          label: 'technique_alpha',
          count: 4,
          factors: [
            { label: 'selected concrete techniques', count: 1 },
            { label: 'compatible logical seed groups', count: 4 },
          ],
          is_baseline: false,
          note: null,
        },
        {
          label: 'technique_beta',
          count: 2,
          factors: [
            { label: 'selected concrete techniques', count: 1 },
            { label: 'compatible logical seed groups', count: 2 },
          ],
          is_baseline: false,
          note: null,
        },
      ],
    }))

    const equation = screen.getByTestId('run-calculation')
    expect(within(equation).getByText('objectives · Technique alpha')).toBeInTheDocument()
    expect(within(equation).getByText('objectives · Technique beta')).toBeInTheDocument()
    expect(equation).toHaveTextContent('4objectives · Technique alpha+2objectives · Technique beta=6planned attacks')
  })

  it('uses parentheses to make baseline precedence explicit', () => {
    renderDetails(makeEstimate({
      total_attack_count: 20,
      components: [
        ...makeEstimate().components,
        {
          label: 'Baseline',
          count: 4,
          factors: [{ label: 'selected logical seed groups', count: 4 }],
          is_baseline: true,
          note: null,
        },
      ],
    }))

    const equation = screen.getByTestId('run-calculation')
    expect(equation).toHaveTextContent('(4techniques×4objectives)+4direct baseline sends=20planned attacks')
    expect(screen.getByRole('group', {
      name: '( 4 techniques multiplied by 4 objectives ) plus 4 direct baseline sends equals 20 planned attacks.',
    })).toBeInTheDocument()
  })

  it('keeps guaranteed and target-conditional terms in one bounded equation', () => {
    renderDetails(makeEstimate({
      status: 'conditional',
      total_attack_count: null,
      minimum_attack_count: 12,
      maximum_attack_count: 20,
      condition: 'target_capabilities',
      components: [
        {
          label: 'Baseline',
          count: 4,
          factors: [{ label: 'objectives', count: 4 }],
          is_baseline: true,
          note: null,
        },
        {
          label: 'Inline jailbreak delivery',
          count: 8,
          factors: [
            { label: 'objectives', count: 4 },
            { label: 'jailbreak templates', count: 2 },
          ],
          is_baseline: false,
          note: null,
        },
        {
          label: 'Native system-prompt jailbreak delivery',
          count: 8,
          factors: [
            { label: 'objectives', count: 4 },
            { label: 'jailbreak templates', count: 2 },
          ],
          is_baseline: false,
          condition: 'target_capabilities',
          note: null,
        },
      ],
    }))

    const equation = screen.getByTestId('run-calculation')
    expect(equation).toHaveTextContent('4objectives · Inline jailbreak delivery')
    expect(equation).toHaveTextContent('4objectives · Native system-prompt jailbreak delivery · if supported')
    expect(equation).toHaveTextContent('4direct baseline sends')
    expect(equation).toHaveTextContent('12–20planned attacks')
  })

  it('shows adaptive progress objectives and the bounded underlying attempt work', () => {
    const estimate = makeEstimate({
      status: 'conditional',
      total_attack_count: null,
      components: [],
      adaptive_details: {
        objective_count: 21,
        selected_candidate_technique_count: 2,
        candidate_technique_count: 2,
        max_attempts_per_objective: 3,
        techniques_per_objective_upper_bound: 2,
        technique_attempt_count_upper_bound: 42,
        stop_on_first_success: true,
        compatibility_may_reduce_attempts: true,
      },
    })
    const state = mapScenarioRunEstimate(estimate, 'request')
    render(
      <TestWrapper>
        <ScenarioRunEstimateSummary state={state} />
        <ScenarioRunEstimateDetails state={state} />
      </TestWrapper>,
    )

    expect(screen.getByText('21 objectives · up to 42 technique attempts')).toBeInTheDocument()
    expect(screen.getByRole('group', {
      name: '21 objectives multiplied by up to 2 techniques per objective, the smaller of 2 compatible candidates and limit 3, equals up to 42 technique attempts.',
    })).toBeInTheDocument()
    expect(screen.getByText('min(2 compatible candidates, limit 3)')).toBeInTheDocument()
    expect(screen.getByText(
      'Progress tracks 21 objectives. Adaptive technique-attempt counts exclude multi-turn target exchanges and retries. Each adaptive objective stops after the first successful technique. Compatibility may reduce how many candidates each objective can try.',
    )).toBeInTheDocument()
  })

  it('adds direct baseline sends to Adaptive progress without inflating technique attempts', () => {
    const estimate = makeEstimate({
      status: 'conditional',
      total_attack_count: null,
      minimum_attack_count: 21,
      maximum_attack_count: 42,
      components: [
        {
          label: 'Baseline',
          count: 21,
          factors: [{ label: 'objectives', count: 21 }],
          is_baseline: true,
          note: null,
        },
        {
          label: 'Adaptive objectives',
          count: 21,
          factors: [{ label: 'compatible objectives', count: 21 }],
          is_baseline: false,
          note: null,
        },
      ],
      adaptive_details: {
        objective_count: 21,
        selected_candidate_technique_count: 2,
        candidate_technique_count: 2,
        max_attempts_per_objective: 3,
        techniques_per_objective_upper_bound: 2,
        technique_attempt_count_upper_bound: 42,
        stop_on_first_success: true,
        compatibility_may_reduce_attempts: true,
      },
    })
    const state = mapScenarioRunEstimate(estimate, 'request')
    render(
      <TestWrapper>
        <ScenarioRunEstimateSummary state={state} />
        <ScenarioRunEstimateDetails state={state} />
      </TestWrapper>,
    )

    expect(screen.getByText('21–42 planned attacks · up to 42 technique attempts')).toBeInTheDocument()
    expect(screen.getByText(
      'Attempt ceiling: 21 direct baseline sends + up to 42 adaptive technique attempts = up to 63 attack starts.',
    )).toBeInTheDocument()
    expect(screen.getByText(
      'Progress tracks 21–42 planned attacks: 21 direct baseline sends plus up to 21 adaptive objectives. Adaptive technique-attempt counts exclude multi-turn target exchanges and retries. Each adaptive objective stops after the first successful technique. Compatibility may reduce how many candidates each objective can try.',
    )).toBeInTheDocument()
    expect(screen.queryByText(/objective envelope|logical seed groups|selected seed groups/i)).not.toBeInTheDocument()
  })

  it('uses the configured max when it is lower than the adaptive candidate pool', () => {
    renderDetails(makeEstimate({
      status: 'conditional',
      total_attack_count: null,
      components: [],
      adaptive_details: {
        objective_count: 21,
        selected_candidate_technique_count: 14,
        candidate_technique_count: 5,
        max_attempts_per_objective: 3,
        techniques_per_objective_upper_bound: 3,
        technique_attempt_count_upper_bound: 63,
        stop_on_first_success: true,
        compatibility_may_reduce_attempts: true,
      },
    }))

    expect(screen.getByText('min(5 compatible candidates from 14 selected, limit 3)')).toBeInTheDocument()
    expect(screen.getByText('up to 63')).toBeInTheDocument()
  })

  it('uses the candidate pool when it is lower than the adaptive max', () => {
    renderDetails(makeEstimate({
      status: 'conditional',
      total_attack_count: null,
      components: [],
      adaptive_details: {
        objective_count: 21,
        selected_candidate_technique_count: 2,
        candidate_technique_count: 2,
        max_attempts_per_objective: 5,
        techniques_per_objective_upper_bound: 2,
        technique_attempt_count_upper_bound: 42,
        stop_on_first_success: true,
        compatibility_may_reduce_attempts: true,
      },
    }))

    expect(screen.getByText('techniques per objective')).toBeInTheDocument()
    expect(screen.getByText('min(2 compatible candidates, limit 5)')).toBeInTheDocument()
    expect(screen.getByText('up to 42')).toBeInTheDocument()
  })

  it('preserves loading, unavailable, stale, and unknown conditional states', () => {
    const loading: ScenarioRunEstimateState = { status: 'loading', scope: 'request' }
    const { rerender } = render(
      <TestWrapper><ScenarioRunEstimateDetails state={loading} /></TestWrapper>,
    )
    expect(screen.getByText('Calculating planned attacks...')).toBeInTheDocument()

    const unavailable = mapScenarioRunEstimate(makeEstimate({
      status: 'unavailable',
      total_attack_count: null,
      components: [],
      datasets: [],
      note: 'Target capability is not available.',
    }), 'request')
    rerender(<TestWrapper><ScenarioRunEstimateDetails state={unavailable} /></TestWrapper>)
    expect(screen.getByText('Estimate unavailable')).toBeInTheDocument()
    expect(screen.getByText('Configured run size unavailable')).toBeInTheDocument()

    const exact = mapScenarioRunEstimate(makeEstimate(), 'request')
    if (exact.status !== 'available') {
      throw new Error('Expected exact estimate.')
    }
    const stale: ScenarioRunEstimateState = {
      status: 'stale',
      estimate: exact.estimate,
      label: 'Showing the last successful estimate.',
      error: 'Preview service timed out.',
    }
    rerender(<TestWrapper><ScenarioRunEstimateDetails state={stale} /></TestWrapper>)
    expect(screen.getByText('Previous estimate not shown')).toBeInTheDocument()
    expect(screen.queryByTestId('run-calculation')).not.toBeInTheDocument()
    expect(screen.getByText(
      'The previous calculation does not match the current configuration.',
    )).toBeInTheDocument()
    expect(screen.getByText('Preview service timed out.')).toBeInTheDocument()

    rerender(
      <TestWrapper>
        <ScenarioRunEstimateDetails state={mapScenarioRunEstimate(makeEstimate({
          status: 'conditional',
          total_attack_count: null,
          minimum_attack_count: null,
          maximum_attack_count: null,
          components: [],
          datasets: [],
        }), 'default')}
        />
      </TestWrapper>,
    )
    expect(screen.getByText('Exact total')).toBeInTheDocument()
    expect(screen.getByText('unavailable')).toBeInTheDocument()
  })

  it('does not render implementation terminology in the shared estimate surfaces', () => {
    const state = mapScenarioRunEstimate(makeEstimate(), 'request')
    render(
      <TestWrapper>
        <ScenarioRunEstimateSummary state={state} />
        <ScenarioRunEstimateDetails state={state} />
      </TestWrapper>,
    )

    expect(screen.queryByText(/logical seed groups/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/selected seed groups/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/planned components/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/objective envelopes/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/how this count is calculated/i)).not.toBeInTheDocument()
  })
})
