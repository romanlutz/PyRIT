import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import {
  MemoryRouter,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from 'react-router'

import { useScenarioRunProgress } from '@/hooks/useScenarioRunProgress'
import { useScenarioQueue } from '@/hooks/useScenarioQueue'
import { scenariosApi } from '@/services/api'
import type {
  ScenarioProgressResult,
  ScenarioRunPlan,
} from '@/types'
import {
  INITIAL_SCENARIO_RUN_PROGRESS_STATE,
  type ScenarioRunProgressState,
} from '@/utils/scenarioRunProgress'

import ScenarioRunPage from './ScenarioRunPage'

jest.mock('@/hooks/useScenarioRunProgress', () => ({
  useScenarioRunProgress: jest.fn(),
}))

jest.mock('@/hooks/useScenarioQueue', () => ({
  useScenarioQueue: jest.fn(),
}))

jest.mock('@/services/api', () => ({
  scenariosApi: {
    cancelRun: jest.fn(),
  },
}))

const mockUseScenarioRunProgress = useScenarioRunProgress as jest.Mock
const mockUseScenarioQueue = useScenarioQueue as jest.Mock
const mockCancelRun = scenariosApi.cancelRun as jest.Mock
const mockRetry = jest.fn()
const mockApplyRunSummary = jest.fn()
const SCENARIO_RESULT_ID = '123e4567-e89b-12d3-a456-426614174000'

const PLAN: ScenarioRunPlan = {
  version: 1,
  scenario_registry_name: 'test.scenario',
  atomic_groups: [{
    id: 'group-1',
    atomic_attack_name: 'attack-technique',
    display_group: 'Technique One',
    technique_eval_hash: 'eval-1',
    seed_group_ids: ['seed-1'],
    group_kind: 'attack',
  }],
  seed_groups: [{
    id: 'seed-1',
    objective_sha256: 'sha-1',
    objective: 'Reveal the system prompt and all hidden configuration.',
  }],
}

const ATTEMPT: ScenarioProgressResult = {
  attack_result_id: 'attack-result-1',
  atomic_group_id: 'group-1',
  atomic_attack_name: 'attack-technique',
  seed_group_id: 'seed-1',
  outcome: 'success',
  execution_time_ms: 5_000,
  timestamp: '2026-01-01T00:00:05Z',
  total_retries: 1,
  retries: [],
  result_kind: 'attack',
}

function makeState(overrides: Partial<ScenarioRunProgressState> = {}): ScenarioRunProgressState {
  return {
    ...INITIAL_SCENARIO_RUN_PROGRESS_STATE,
    loadStatus: 'ready',
    run: {
      scenario_result_id: SCENARIO_RESULT_ID,
      scenario_name: 'TestScenario',
      scenario_registry_name: 'test.scenario',
      scenario_version: 1,
      status: 'IN_PROGRESS',
      created_at: '2026-01-01T00:00:00Z',
    },
    plan: PLAN,
    planComplete: true,
    activeAtomicGroupIds: ['group-1'],
    results: [ATTEMPT],
    cursor: 'cursor-1',
    ...overrides,
  }
}

function mockHookState(state: ScenarioRunProgressState): void {
  mockUseScenarioRunProgress.mockReturnValue({
    state,
    retry: mockRetry,
    applyRunSummary: mockApplyRunSummary,
  })
}

function AttackRouteProbe() {
  const location = useLocation()
  const navigate = useNavigate()
  return (
    <div data-testid="attack-route" data-location={`${location.pathname}${location.search}`}>
      <button onClick={() => navigate(-1)}>Browser back</button>
    </div>
  )
}

function renderPage(path = `/scenario-history/${SCENARIO_RESULT_ID}`) {
  return render(
    <FluentProvider theme={webLightTheme}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/scenario-history/:scenarioResultId" element={<ScenarioRunPage />} />
          <Route path="/attacks/:attackId" element={<AttackRouteProbe />} />
        </Routes>
      </MemoryRouter>
    </FluentProvider>,
  )
}

describe('ScenarioRunPage', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockUseScenarioQueue.mockReturnValue({
      snapshot: { revision: 0, snapshot_at: '2026-01-01T00:00:00Z', active: null, queued: [] },
      loading: false,
      stale: false,
      error: null,
      retry: jest.fn(),
    })
    mockHookState(makeState())
  })

  it('renders a live dashboard with accessible progress and semantic tables', () => {
    renderPage()

    expect(screen.getByRole('heading', { name: 'test.scenario', level: 1 })).toBeInTheDocument()
    expect(screen.getByTestId('run-state-badge')).toHaveTextContent('In progress')
    expect(screen.getByRole('progressbar', { name: 'Overall scenario run progress' })).toHaveAttribute(
      'aria-valuetext',
      '1 of 1 progress units completed',
    )
    expect(screen.getByRole('table', { name: 'Atomic attack groups' })).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Objectives' })).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Target-facing attacks' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Cancel run' })).toBeInTheDocument()
    expect(screen.queryByRole('columnheader', { name: 'Actions' })).not.toBeInTheDocument()
  })

  it('leads with target-facing attacks and keeps orchestration records secondary', async () => {
    const user = userEvent.setup()
    const objectiveIds = ['seed-1', 'seed-2', 'seed-3', 'seed-4']
    const plan: ScenarioRunPlan = {
      version: 1,
      scenario_registry_name: 'adaptive.text',
      seed_groups: objectiveIds.map((id, index) => ({
        id,
        objective_sha256: `sha-${index + 1}`,
        objective: `Objective ${index + 1}`,
      })),
      atomic_groups: [
        {
          id: 'baseline',
          atomic_attack_name: 'baseline',
          display_group: 'Direct baseline',
          technique_eval_hash: 'baseline-eval',
          seed_group_ids: objectiveIds,
          group_kind: 'direct_baseline',
        },
        ...objectiveIds.map((seedId, index) => ({
          id: `adaptive-${index + 1}`,
          atomic_attack_name: 'adaptive',
          display_group: index === 0 ? 'Fairness' : 'Harassment',
          technique_eval_hash: `adaptive-eval-${index + 1}`,
          seed_group_ids: [seedId],
          group_kind: 'adaptive' as const,
        })),
      ],
    }
    const results: ScenarioProgressResult[] = objectiveIds.flatMap((seedId, index) => {
      const adaptiveGroupId = `adaptive-${index + 1}`
      return [
        {
          ...ATTEMPT,
          attack_result_id: `baseline-${index}`,
          atomic_group_id: 'baseline',
          atomic_attack_name: 'baseline',
          seed_group_id: seedId,
          timestamp: `2026-01-01T00:${String(index * 3).padStart(2, '0')}:00Z`,
          total_retries: 0,
          result_kind: 'direct_baseline',
        },
        {
          ...ATTEMPT,
          attack_result_id: `technique-${index}`,
          atomic_group_id: adaptiveGroupId,
          atomic_attack_name: 'adaptive',
          seed_group_id: seedId,
          timestamp: `2026-01-01T00:${String(index * 3 + 1).padStart(2, '0')}:00Z`,
          total_retries: 0,
          result_kind: 'adaptive_technique',
          technique_name: index === 0 ? 'Fairness technique' : 'Harassment technique',
          attempt_index: 1,
        },
        {
          ...ATTEMPT,
          attack_result_id: `envelope-${index}`,
          atomic_group_id: adaptiveGroupId,
          atomic_attack_name: 'adaptive',
          seed_group_id: seedId,
          timestamp: `2026-01-01T00:${String(index * 3 + 2).padStart(2, '0')}:00Z`,
          total_retries: 7,
          result_kind: 'aggregate_parent',
        },
      ]
    })
    mockHookState(makeState({
      run: {
        ...makeState().run!,
        scenario_registry_name: 'adaptive.text',
        status: 'COMPLETED',
        completed_at: '2026-01-01T00:15:00Z',
      },
      plan,
      activeAtomicGroupIds: [],
      results,
    }))

    renderPage()

    expect(screen.getByRole('group', {
      name: '4 objectives multiplied by 2 observed attacks each equals 8 target-facing attacks. 8/8 planned progress units completed. 12 persisted result records: 8 target-facing attack results + 4 Adaptive orchestration summaries. 0 actual retries.',
    })).toBeInTheDocument()
    expect(screen.getByText(
      'Per objective: 1 direct baseline + 1 Adaptive technique.',
    )).toBeInTheDocument()
    expect(screen.getByText(
      '8/8 planned progress units completed · 12 persisted result records: 8 target-facing attack results + 4 Adaptive orchestration summaries · 0 actual retries',
    )).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Technique summary' })).not.toBeInTheDocument()
    expect(screen.getAllByText('Fairness technique').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Harassment technique').length).toBeGreaterThan(0)

    const objectiveTable = screen.getByRole('table', { name: 'Objectives' })
    const firstObjectiveCells = within(within(objectiveTable).getAllByRole('row')[1]).getAllByRole('cell')
    expect(firstObjectiveCells[1]).toHaveTextContent('2/2')
    expect(firstObjectiveCells[2]).toHaveTextContent('3')
    expect(firstObjectiveCells[3]).toHaveTextContent('2')
    expect(firstObjectiveCells[6]).toHaveTextContent('0')
    const attackTable = screen.getByRole('table', { name: 'Target-facing attacks' })
    expect(within(attackTable).getAllByRole('row')).toHaveLength(9)
    expect(within(attackTable).queryByText('envelope-0')).not.toBeInTheDocument()

    const disclosure = screen.getByText('Orchestration results (4)')
    expect(disclosure.closest('details')).not.toHaveAttribute('open')
    await user.click(disclosure)

    const orchestrationTable = screen.getByRole('table', { name: 'Orchestration results' })
    expect(within(orchestrationTable).getAllByRole('row')).toHaveLength(5)
    const aggregateRow = within(orchestrationTable).getByText('envelope-0').closest('tr')
    expect(aggregateRow).not.toBeNull()
    expect(within(aggregateRow!).queryByRole('link')).not.toBeInTheDocument()
    expect(aggregateRow).toHaveTextContent('Aggregate parent')
  })

  it('does not force a per-objective equation for nonuniform observed attacks', () => {
    const plan: ScenarioRunPlan = {
      ...PLAN,
      atomic_groups: [{
        ...PLAN.atomic_groups[0],
        seed_group_ids: ['seed-1', 'seed-2'],
      }],
      seed_groups: [
        PLAN.seed_groups[0],
        {
          id: 'seed-2',
          objective_sha256: 'sha-2',
          objective: 'A second objective with a different observed attack count.',
        },
      ],
    }
    mockHookState(makeState({
      run: {
        ...makeState().run!,
        status: 'COMPLETED',
        completed_at: '2026-01-01T00:15:00Z',
      },
      plan,
      activeAtomicGroupIds: [],
      results: [
        { ...ATTEMPT, total_retries: 0 },
        { ...ATTEMPT, attack_result_id: 'attack-result-2', seed_group_id: 'seed-2', total_retries: 0 },
        { ...ATTEMPT, attack_result_id: 'attack-result-3', seed_group_id: 'seed-2', total_retries: 0 },
      ],
    }))

    renderPage()

    expect(screen.getByRole('group', {
      name: '3 target-facing attacks. 2/2 planned progress units completed. 3 persisted result records. 1 actual retries.',
    })).toBeInTheDocument()
    expect(screen.queryByText('observed attacks each')).not.toBeInTheDocument()
    expect(screen.queryByText(/^Per objective:/)).not.toBeInTheDocument()
  })

  it('accounts for unclassified legacy records in persisted storage provenance', () => {
    mockHookState(makeState({
      results: [
        { ...ATTEMPT, total_retries: 0 },
        {
          ...ATTEMPT,
          attack_result_id: 'legacy-unknown',
          atomic_group_id: 'legacy-group',
          atomic_attack_name: 'legacy-attack',
          result_kind: 'unknown',
          total_retries: 0,
        },
      ],
    }))

    renderPage()

    expect(screen.getByRole('group', {
      name: '1 objective multiplied by 1 observed attack each equals 1 target-facing attack. 1/1 planned progress units completed. 2 persisted result records: 1 target-facing attack result + 1 unclassified record. 0 actual retries.',
    })).toBeInTheDocument()
  })

  it('renders contract-backed safe target and run configuration metadata', () => {
    mockHookState(makeState({
      run: {
        ...makeState().run!,
        target: {
          target_type: 'OpenAIChatTarget',
          endpoint: 'https://example.test/v1',
          model_name: 'gpt-4o',
          identifier_hash: 'safe-hash',
        },
        techniques_used: ['Technique One'],
        datasets_used: ['harmbench'],
        scenario_parameters: { max_turns: 5 },
        labels: { operator: 'alice' },
        pyrit_version: '0.10.0',
      },
    }))

    renderPage()

    expect(screen.getByText('gpt-4o')).toBeInTheDocument()
    expect(screen.getByText('https://example.test/v1')).toBeInTheDocument()
    expect(screen.getByText('safe-hash')).toBeInTheDocument()
    expect(screen.getByText('harmbench')).toBeInTheDocument()
    expect(screen.getByText('max_turns: 5')).toBeInTheDocument()
    expect(screen.getByText('operator: alice')).toBeInTheDocument()
    expect(screen.getByText('0.10.0')).toBeInTheDocument()
  })

  it('keeps legacy runs useful without misleading totals, ETA, or a progress bar', () => {
    mockHookState(makeState({ planComplete: false }))

    renderPage()

    expect(screen.getByText(/legacy run has no complete persisted progress plan/i)).toBeInTheDocument()
    expect(screen.getAllByText(/1 known completed progress units; planned total unavailable/i)).toHaveLength(3)
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
    expect(screen.getByText('Progress percentage unavailable')).toBeInTheDocument()
    expect(screen.getAllByText('Unavailable').length).toBeGreaterThan(0)
    expect(screen.getAllByText('1/total unavailable').length).toBeGreaterThan(0)
    expect(screen.queryByText('1/1')).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open attack attack-result-1' })).toBeInTheDocument()
  })

  it('shows a stale warning and retries from the explicit action', async () => {
    const user = userEvent.setup()
    mockHookState(makeState({ stale: true, error: 'Network unavailable' }))

    renderPage()
    await user.click(screen.getByRole('button', { name: 'Retry' }))

    expect(mockRetry).toHaveBeenCalledTimes(1)
    expect(screen.getByText(/showing the last successfully loaded progress/i)).toBeInTheDocument()
  })

  it('cancels a queued run after confirmation and immediately applies the terminal state', async () => {
    const user = userEvent.setup()
    const cancelledRun = {
      scenario_result_id: 'run-1',
      scenario_name: 'TestScenario',
      scenario_registry_name: 'test.scenario',
      scenario_version: 1,
      status: 'CANCELLED',
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:01:00Z',
      completed_at: '2026-01-01T00:01:00Z',
      techniques_used: [],
      total_attacks: 1,
      completed_attacks: 1,
      objective_achieved_rate: 100,
      failed_attacks: [],
      attack_retries: [],
      total_retries: 0,
      labels: {},
    }
    mockCancelRun.mockResolvedValueOnce(cancelledRun)
    mockHookState(makeState({
      run: {
        ...makeState().run!,
        status: 'QUEUED',
        queue_position: 1,
        active_scenario_result_id: 'active-run',
      },
      results: [],
      activeAtomicGroupIds: [],
    }))

    renderPage()
    await user.click(screen.getByRole('button', { name: 'Cancel run' }))
    const dialog = screen.getByRole('dialog', { name: 'Cancel this scenario run?' })
    expect(within(dialog).getByText(/removed from the queue and will never execute/i)).toBeInTheDocument()
    await user.click(within(dialog).getByRole('button', { name: 'Cancel run' }))

    await waitFor(() => expect(mockApplyRunSummary).toHaveBeenCalledWith(cancelledRun))
    expect(mockCancelRun).toHaveBeenCalledWith(SCENARIO_RESULT_ID)
  })

  it('keeps the confirmation open and shows cancel conflicts', async () => {
    const user = userEvent.setup()
    mockCancelRun.mockRejectedValueOnce(new Error('Cannot cancel a completed run.'))

    renderPage()
    await user.click(screen.getByRole('button', { name: 'Cancel run' }))
    const dialog = screen.getByRole('dialog', { name: 'Cancel this scenario run?' })
    await user.click(within(dialog).getByRole('button', { name: 'Cancel run' }))

    expect(await within(dialog).findByText('Cannot cancel a completed run.')).toBeInTheDocument()
    expect(mockApplyRunSummary).not.toHaveBeenCalled()
  })

  it('shows full objective details and restores focus on close', async () => {
    const user = userEvent.setup()
    renderPage()
    const detailsButton = screen.getByRole('button', {
      name: 'View details for result record attack-result-1',
    })

    await user.click(detailsButton)
    const dialog = screen.getByRole('dialog', { name: 'Result record details' })
    expect(within(dialog).getByText(PLAN.seed_groups[0].objective)).toBeInTheDocument()
    await user.click(within(dialog).getByRole('button', { name: 'Close' }))

    await waitFor(() => expect(detailsButton).toHaveFocus())
  })

  it('puts the essential attack link in the first column with bounded provenance', () => {
    renderPage()

    const attackLink = screen.getByRole('link', { name: 'Open attack attack-result-1' })
    expect(attackLink).toHaveAttribute(
      'href',
      `/attacks/attack-result-1?scenarioResultId=${SCENARIO_RESULT_ID}`,
    )
    expect(attackLink).toHaveTextContent('attack-result-1')
    const attemptsTable = screen.getByRole('table', { name: 'Target-facing attacks' })
    expect(within(attemptsTable).getByRole('columnheader', { name: 'Result record' })).toBeInTheDocument()
    const firstBodyRow = within(attemptsTable).getAllByRole('row')[1]
    expect(within(firstBodyRow).getAllByRole('cell')[0]).toContainElement(
      attackLink,
    )
  })

  it('navigates from non-interactive row content and browser Back returns to the run', async () => {
    const user = userEvent.setup()
    renderPage()

    const attemptRow = screen.getByRole('row', {
      name: 'Open attack attack-result-1',
    })
    await user.click(within(attemptRow).getByText('Technique One'))

    expect(screen.getByTestId('attack-route')).toBeInTheDocument()
    expect(screen.getByTestId('attack-route')).toHaveAttribute(
      'data-location',
      `/attacks/attack-result-1?scenarioResultId=${SCENARIO_RESULT_ID}`,
    )

    await user.click(screen.getByRole('button', { name: 'Browser back' }))

    expect(screen.getByRole('heading', { name: 'test.scenario', level: 1 })).toBeInTheDocument()
  })

  it('supports Enter and Space row activation', async () => {
    const user = userEvent.setup()
    renderPage()
    const row = screen.getByRole('row', { name: 'Open attack attack-result-1' })

    row.focus()
    await user.keyboard('{Enter}')
    expect(screen.getByTestId('attack-route')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Browser back' }))

    const restoredRow = screen.getByRole('row', { name: 'Open attack attack-result-1' })
    restoredRow.focus()
    await user.keyboard(' ')
    expect(screen.getByTestId('attack-route')).toBeInTheDocument()
  })

  it('does not hijack modified, non-primary, or nested-control clicks', async () => {
    const user = userEvent.setup()
    renderPage()
    const row = screen.getByRole('row', { name: 'Open attack attack-result-1' })

    fireEvent.click(row, { ctrlKey: true })
    fireEvent.click(row, { metaKey: true })
    fireEvent.click(row, { shiftKey: true })
    fireEvent.click(row, { altKey: true })
    fireEvent.click(row, { button: 1 })
    expect(screen.queryByTestId('attack-route')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', {
      name: 'View details for result record attack-result-1',
    }))
    expect(screen.getByRole('dialog', { name: 'Result record details' })).toBeInTheDocument()
    expect(screen.queryByTestId('attack-route')).not.toBeInTheDocument()
  })

  it('leaves modified first-column link clicks to native new-tab behavior', () => {
    renderPage()
    const link = screen.getByRole('link', { name: 'Open attack attack-result-1' })
    const modifiedClick = new MouseEvent('click', {
      bubbles: true,
      cancelable: true,
      ctrlKey: true,
    })

    expect(link.dispatchEvent(modifiedClick)).toBe(true)
    expect(modifiedClick.defaultPrevented).toBe(false)
    expect(screen.queryByTestId('attack-route')).not.toBeInTheDocument()
  })

  it('renders loading, not-found, and initial error states with accessible recovery', () => {
    mockHookState({ ...INITIAL_SCENARIO_RUN_PROGRESS_STATE })
    const { unmount } = renderPage()
    expect(screen.getByLabelText('Loading scenario run')).toBeInTheDocument()
    unmount()

    mockHookState({
      ...INITIAL_SCENARIO_RUN_PROGRESS_STATE,
      loadStatus: 'not-found',
      error: 'Run not found',
    })

    const notFound = renderPage()
    expect(screen.getByRole('heading', { name: 'Scenario run not found' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
    notFound.unmount()

    mockHookState({
      ...INITIAL_SCENARIO_RUN_PROGRESS_STATE,
      loadStatus: 'error',
      error: 'Backend unavailable',
    })
    renderPage()
    expect(screen.getByRole('heading', { name: 'Unable to load scenario run' })).toBeInTheDocument()
    expect(screen.getByText('Backend unavailable')).toBeInTheDocument()
  })

  it('renders queued position without progress percentage or ETA', () => {
    mockHookState(makeState({
      run: {
        ...makeState().run!,
        status: 'QUEUED',
        queue_position: 2,
        active_scenario_result_id: 'active-run',
      },
      results: [],
      activeAtomicGroupIds: [],
    }))

    renderPage()

    expect(screen.getByTestId('run-state-badge')).toHaveTextContent('Queued')
    expect(screen.getByTestId('queued-run-progress')).toHaveTextContent('Position 2')
    expect(screen.getByText(/waiting for active run active-run/i)).toBeInTheDocument()
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
    expect(screen.getByText('Available after start')).toBeInTheDocument()
  })

  it('shows structured overload roles, counts, and non-adaptive retry guidance', () => {
    mockHookState(makeState({
      overloadSummaries: [{
        component_role: 'adversarial_chat',
        count: 3,
        rate_limit_count: 2,
        server_error_count: 1,
        status_codes: [429, 503],
        latest_timestamp: '2026-01-01T00:00:06Z',
      }],
    }))

    renderPage()

    const warning = screen.getByTestId('scenario-overload-warning')
    expect(warning).toHaveTextContent('Adversarial chat')
    expect(warning).toHaveTextContent('3 × HTTP 429/503')
    expect(warning).toHaveTextContent(/without adaptive throttling/i)
  })

  it('decodes route IDs and does not offer cancellation for terminal runs', () => {
    mockHookState(makeState({
      run: {
        scenario_result_id: 'run/1',
        scenario_name: 'TestScenario',
        scenario_registry_name: 'test.scenario',
        scenario_version: 1,
        status: 'COMPLETED',
        created_at: '2026-01-01T00:00:00Z',
        completed_at: '2026-01-01T00:01:00Z',
      },
    }))

    renderPage('/scenario-history/run%2F1')

    expect(mockUseScenarioRunProgress).toHaveBeenCalledWith('run/1')
    expect(screen.queryByRole('button', { name: 'Cancel run' })).not.toBeInTheDocument()
  })
})
