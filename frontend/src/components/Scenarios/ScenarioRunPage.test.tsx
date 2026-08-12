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
      '1 of 1 executable units completed',
    )
    expect(screen.getByRole('table', { name: 'Atomic attack groups' })).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Logical seed groups' })).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Persisted attack attempts' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Cancel run' })).toBeInTheDocument()
    expect(screen.queryByRole('columnheader', { name: 'Actions' })).not.toBeInTheDocument()
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

    expect(screen.getByText(/legacy run has no complete persisted execution plan/i)).toBeInTheDocument()
    expect(screen.getAllByText(/1 known completed units; planned total unavailable/i)).toHaveLength(2)
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
      name: 'View details for attack attempt attack-result-1',
    })

    await user.click(detailsButton)
    const dialog = screen.getByRole('dialog', { name: 'Attack attempt details' })
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
    const attemptsTable = screen.getByRole('table', { name: 'Persisted attack attempts' })
    expect(within(attemptsTable).getByRole('columnheader', { name: 'Attack' })).toBeInTheDocument()
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
      name: 'View details for attack attempt attack-result-1',
    }))
    expect(screen.getByRole('dialog', { name: 'Attack attempt details' })).toBeInTheDocument()
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
