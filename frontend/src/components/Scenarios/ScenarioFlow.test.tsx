import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'

import { useScenarioRunProgress } from '@/hooks/useScenarioRunProgress'
import { scenariosApi, targetsApi } from '@/services/api'
import type {
  RegisteredScenario,
  ScenarioDefaultRunSizeEstimate,
  TargetInstance,
} from '@/types'
import type { ScenarioRunProgressState } from '@/utils/scenarioRunProgress'

import ScenarioCatalog from './ScenarioCatalog'
import ScenarioDetail from './ScenarioDetail'
import ScenarioRunPage from './ScenarioRunPage'

jest.mock('@/hooks/useScenarioRunProgress', () => ({
  useScenarioRunProgress: jest.fn(),
}))

jest.mock('@/services/api', () => ({
  scenariosApi: {
    cancelRun: jest.fn(),
    estimateRun: jest.fn(),
    getScenario: jest.fn(),
    listCatalog: jest.fn(),
    startRun: jest.fn(),
  },
  targetsApi: {
    listTargets: jest.fn(),
  },
}))

const mockUseScenarioRunProgress = useScenarioRunProgress as jest.Mock
const mockEstimateRun = scenariosApi.estimateRun as jest.Mock
const mockGetScenario = scenariosApi.getScenario as jest.Mock
const mockListCatalog = scenariosApi.listCatalog as jest.Mock
const mockStartRun = scenariosApi.startRun as jest.Mock
const mockListTargets = targetsApi.listTargets as jest.Mock

const SCENARIO_NAME = 'foundry.red_team_agent'
const RUN_ID = '123e4567-e89b-12d3-a456-426614174000'

const SCENARIO: RegisteredScenario = {
  scenario_name: SCENARIO_NAME,
  scenario_type: 'RedTeamAgentScenario',
  scenario_version: 1,
  description: 'Red teams a configured target.',
  description_markdown: 'Red teams a configured target.',
  default_technique: 'default_technique',
  default_techniques: ['crescendo'],
  aggregate_techniques: ['default_technique'],
  aggregate_technique_expansions: {
    default_technique: ['crescendo'],
  },
  all_techniques: ['crescendo'],
  default_datasets: ['harmbench'],
  default_dataset_summaries: [],
  baseline_policy: 'enabled',
  include_baseline_by_default: true,
  supported_parameters: [],
  default_run_size: {
    version: 1,
    status: 'exact',
    total_attack_count: 2,
    minimum_attack_count: null,
    maximum_attack_count: null,
    condition: null,
    components: [],
    datasets: [],
    note: null,
    retries_included: false,
  },
}

const TARGET: TargetInstance = {
  target_registry_name: 'target-a',
  identifier: {
    class_name: 'OpenAIChatTarget',
    hash: 'target-a-hash',
  },
}

const ESTIMATE: ScenarioDefaultRunSizeEstimate = {
  version: 1,
  status: 'exact',
  total_attack_count: 2,
  minimum_attack_count: null,
  maximum_attack_count: null,
  condition: null,
  components: [{
    label: 'Configured attacks',
    count: 2,
    factors: [],
    is_baseline: false,
    note: null,
  }],
  datasets: [],
  note: null,
  retries_included: false,
}

const RUN_STATE: ScenarioRunProgressState = {
  loadStatus: 'ready',
  run: {
    scenario_result_id: RUN_ID,
    scenario_name: 'RedTeamAgentScenario',
    scenario_registry_name: SCENARIO_NAME,
    scenario_version: 1,
    status: 'IN_PROGRESS',
    created_at: '2026-08-07T18:00:00Z',
  },
  plan: {
    version: 1,
    scenario_registry_name: SCENARIO_NAME,
    atomic_groups: [],
    seed_groups: [],
  },
  planComplete: true,
  activeAtomicGroupIds: [],
  results: [],
  cursor: 'cursor-0',
  hasMore: false,
  error: null,
  stale: false,
}

function LocationProbe() {
  const location = useLocation()
  return <output aria-label="Current route">{`${location.pathname}${location.search}`}</output>
}

function renderFlow(): void {
  render(
    <FluentProvider theme={webLightTheme}>
      <MemoryRouter initialEntries={['/scenarios']}>
        <LocationProbe />
        <Routes>
          <Route path="/scenarios" element={<ScenarioCatalog />} />
          <Route
            path="/scenarios/:scenarioName"
            element={(
              <ScenarioDetail
                activeTarget={null}
                labels={{ operator: 'integration-test' }}
                onNavigate={jest.fn()}
              />
            )}
          />
          <Route path="/scenario-history/:scenarioResultId" element={<ScenarioRunPage />} />
        </Routes>
      </MemoryRouter>
    </FluentProvider>,
  )
}

describe('Scenario catalog-to-run integration', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockListCatalog.mockResolvedValue({
      items: [SCENARIO],
      pagination: { limit: 200, has_more: false },
    })
    mockGetScenario.mockResolvedValue(SCENARIO)
    mockListTargets.mockResolvedValue({
      items: [TARGET],
      pagination: { limit: 200, has_more: false },
    })
    mockEstimateRun.mockResolvedValue(ESTIMATE)
    mockStartRun.mockResolvedValue({ scenario_result_id: RUN_ID })
    mockUseScenarioRunProgress.mockReturnValue({
      state: RUN_STATE,
      retry: jest.fn(),
      applyRunSummary: jest.fn(),
    })
  })

  it('carries one configured request from catalog detail through estimate, launch, and run hydration', async () => {
    const user = userEvent.setup()
    renderFlow()

    await user.click(await screen.findByRole('link', { name: SCENARIO_NAME }))
    expect(await screen.findByRole('heading', { level: 1, name: SCENARIO_NAME })).toBeInTheDocument()

    const expectedEstimateRequest = {
      target_name: TARGET.target_registry_name,
      techniques: ['default_technique'],
      include_baseline: true,
    }
    await waitFor(() => expect(mockEstimateRun).toHaveBeenLastCalledWith(
      SCENARIO_NAME,
      expectedEstimateRequest,
      expect.any(AbortSignal),
    ))
    expect(within(screen.getByRole('complementary', { name: 'Run preview' }))
      .getByText('2 planned attacks')).toBeInTheDocument()

    await user.click(screen.getByTestId('launch-scenario-btn'))

    await waitFor(() => expect(mockStartRun).toHaveBeenCalledWith({
      scenario_name: SCENARIO_NAME,
      target_name: TARGET.target_registry_name,
      techniques: expectedEstimateRequest.techniques,
      max_concurrency: 10,
      max_retries: 0,
      include_baseline: expectedEstimateRequest.include_baseline,
      labels: { operator: 'integration-test' },
    }))
    expect(await screen.findByTestId('scenario-run-page')).toBeInTheDocument()
    expect(screen.getByLabelText('Current route')).toHaveTextContent(
      `/scenario-history/${RUN_ID}`,
    )
    expect(screen.getByRole('heading', { level: 1, name: SCENARIO_NAME })).toBeInTheDocument()
  })
})
