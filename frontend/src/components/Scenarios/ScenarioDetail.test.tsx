import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

import { scenariosApi, targetsApi } from '@/services/api'
import type { RegisteredScenario, TargetInstance } from '@/types'

import ScenarioDetail from './ScenarioDetail'

jest.mock('@/services/api', () => ({
  scenariosApi: {
    estimateRun: jest.fn(),
    getScenario: jest.fn(),
    startRun: jest.fn(),
  },
  targetsApi: {
    listTargets: jest.fn(),
  },
}))

const mockGetScenario = scenariosApi.getScenario as jest.Mock
const mockEstimateRun = scenariosApi.estimateRun as jest.Mock
const mockStartRun = scenariosApi.startRun as jest.Mock
const mockListTargets = targetsApi.listTargets as jest.Mock

const mockNavigate = jest.fn()
const RAW_IMAGE_HTML = ['<', 'img src=x onerror="alert(1)">'].join('')

jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}))

function makeScenario(overrides: Partial<RegisteredScenario> = {}): RegisteredScenario {
  return {
    scenario_name: 'foundry.red_team_agent',
    scenario_type: 'RedTeamAgentScenario',
    description: 'Red teams a target.',
    default_technique: 'default_technique',
    aggregate_techniques: ['default_technique'],
    all_techniques: ['default_technique', 'crescendo'],
    default_datasets: ['harmbench'],
    baseline_policy: 'enabled',
    include_baseline_by_default: true,
    supported_parameters: [],
    ...overrides,
  }
}

function makeTarget(name: string): TargetInstance {
  return {
    target_registry_name: name,
    identifier: { class_name: 'OpenAIChatTarget', hash: `${name}-hash` },
  }
}

function renderDetail(
  path: string,
  props: Partial<{
    activeTarget: TargetInstance | null
    labels: Record<string, string>
    onNavigate: (view: string) => void
  }> = {},
) {
  const defaultProps = {
    activeTarget: null,
    labels: { operator: 'roakey' },
    onNavigate: jest.fn(),
  }
  const merged = { ...defaultProps, ...props }
  return render(
    <FluentProvider theme={webLightTheme}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route
            path="/scenarios/:scenarioName"
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            element={<ScenarioDetail {...(merged as any)} />}
          />
        </Routes>
      </MemoryRouter>
    </FluentProvider>,
  )
}

describe('ScenarioDetail', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockGetScenario.mockReset()
    mockEstimateRun.mockReset()
    mockListTargets.mockReset()
    mockStartRun.mockReset()
    mockListTargets.mockResolvedValue({
      items: [makeTarget('target-a'), makeTarget('target-b')],
      pagination: { limit: 200, has_more: false },
    })
    mockGetScenario.mockResolvedValue(makeScenario())
    mockEstimateRun.mockReturnValue(new Promise(() => {}))
    mockStartRun.mockResolvedValue({ scenario_result_id: 'sr-default' })
  })

  it('shows a loading state while fetching', () => {
    mockGetScenario.mockReturnValue(new Promise(() => {}))
    mockListTargets.mockReturnValue(new Promise(() => {}))
    renderDetail('/scenarios/foundry.red_team_agent')
    expect(screen.getByText('Loading scenario...')).toBeInTheDocument()
  })

  it('decodes the scenario name from the URL exactly once', async () => {
    renderDetail('/scenarios/foundry.red_team_agent');
    await screen.findByTestId('scenario-target-select')
    expect(mockGetScenario).toHaveBeenCalledWith('foundry.red_team_agent')
  })

  it('decodes a slash-bearing encoded scenario name back to the original', async () => {
    renderDetail('/scenarios/foundry%2Fred_team_agent')
    await waitFor(() => expect(mockGetScenario).toHaveBeenCalledWith('foundry/red_team_agent'))
  })

  it('preserves a literal percent sequence in a scenario registry name', async () => {
    renderDetail('/scenarios/discount%2550')
    await waitFor(() => expect(mockGetScenario).toHaveBeenCalledWith('discount%50'))
  })

  it('handles a malformed percent sequence without throwing during render', async () => {
    const consoleWarn = jest.spyOn(console, 'warn').mockImplementation(() => {})
    mockGetScenario.mockRejectedValueOnce({
      isAxiosError: true,
      response: { status: 404, data: { detail: 'not found' } },
    })
    renderDetail('/scenarios/%zz')
    expect(await screen.findByTestId('scenario-not-found')).toBeInTheDocument()
    expect(mockGetScenario).toHaveBeenCalledWith('%zz')
    consoleWarn.mockRestore()
  })

  it('shows a distinct not-found state for a 404, with a link back to the catalog', async () => {
    mockGetScenario.mockRejectedValueOnce({
      isAxiosError: true,
      response: { status: 404, data: { detail: 'not found' } },
    })

    renderDetail('/scenarios/missing.scenario')

    expect(await screen.findByTestId('scenario-not-found')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /back to scenarios/i })).toHaveAttribute('href', '/scenarios')
    expect(screen.queryByTestId('scenario-error')).not.toBeInTheDocument()
  })

  it('shows a generic error state with retry for a non-404 failure', async () => {
    const user = userEvent.setup()
    mockGetScenario
      .mockRejectedValueOnce({ isAxiosError: true, response: { status: 500, data: { detail: 'boom' } } })
      .mockResolvedValueOnce(makeScenario())

    renderDetail('/scenarios/foundry.red_team_agent')

    expect(await screen.findByTestId('scenario-error')).toBeInTheDocument()
    expect(screen.getByText('boom')).toBeInTheDocument()
    expect(screen.queryByTestId('scenario-not-found')).not.toBeInTheDocument()

    await user.click(screen.getByTestId('retry-btn'))
    expect(await screen.findByTestId('scenario-target-select')).toBeInTheDocument()
  })

  it('shows a no-targets state directing to Configuration when none are registered', async () => {
    const onNavigate = jest.fn()
    mockListTargets.mockResolvedValueOnce({ items: [], pagination: { limit: 200, has_more: false } })

    renderDetail('/scenarios/foundry.red_team_agent', { onNavigate })

    const user = userEvent.setup()
    expect(await screen.findByTestId('no-targets-state')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Configure target' }))
    expect(onNavigate).toHaveBeenCalledWith('config')
  })

  it('defaults the target selector to the active target when it is among the fetched targets', async () => {
    renderDetail('/scenarios/foundry.red_team_agent', { activeTarget: makeTarget('target-b') })

    expect(await screen.findByTestId('scenario-target-select')).toHaveValue('target-b')
  })

  it('defaults the target selector to the first fetched target when there is no matching active target', async () => {
    renderDetail('/scenarios/foundry.red_team_agent')

    expect(await screen.findByTestId('scenario-target-select')).toHaveValue('target-a')
  })

  it('exposes the configuration form and run preview as ordered landmarks', async () => {
    renderDetail('/scenarios/foundry.red_team_agent')

    expect(await screen.findByRole('form', { name: 'Scenario run configuration' })).toBeInTheDocument()
    expect(screen.getByRole('complementary', { name: 'Run preview' })).toBeInTheDocument()
  })

  it('renders MyST literals through the shared safe Markdown renderer', async () => {
    mockGetScenario.mockResolvedValue(
      makeScenario({
        description: `Set \`\`num_jailbreaks\`\`.\n\n${RAW_IMAGE_HTML}unsafe`,
      }),
    )
    renderDetail('/scenarios/foundry.red_team_agent')

    const description = await screen.findByTestId('scenario-detail-description')
    expect(within(description).getByText('num_jailbreaks').tagName).toBe('CODE')
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(
      within(description).getByText((content: string) => content.includes(`${RAW_IMAGE_HTML}unsafe`)),
    ).toBeInTheDocument()
  })

  it('initializes the technique selection from default_technique', async () => {
    renderDetail('/scenarios/foundry.red_team_agent')

    await screen.findByTestId('scenario-target-select')
    expect(screen.getByTestId('technique-default_technique')).toBeChecked()
    expect(screen.getByTestId('technique-crescendo')).not.toBeChecked()
  })

  it('switches from the default preset to a multi-technique custom selection', async () => {
    mockGetScenario.mockResolvedValue(
      makeScenario({
        aggregate_techniques: ['default_technique', 'all_garak'],
        all_techniques: ['default_technique', 'crescendo', 'prompt_sending', 'all_garak'],
      }),
    )
    const user = userEvent.setup()

    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    // 'all_garak' is both an aggregate and (accidentally) listed under all_techniques —
    // it must render exactly once (deduped), under the aggregate group.
    expect(screen.getAllByTestId('technique-all_garak')).toHaveLength(1)

    await user.click(screen.getByTestId('technique-crescendo'))
    expect(screen.getByTestId('technique-default_technique')).not.toBeChecked()
    expect(screen.getByTestId('technique-crescendo')).toBeChecked()

    await user.click(screen.getByTestId('technique-prompt_sending'))
    await user.click(screen.getByTestId('launch-scenario-btn'))

    await waitFor(() => expect(mockStartRun).toHaveBeenCalled())
    const request = mockStartRun.mock.calls[0][0]
    expect(request.techniques).toEqual(['crescendo', 'prompt_sending'])
    expect(new Set(request.techniques).size).toBe(request.techniques.length)
  })

  it('selecting a preset replaces the custom concrete list', async () => {
    mockGetScenario.mockResolvedValue(
      makeScenario({
        aggregate_techniques: ['default_technique', 'all_garak'],
        all_techniques: ['default_technique', 'crescendo'],
      }),
    )
    const user = userEvent.setup()
    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    await user.click(screen.getByTestId('technique-crescendo'))
    await user.click(screen.getByTestId('technique-all_garak'))
    expect(screen.getByTestId('technique-all_garak')).toBeChecked()
    expect(screen.getByTestId('technique-crescendo')).not.toBeChecked()

    await user.click(screen.getByTestId('launch-scenario-btn'))
    await waitFor(() => expect(mockStartRun).toHaveBeenCalled())
    expect(mockStartRun.mock.calls[0][0].techniques).toEqual(['all_garak'])
  })

  it('initializes a concrete default as custom and allows adding another concrete technique', async () => {
    mockGetScenario.mockResolvedValue(
      makeScenario({
        default_technique: 'prompt_sending',
        aggregate_techniques: ['all_garak'],
        all_techniques: ['prompt_sending', 'crescendo'],
      }),
    )
    const user = userEvent.setup()
    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    expect(screen.getByTestId('technique-prompt_sending')).toBeChecked()
    await user.click(screen.getByTestId('technique-crescendo'))
    expect(screen.getByTestId('technique-prompt_sending')).toBeChecked()
    expect(screen.getByTestId('technique-crescendo')).toBeChecked()

    await user.click(screen.getByTestId('launch-scenario-btn'))
    await waitFor(() => expect(mockStartRun).toHaveBeenCalled())
    expect(mockStartRun.mock.calls[0][0].techniques).toEqual(['prompt_sending', 'crescendo'])
  })

  it('keeps an explicit invalid custom state when the last concrete technique is removed', async () => {
    const user = userEvent.setup()
    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    await user.click(screen.getByTestId('technique-crescendo'))
    await user.click(screen.getByTestId('technique-crescendo'))

    expect(await screen.findByRole('alert')).toHaveTextContent('Select at least one technique.')
    expect(screen.getByTestId('technique-default_technique')).not.toBeChecked()
    expect(screen.getByTestId('launch-scenario-btn')).toBeDisabled()
    expect(mockStartRun).not.toHaveBeenCalled()
  })

  it('defaults the baseline checkbox from include_baseline_by_default when enabled, and allows editing', async () => {
    const user = userEvent.setup()
    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    const checkbox = screen.getByTestId('baseline-checkbox')
    expect(checkbox).toBeChecked()

    await user.click(checkbox)
    await user.click(screen.getByTestId('launch-scenario-btn'))

    await waitFor(() => expect(mockStartRun).toHaveBeenCalled())
    expect(mockStartRun.mock.calls[0][0].include_baseline).toBe(false)
  })

  it('defaults the baseline checkbox to unchecked when the policy is disabled with include_baseline_by_default false', async () => {
    mockGetScenario.mockResolvedValue(
      makeScenario({ baseline_policy: 'disabled', include_baseline_by_default: false }),
    )
    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    expect(screen.getByTestId('baseline-checkbox')).not.toBeChecked()
  })

  it('disables and forces the baseline checkbox false when the policy is forbidden', async () => {
    mockGetScenario.mockResolvedValue(makeScenario({ baseline_policy: 'forbidden' }))
    const user = userEvent.setup()

    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    const checkbox = screen.getByTestId('baseline-checkbox')
    expect(checkbox).toBeDisabled()
    expect(checkbox).not.toBeChecked()

    await user.click(screen.getByTestId('launch-scenario-btn'))
    await waitFor(() => expect(mockStartRun).toHaveBeenCalled())
    expect(mockStartRun.mock.calls[0][0].include_baseline).toBe(false)
  })

  it('renders scenario-specific parameters and omits common/opaque parameter names', async () => {
    mockGetScenario.mockResolvedValue(
      makeScenario({
        supported_parameters: [
          { name: 'objective_target', type_name: 'any', required: false, default: null, choices: null, is_list: false },
          { name: 'max_concurrency', type_name: 'int', required: false, default: null, choices: null, is_list: false },
          { name: 'technique_converters', type_name: 'any', required: false, default: null, choices: null, is_list: false },
          { name: 'custom_flag', type_name: 'bool', required: false, default: null, choices: null, is_list: false },
          { name: 'iterations', type_name: 'int', required: false, default: '3', choices: null, is_list: false },
        ],
      }),
    )

    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    expect(screen.queryByTestId('scenario-param-objective_target')).not.toBeInTheDocument()
    expect(screen.queryByTestId('scenario-param-max_concurrency')).not.toBeInTheDocument()
    expect(screen.queryByTestId('scenario-param-technique_converters')).not.toBeInTheDocument()
    expect(screen.getByTestId('scenario-param-custom_flag')).toBeInTheDocument()
    expect(screen.getByTestId('scenario-param-iterations')).toHaveValue(3)
  })

  it('reports a validation error for an invalid custom parameter and blocks submission', async () => {
    mockGetScenario.mockResolvedValue(
      makeScenario({
        supported_parameters: [
          { name: 'iterations', type_name: 'int', required: false, default: null, choices: null, is_list: false },
        ],
      }),
    )
    const user = userEvent.setup()

    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    // A number-typed HTML input rejects non-numeric characters outright, so a
    // decimal (a valid *number* but not a valid *integer*) exercises the same
    // coercion/validation path a real user could actually trigger.
    fireEvent.change(screen.getByTestId('scenario-param-iterations'), { target: { value: '1.5' } })
    await user.click(screen.getByTestId('launch-scenario-btn'))

    expect(await screen.findByRole('alert')).toHaveTextContent('iterations must be an integer.')
    expect(mockStartRun).not.toHaveBeenCalled()
  })

  it('omits the dataset override and max dataset size when left blank, sending default concurrency/retries', async () => {
    const user = userEvent.setup()
    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    await user.click(screen.getByRole('button', { name: 'Advanced options' }))
    await user.click(screen.getByTestId('launch-scenario-btn'))

    await waitFor(() => expect(mockStartRun).toHaveBeenCalled())
    const request = mockStartRun.mock.calls[0][0]
    expect(request).not.toHaveProperty('dataset_names')
    expect(request).not.toHaveProperty('max_dataset_size')
    expect(request.max_concurrency).toBe(10)
    expect(request.max_retries).toBe(0)
  })

  it('includes dataset override and max dataset size when provided', async () => {
    const user = userEvent.setup()
    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    await user.click(screen.getByRole('button', { name: 'Advanced options' }))
    await user.type(screen.getByTestId('dataset-override-input'), 'ds_a, ds_b')
    await user.type(screen.getByTestId('max-dataset-size-input'), '25')
    await user.click(screen.getByTestId('launch-scenario-btn'))

    await waitFor(() => expect(mockStartRun).toHaveBeenCalled())
    const request = mockStartRun.mock.calls[0][0]
    expect(request.dataset_names).toEqual(['ds_a', 'ds_b'])
    expect(request.max_dataset_size).toBe(25)
  })

  it('rejects a non-positive-integer max dataset size', async () => {
    const user = userEvent.setup()
    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    await user.click(screen.getByRole('button', { name: 'Advanced options' }))
    await user.type(screen.getByTestId('max-dataset-size-input'), '0')
    await user.click(screen.getByTestId('launch-scenario-btn'))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Max dataset size must be a positive integer.',
    )
    expect(mockStartRun).not.toHaveBeenCalled()
  })

  it('validates advanced concurrency and retry bounds before launching', async () => {
    const user = userEvent.setup()
    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    await user.click(screen.getByRole('button', { name: 'Advanced options' }))
    fireEvent.change(screen.getByTestId('max-concurrency-input'), { target: { value: '500' } })
    fireEvent.blur(screen.getByTestId('max-concurrency-input'))
    await user.click(screen.getByTestId('launch-scenario-btn'))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Max concurrency must be an integer from 1 to 100.',
    )
    expect(mockStartRun).not.toHaveBeenCalled()
  })

  it('sends the exact RunScenarioRequest payload and attaches labels automatically', async () => {
    const user = userEvent.setup()
    mockStartRun.mockResolvedValueOnce({ scenario_result_id: 'sr-1' })

    renderDetail('/scenarios/foundry.red_team_agent', { labels: { operator: 'roakey', operation: 'op1' } })
    await screen.findByTestId('scenario-target-select')

    await user.click(screen.getByTestId('launch-scenario-btn'))

    await waitFor(() => expect(mockStartRun).toHaveBeenCalledTimes(1))
    expect(mockStartRun).toHaveBeenCalledWith({
      scenario_name: 'foundry.red_team_agent',
      target_name: 'target-a',
      techniques: ['default_technique'],
      max_concurrency: 10,
      max_retries: 0,
      include_baseline: true,
      labels: { operator: 'roakey', operation: 'op1' },
    })
  })

  it('sends only prompt_sending for the Jailbreak regression and displays the backend total of 8', async () => {
    const user = userEvent.setup()
    mockGetScenario.mockResolvedValue(
      makeScenario({
        scenario_name: 'airt.jailbreak',
        scenario_type: 'Jailbreak',
        description: 'Runs jailbreak templates.',
        default_technique: 'default',
        aggregate_techniques: ['default'],
        all_techniques: ['prompt_sending', 'jailbreak_system_prompt'],
        default_datasets: ['harmbench'],
        include_baseline_by_default: true,
        supported_parameters: [
          {
            name: 'num_jailbreaks',
            type_name: 'int',
            required: false,
            default: null,
            choices: null,
            is_list: false,
          },
          {
            name: 'num_jailbreak_attempts',
            type_name: 'int',
            required: false,
            default: '1',
            choices: null,
            is_list: false,
          },
        ],
      }),
    )
    mockEstimateRun.mockResolvedValue({
      status: 'available',
      estimate: {
        scope: 'request',
        total: 8,
        additiveComponents: [
          { id: 'objectives', label: 'Objectives', value: 4 },
        ],
        multiplicativeFactors: [
          { id: 'techniques', label: 'Concrete techniques', value: 1 },
          { id: 'jailbreaks', label: 'Jailbreak templates', value: 2 },
          { id: 'attempts', label: 'Attempts', value: 1 },
        ],
        resolvedTechniques: ['prompt_sending'],
        formula: '4 objectives * 1 technique * 2 jailbreaks * 1 attempt',
        caveat: 'The backend total is authoritative.',
      },
    })

    renderDetail('/scenarios/airt.jailbreak')
    await screen.findByTestId('scenario-target-select')

    await user.click(screen.getByTestId('technique-prompt_sending'))
    await user.clear(screen.getByTestId('scenario-param-num_jailbreaks'))
    await user.type(screen.getByTestId('scenario-param-num_jailbreaks'), '2')
    await user.clear(screen.getByTestId('scenario-param-num_jailbreak_attempts'))
    await user.type(screen.getByTestId('scenario-param-num_jailbreak_attempts'), '1')
    await user.click(screen.getByTestId('baseline-checkbox'))

    const expectedRequest = {
      scenario_name: 'airt.jailbreak',
      target_name: 'target-a',
      techniques: ['prompt_sending'],
      max_concurrency: 10,
      max_retries: 0,
      include_baseline: false,
      labels: { operator: 'roakey' },
      scenario_params: {
        num_jailbreaks: 2,
        num_jailbreak_attempts: 1,
      },
    }

    await waitFor(() => expect(mockEstimateRun).toHaveBeenLastCalledWith(expectedRequest))
    const preview = screen.getByRole('complementary', { name: 'Run preview' })
    expect(within(preview).getByText('prompt_sending')).toBeInTheDocument()
    expect(within(preview).getByText('harmbench')).toBeInTheDocument()
    expect(within(preview).getByText('Not included')).toBeInTheDocument()
    expect(within(preview).getByText('8 attacks')).toBeInTheDocument()
    expect(within(preview).getByText('2')).toBeInTheDocument()

    await user.click(screen.getByTestId('launch-scenario-btn'))

    await waitFor(() => expect(mockStartRun).toHaveBeenCalledTimes(1))
    expect(mockStartRun).toHaveBeenCalledWith(expectedRequest)
    expect(mockStartRun.mock.calls[0][0].techniques).not.toContain('default')
  })

  it('navigates to the scenario-history route with the encoded run id on success', async () => {
    const user = userEvent.setup()
    mockStartRun.mockResolvedValueOnce({ scenario_result_id: 'sr/1' })

    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    await user.click(screen.getByTestId('launch-scenario-btn'))

    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith(
        '/scenario-history/sr%2F1',
        expect.objectContaining({ state: expect.objectContaining({ scenarioName: 'foundry.red_team_agent' }) }),
      ),
    )
  })

  it('shows an API error in a MessageBar and re-enables the button on failure', async () => {
    const user = userEvent.setup()
    mockStartRun.mockRejectedValueOnce({
      isAxiosError: true,
      response: { status: 400, data: { detail: 'Invalid target' } },
    })

    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    await user.click(screen.getByTestId('launch-scenario-btn'))

    expect(await screen.findByText('Invalid target')).toBeInTheDocument()
    expect(screen.getByTestId('launch-scenario-btn')).not.toBeDisabled()
    expect(mockNavigate).not.toHaveBeenCalled()
  })

  it('guards against a duplicate submit from a fast double click', async () => {
    let resolveStartRun: (value: { scenario_result_id: string }) => void = () => {}
    mockStartRun.mockReturnValue(
      new Promise((resolve) => {
        resolveStartRun = resolve
      }),
    )

    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    const button = screen.getByTestId('launch-scenario-btn')
    // Fire two rapid clicks without waiting between them (userEvent.click awaits internally,
    // so dispatch native clicks to simulate a true double-click within one tick).
    act(() => {
      button.click()
      button.click()
    })

    await waitFor(() => expect(mockStartRun).toHaveBeenCalledTimes(1))
    resolveStartRun({ scenario_result_id: 'sr-1' })
    await waitFor(() => expect(button).not.toBeDisabled())
  })

  it('preserves entered values and preview content after a failed submission', async () => {
    const user = userEvent.setup()
    mockGetScenario.mockResolvedValue(
      makeScenario({
        supported_parameters: [
          {
            name: 'attempts',
            type_name: 'int',
            required: false,
            default: 1,
            choices: null,
            is_list: false,
          },
        ],
      }),
    )
    mockStartRun.mockRejectedValueOnce({
      isAxiosError: true,
      response: { status: 400, data: { detail: 'boom' } },
    })

    renderDetail('/scenarios/foundry.red_team_agent')
    await screen.findByTestId('scenario-target-select')

    await user.selectOptions(screen.getByTestId('scenario-target-select'), 'target-b')
    await user.click(screen.getByTestId('technique-crescendo'))
    await user.clear(screen.getByTestId('scenario-param-attempts'))
    await user.type(screen.getByTestId('scenario-param-attempts'), '3')
    await user.click(screen.getByTestId('launch-scenario-btn'))

    await screen.findByText('boom')
    expect(screen.getByTestId('scenario-target-select')).toHaveValue('target-b')
    expect(screen.getByTestId('technique-crescendo')).toBeChecked()
    expect(screen.getByTestId('technique-default_technique')).not.toBeChecked()
    expect(screen.getByTestId('scenario-param-attempts')).toHaveValue(3)

    const preview = screen.getByRole('complementary', { name: 'Run preview' })
    expect(within(preview).getByText('target-b')).toBeInTheDocument()
    expect(within(preview).getByText('crescendo')).toBeInTheDocument()
    expect(within(preview).getByText('harmbench')).toBeInTheDocument()
    expect(within(preview).getByText('3')).toBeInTheDocument()
  })
})
