import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { MemoryRouter } from 'react-router-dom'

import { scenariosApi } from '@/services/api'
import type { RegisteredScenario } from '@/types'

import ScenarioCatalog from './ScenarioCatalog'

jest.mock('@/services/api', () => ({
  scenariosApi: {
    listCatalog: jest.fn(),
  },
}))

const mockListCatalog = scenariosApi.listCatalog as jest.Mock

const RAW_IMAGE_HTML = ['<', 'img src=x onerror="alert(1)">'].join('')

function TestWrapper({ children }: { children: React.ReactNode }) {
  return (
    <FluentProvider theme={webLightTheme}>
      <MemoryRouter>{children}</MemoryRouter>
    </FluentProvider>
  )
}

function makeScenario(overrides: Partial<RegisteredScenario> & { scenario_name: string }): RegisteredScenario {
  return {
    scenario_type: 'DemoScenario',
    description: 'A demo scenario.',
    default_technique: 'default_technique',
    aggregate_techniques: [],
    all_techniques: ['default_technique'],
    default_datasets: [],
    baseline_policy: 'enabled',
    include_baseline_by_default: true,
    supported_parameters: [],
    ...overrides,
  }
}

describe('ScenarioCatalog', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('shows a loading state while fetching', () => {
    mockListCatalog.mockReturnValue(new Promise(() => {}))
    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)
    expect(screen.getByText('Loading scenarios...')).toBeInTheDocument()
  })

  it('renders every scenario from a single page', async () => {
    mockListCatalog.mockResolvedValueOnce({
      items: [
        makeScenario({ scenario_name: 'foundry.red_team_agent', description: 'Red teams a target.' }),
        makeScenario({ scenario_name: 'encoding.base64', description: 'Encodes prompts.' }),
      ],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    expect(await screen.findByText('foundry.red_team_agent')).toBeInTheDocument()
    expect(screen.getByText('encoding.base64')).toBeInTheDocument()
    expect(mockListCatalog).toHaveBeenCalledTimes(1)
  })

  it('explains how scenarios become backend run plans and renders a semantic comparison table', async () => {
    mockListCatalog.mockResolvedValueOnce({
      items: [makeScenario({ scenario_name: 'foundry.red_team_agent' })],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    expect(await screen.findByRole('table', { name: 'Registered scenarios' })).toBeInTheDocument()
    expect(screen.getByText(/packages objective datasets, selected or aggregate techniques/i)).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'Scenario / purpose' })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'Default run size' })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'Techniques' })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'Datasets' })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'Action' })).toBeInTheDocument()
  })

  it('follows the cursor to load every page automatically', async () => {
    mockListCatalog
      .mockResolvedValueOnce({
        items: [makeScenario({ scenario_name: 'scenario.page1' })],
        pagination: { limit: 1, has_more: true, next_cursor: 'cursor-1' },
      })
      .mockResolvedValueOnce({
        items: [makeScenario({ scenario_name: 'scenario.page2' })],
        pagination: { limit: 1, has_more: false },
      })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    expect(await screen.findByText('scenario.page1')).toBeInTheDocument()
    expect(screen.getByText('scenario.page2')).toBeInTheDocument()
    expect(mockListCatalog).toHaveBeenCalledTimes(2)
    expect(mockListCatalog).toHaveBeenNthCalledWith(2, 200, 'cursor-1')
  })

  it('stops paging if the backend repeats a cursor instead of looping forever', async () => {
    mockListCatalog.mockResolvedValue({
      items: [makeScenario({ scenario_name: 'scenario.loop' })],
      pagination: { limit: 1, has_more: true, next_cursor: 'same-cursor' },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    expect(await screen.findAllByText('scenario.loop')).toHaveLength(1)
    await waitFor(() => expect(mockListCatalog).toHaveBeenCalledTimes(2))
    // Give any additional (incorrect) fetch a chance to fire before asserting it didn't.
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(mockListCatalog).toHaveBeenCalledTimes(2)
  })

  it('shows an empty state when no scenarios are registered', async () => {
    mockListCatalog.mockResolvedValueOnce({ items: [], pagination: { limit: 200, has_more: false } })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    expect(await screen.findByTestId('empty-state')).toBeInTheDocument()
  })

  it('shows an error MessageBar with a retry action on failure', async () => {
    mockListCatalog.mockRejectedValueOnce(new Error('Network error — check that the backend is running and reachable.'))

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    expect(await screen.findByTestId('error-state')).toBeInTheDocument()
    expect(screen.getByText(/Network error/)).toBeInTheDocument()
    expect(screen.getByTestId('retry-btn')).toBeInTheDocument()
  })

  it('retries the fetch when Retry is clicked', async () => {
    const user = userEvent.setup()
    mockListCatalog
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce({
        items: [makeScenario({ scenario_name: 'scenario.recovered' })],
        pagination: { limit: 200, has_more: false },
      })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    await screen.findByTestId('error-state')
    await user.click(screen.getByTestId('retry-btn'))

    expect(await screen.findByText('scenario.recovered')).toBeInTheDocument()
    expect(mockListCatalog).toHaveBeenCalledTimes(2)
  })

  it('filters scenarios by the search box across name, description, techniques, and datasets', async () => {
    const user = userEvent.setup()
    mockListCatalog.mockResolvedValueOnce({
      items: [
        makeScenario({ scenario_name: 'foundry.red_team_agent', description: 'Red teams a target.' }),
        makeScenario({
          scenario_name: 'encoding.base64',
          description: 'Applies text encodings.',
          default_datasets: ['harmbench'],
        }),
      ],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    await screen.findByText('foundry.red_team_agent')

    await user.type(screen.getByLabelText('Search scenarios'), 'harmbench')

    expect(screen.queryByText('foundry.red_team_agent')).not.toBeInTheDocument()
    expect(screen.getByText('encoding.base64')).toBeInTheDocument()
  })

  it('shows a no-results state when the search matches nothing', async () => {
    const user = userEvent.setup()
    mockListCatalog.mockResolvedValueOnce({
      items: [makeScenario({ scenario_name: 'foundry.red_team_agent' })],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)
    await screen.findByText('foundry.red_team_agent')

    await user.type(screen.getByLabelText('Search scenarios'), 'no-such-scenario')

    expect(await screen.findByTestId('no-results-state')).toBeInTheDocument()
  })

  it('links each card to its encoded scenario detail route', async () => {
    mockListCatalog.mockResolvedValueOnce({
      items: [makeScenario({ scenario_name: 'foundry/red_team_agent' })],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    const card = await screen.findByRole('link', { name: /foundry\/red_team_agent/i })
    expect(card).toHaveAttribute('href', '/scenarios/foundry%2Fred_team_agent')
  })

  it('keeps Markdown links out of the clipped summary until the disclosure is opened', async () => {
    const user = userEvent.setup()
    mockListCatalog.mockResolvedValueOnce({
      items: [
        makeScenario({
          scenario_name: 'garak.doctor',
          description: 'Read the [scenario guide](https://example.com/guide) before launch.',
        }),
      ],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)
    const row = await screen.findByTestId('scenario-card-garak.doctor')
    expect(screen.queryByRole('link', { name: 'scenario guide' })).not.toBeInTheDocument()

    await user.click(within(row).getByRole('button', { name: 'Show details' }))

    const guide = screen.getByRole('link', { name: 'scenario guide' })
    expect(guide).toHaveAttribute('target', '_blank')
    expect(guide).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('discloses every technique and dataset without an unactionable +N summary', async () => {
    const user = userEvent.setup()
    mockListCatalog.mockResolvedValueOnce({
      items: [
        makeScenario({
          scenario_name: 'foundry.red_team_agent',
          default_technique: 'default',
          aggregate_techniques: ['default', 'easy', 'moderate'],
          all_techniques: ['default', 'easy', 'moderate', 'crescendo', 'jailbreak', 'prompt_sending'],
          default_datasets: ['harmbench', 'advbench'],
        }),
      ],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    const row = await screen.findByTestId('scenario-card-foundry.red_team_agent')
    const disclosure = within(row).getByRole('button', { name: 'Show details' })
    expect(disclosure).toHaveAttribute('aria-expanded', 'false')

    await user.click(disclosure)

    expect(disclosure).toHaveAttribute('aria-expanded', 'true')
    const details = screen.getByRole('region', { name: 'foundry.red_team_agent details' })
    expect(within(details).getByText('default')).toBeInTheDocument()
    expect(within(details).getByText('easy')).toBeInTheDocument()
    expect(within(details).getByText('moderate')).toBeInTheDocument()
    expect(within(details).getByText('crescendo')).toBeInTheDocument()
    expect(within(details).getByText('jailbreak')).toBeInTheDocument()
    expect(within(details).getByText('prompt_sending')).toBeInTheDocument()
    expect(within(details).getByText('harmbench')).toBeInTheDocument()
    expect(within(details).getByText('advbench')).toBeInTheDocument()
    expect(within(details).getAllByText('Count unavailable')).toHaveLength(2)
    expect(screen.queryByText(/\+\d+ more/i)).not.toBeInTheDocument()
  })

  it('normalizes MyST literals while keeping scenario Markdown and raw HTML safe', async () => {
    const user = userEvent.setup()
    mockListCatalog.mockResolvedValueOnce({
      items: [
        makeScenario({
          scenario_name: 'airt.jailbreak',
          description: `Set \`\`num_jailbreaks\`\`.\n\n${RAW_IMAGE_HTML}unsafe`,
        }),
      ],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)
    const row = await screen.findByTestId('scenario-card-airt.jailbreak')
    await user.click(within(row).getByRole('button', { name: 'Show details' }))

    const details = screen.getByRole('region', { name: 'airt.jailbreak details' })
    const literal = within(details).getByText('num_jailbreaks')
    expect(literal.tagName).toBe('CODE')
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(
      within(details).getByText((content: string) => content.includes(`${RAW_IMAGE_HTML}unsafe`)),
    ).toBeInTheDocument()
  })
})
