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
  const description = overrides.description ?? 'A demo scenario.'
  const defaultTechnique = overrides.default_technique ?? 'default_technique'
  return {
    scenario_type: 'DemoScenario',
    scenario_version: 1,
    aggregate_techniques: [],
    aggregate_technique_expansions: {},
    all_techniques: ['default_technique'],
    default_datasets: [],
    default_dataset_summaries: [],
    baseline_policy: 'enabled',
    include_baseline_by_default: true,
    supported_parameters: [],
    default_run_size: {
      version: 1,
      status: 'unavailable',
      total_attack_count: null,
      minimum_attack_count: null,
      maximum_attack_count: null,
      condition: null,
      components: [],
      datasets: [],
      note: 'Default sizing is not available.',
      retries_included: false,
    },
    ...overrides,
    description,
    description_markdown: overrides.description_markdown ?? description,
    default_technique: defaultTechnique,
    default_techniques: overrides.default_techniques ?? [defaultTechnique],
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

  it('explains scenario run plans and applies spacing classes to every table cell', async () => {
    mockListCatalog.mockResolvedValueOnce({
      items: [makeScenario({ scenario_name: 'foundry.red_team_agent' })],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    const table = await screen.findByRole('table', { name: 'Registered scenarios' })
    expect(screen.getByText(/packages objective datasets, selected or aggregate techniques/i)).toBeInTheDocument()
    const headers = within(table).getAllByRole('columnheader')
    expect(headers).toHaveLength(5)
    expect(headers.map((header) => header.textContent)).toEqual([
      'Scenario / purpose',
      'Default dataset size',
      'Default techniques',
      'Default run size',
      'Action',
    ])
    expect(headers.every((cell) => cell.classList.contains('scenario-catalog-cell-padding'))).toBe(true)
    const cells = within(screen.getByTestId('scenario-card-foundry.red_team_agent')).getAllByRole('cell')
    expect(cells).toHaveLength(5)
    expect(cells.every((cell) => cell.classList.contains('scenario-catalog-cell-padding'))).toBe(true)
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

  it('shows multiple default populations separately instead of summing them', async () => {
    mockListCatalog.mockResolvedValueOnce({
      items: [
        makeScenario({
          scenario_name: 'scenario.compound',
          default_datasets: ['population-a', 'population-b'],
          default_dataset_summaries: [
            {
              name: 'population-a',
              kind: 'dataset',
              logical_seed_group_count: 100,
              selected_seed_group_count: 4,
              configured_caps: [],
              selection_note: null,
            },
            {
              name: 'population-b',
              kind: 'synthesized',
              logical_seed_group_count: 20,
              selected_seed_group_count: 2,
              configured_caps: [],
              selection_note: null,
            },
          ],
        }),
      ],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    const row = await screen.findByTestId('scenario-card-scenario.compound')
    expect(within(row).getByText('population-a: 4 · population-b: 2')).toBeInTheDocument()
    expect(within(row).queryByText('6 selected seed groups')).not.toBeInTheDocument()
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

  it('keeps declared datasets visible when backend population summaries are unavailable', async () => {
    const user = userEvent.setup()
    mockListCatalog.mockResolvedValueOnce({
      items: [
        makeScenario({
          scenario_name: 'scenario.unsized',
          default_datasets: ['harmbench'],
          default_dataset_summaries: [],
        }),
      ],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)
    const row = await screen.findByTestId('scenario-card-scenario.unsized')
    expect(within(row).getByText('Population counts unavailable')).toBeInTheDocument()
    await user.click(within(row).getByRole('button', { name: 'Show details' }))

    const details = screen.getByRole('region', { name: 'scenario.unsized details' })
    expect(within(details).getByText('harmbench')).toBeInTheDocument()
    expect(within(details).getByText('Population unavailable')).toBeInTheDocument()
    expect(within(details).getByText(
      'Population counts and configured caps aren’t available.',
    )).toBeInTheDocument()
    expect(within(details).queryByText(
      'This scenario does not declare a default dataset.',
    )).not.toBeInTheDocument()
  })

  it('discloses every technique and dataset without an unactionable +N summary', async () => {
    const user = userEvent.setup()
    mockListCatalog.mockResolvedValueOnce({
      items: [
        makeScenario({
          scenario_name: 'airt.jailbreak',
          scenario_version: 4,
          default_technique: 'default',
          default_techniques: ['prompt_sending', 'jailbreak_system_prompt'],
          aggregate_techniques: ['default', 'easy'],
          aggregate_technique_expansions: {
            default: ['prompt_sending', 'jailbreak_system_prompt'],
            easy: ['prompt_sending'],
          },
          all_techniques: ['prompt_sending', 'jailbreak_system_prompt', 'flip'],
          default_datasets: ['harmbench'],
          default_dataset_summaries: [
            {
              name: 'harmbench',
              kind: 'dataset',
              logical_seed_group_count: 400,
              selected_seed_group_count: 4,
              configured_caps: [
                {
                  label: 'Jailbreak templates',
                  count: 2,
                  configured_on: 'configuration',
                  dataset_name: null,
                },
              ],
              selection_note: 'One incompatible group is excluded.',
            },
          ],
          default_run_size: {
            version: 1,
            status: 'conditional',
            total_attack_count: null,
            minimum_attack_count: 12,
            maximum_attack_count: 20,
            condition: 'target_capabilities',
            components: [
              {
                label: 'Default attacks',
                count: 8,
                factors: [
                  { label: 'selected seed groups', count: 4 },
                  { label: 'default techniques', count: 2 },
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
                configured_caps: [
                  {
                    label: 'Jailbreak templates',
                    count: 2,
                    configured_on: 'configuration',
                    dataset_name: null,
                  },
                ],
                selection_note: 'One incompatible group is excluded.',
              },
            ],
            note: 'Retries and internal turns are excluded.',
            retries_included: false,
          },
        }),
      ],
      pagination: { limit: 200, has_more: false },
    })

    render(<TestWrapper><ScenarioCatalog /></TestWrapper>)

    const row = await screen.findByTestId('scenario-card-airt.jailbreak')
    const disclosure = within(row).getByRole('button', { name: 'Show details' })
    expect(disclosure).toHaveAttribute('aria-expanded', 'false')
    expect(within(row).getByText('4 selected seed groups')).toBeInTheDocument()
    expect(within(row).getByText('harmbench · 400 available')).toBeInTheDocument()
    expect(within(row).getByText('2 techniques')).toBeInTheDocument()
    expect(within(row).getByText('12–20 planned attacks')).toBeInTheDocument()
    expect(within(row).queryByText('default')).not.toBeInTheDocument()
    expect(within(row).queryByText(/aggregate presets|compatible concrete/i)).not.toBeInTheDocument()
    expect(within(row).queryByText(/Run size calculated|Final count set at launch/i)).not.toBeInTheDocument()

    await user.click(disclosure)

    expect(disclosure).toHaveAttribute('aria-expanded', 'true')
    const details = screen.getByRole('region', { name: 'airt.jailbreak details' })
    expect(within(details).getAllByText('default').length).toBeGreaterThan(0)
    expect(within(details).getByText('easy')).toBeInTheDocument()
    expect(within(details).getAllByText('prompt_sending').length).toBeGreaterThan(0)
    expect(within(details).getAllByText('jailbreak_system_prompt').length).toBeGreaterThan(0)
    expect(within(details).getByText('flip')).toBeInTheDocument()
    expect(within(details).getAllByText('harmbench').length).toBeGreaterThan(0)
    expect(within(details).getAllByText('Jailbreak templates: 2 (configuration)').length)
      .toBeGreaterThan(0)
    expect(within(details).getAllByText('One incompatible group is excluded.').length)
      .toBeGreaterThan(0)
    expect(within(details).getByText('12–20 planned attacks')).toBeInTheDocument()
    expect(within(details).getAllByText('Included by default')).not.toHaveLength(0)
    expect(within(details).queryByText(/Backend estimate|Conditional estimate|Backend formula/i))
      .not.toBeInTheDocument()
    expect(within(details).queryByText('context_compliance')).not.toBeInTheDocument()
    expect(within(details).queryByText(/multi.?turn/i)).not.toBeInTheDocument()
    expect(within(details).queryByText(/simulated/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/\+\d+ more/i)).not.toBeInTheDocument()
  })

  it('normalizes MyST literals while keeping scenario Markdown and raw HTML safe', async () => {
    const user = userEvent.setup()
    mockListCatalog.mockResolvedValueOnce({
      items: [
        makeScenario({
          scenario_name: 'airt.jailbreak',
          description: 'Configure the Jailbreak scenario.',
          description_markdown: `Set \`\`num_jailbreaks\`\`.\n\n${RAW_IMAGE_HTML}unsafe`,
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
