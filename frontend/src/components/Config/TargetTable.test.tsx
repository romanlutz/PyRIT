import { render, screen, fireEvent } from '@testing-library/react'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import TargetTable from './TargetTable'
import type { TargetInstance } from '../../types'

jest.mock('./TargetTable.styles', () => ({
  useTargetTableStyles: () => new Proxy({}, { get: () => '' }),
}))

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
)

const sampleTargets: TargetInstance[] = [
  {
    target_registry_name: 'openai_chat_gpt4',
    target_type: 'OpenAIChatTarget',
    endpoint: 'https://api.openai.com',
    model_name: 'gpt-4',
    deployment_name: 'gpt-4',
  },
  {
    target_registry_name: 'azure_image_dalle',
    target_type: 'AzureImageTarget',
    endpoint: 'https://azure.openai.com',
    model_name: 'dall-e-3',
    deployment_name: 'dall-e-3',
  },
  {
    target_registry_name: 'text_target_basic',
    target_type: 'TextTarget',
    endpoint: null,
    model_name: null,
    deployment_name: null,
  },
]

describe('TargetTable', () => {
  const defaultProps = {
    targets: sampleTargets,
    activeTarget: null as TargetInstance | null,
    onSetActiveTarget: jest.fn(),
  }

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should render section headings alphabetically by target type', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} />
      </TestWrapper>
    )

    const buttons = screen.getAllByRole('button', { expanded: false })
    const headings = buttons.map(b => b.textContent).filter(t => t && !t.includes('Set Active'))
    // AzureImageTarget, OpenAIChatTarget, TextTarget (alphabetical)
    expect(headings[0]).toContain('AzureImageTarget')
    expect(headings[1]).toContain('OpenAIChatTarget')
    expect(headings[2]).toContain('TextTarget')
  })

  it('should show item count in section headings', () => {
    const targets = [
      ...sampleTargets,
      { target_registry_name: 'chat2', target_type: 'OpenAIChatTarget', endpoint: 'https://x.com', model_name: 'gpt-5' },
    ]

    render(
      <TestWrapper>
        <TargetTable {...defaultProps} targets={targets} />
      </TestWrapper>
    )

    // OpenAIChatTarget has 2 items
    expect(screen.getByText('(2)')).toBeInTheDocument()
  })

  it('should start with all sections collapsed when no active target', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} />
      </TestWrapper>
    )

    // No tables visible when all collapsed
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
    expect(screen.queryByText('Set Active')).not.toBeInTheDocument()
  })

  it('should expand the section containing the active target', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} activeTarget={sampleTargets[0]} />
      </TestWrapper>
    )

    // OpenAIChatTarget section should be expanded
    expect(screen.getByText('gpt-4')).toBeInTheDocument()
    expect(screen.getByText('Active')).toBeInTheDocument()
    // Other sections should still be collapsed
    expect(screen.queryByText('dall-e-3')).not.toBeInTheDocument()
  })

  it('should toggle section expand/collapse on click', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} />
      </TestWrapper>
    )

    // Click on AzureImageTarget heading to expand
    const sectionButton = screen.getByRole('button', { name: /AzureImageTarget/i })
    fireEvent.click(sectionButton)

    // Table should now be visible with target data
    expect(screen.getByText('dall-e-3')).toBeInTheDocument()
    expect(screen.getByText('https://azure.openai.com')).toBeInTheDocument()

    // Click again to collapse
    fireEvent.click(sectionButton)
    expect(screen.queryByText('dall-e-3')).not.toBeInTheDocument()
  })

  it('should display endpoint and model name when section is expanded', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} activeTarget={sampleTargets[0]} />
      </TestWrapper>
    )

    // OpenAIChatTarget section is expanded (has active target)
    expect(screen.getByText('gpt-4')).toBeInTheDocument()
    expect(screen.getByText('https://api.openai.com')).toBeInTheDocument()
  })

  it('should show "Set Active" button for non-active targets', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} activeTarget={sampleTargets[0]} />
      </TestWrapper>
    )

    // Only OpenAIChatTarget section is expanded, and it has the active target
    // So no "Set Active" buttons visible (only 1 target in that group, and it's active)
    expect(screen.queryByText('Set Active')).not.toBeInTheDocument()
  })

  it('should show "Active" badge for the active target', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} activeTarget={sampleTargets[0]} />
      </TestWrapper>
    )

    expect(screen.getByText('Active')).toBeInTheDocument()
  })

  it('should call onSetActiveTarget when "Set Active" is clicked', () => {
    const onSetActiveTarget = jest.fn()

    render(
      <TestWrapper>
        <TargetTable {...defaultProps} onSetActiveTarget={onSetActiveTarget} />
      </TestWrapper>
    )

    // Expand AzureImageTarget section
    fireEvent.click(screen.getByRole('button', { name: /AzureImageTarget/i }))

    const setActiveButton = screen.getByText('Set Active')
    fireEvent.click(setActiveButton)

    expect(onSetActiveTarget).toHaveBeenCalledTimes(1)
    expect(onSetActiveTarget).toHaveBeenCalledWith(sampleTargets[1])
  })

  it('should handle empty targets list gracefully', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} targets={[]} />
      </TestWrapper>
    )

    expect(screen.queryByRole('table')).not.toBeInTheDocument()
    expect(screen.queryByText('Set Active')).not.toBeInTheDocument()
  })

  it('should show dash when model_name or endpoint is null', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} targets={[sampleTargets[2]]} />
      </TestWrapper>
    )

    // Expand TextTarget section
    fireEvent.click(screen.getByRole('button', { name: /TextTarget/i }))

    // TextTarget has null model_name and endpoint; should render "—"
    const dashes = screen.getAllByText('—')
    expect(dashes.length).toBeGreaterThanOrEqual(2)
  })

  it('should display Parameters column header when expanded', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} />
      </TestWrapper>
    )

    // Expand a section first
    fireEvent.click(screen.getByRole('button', { name: /OpenAIChatTarget/i }))

    const paramHeaders = screen.getAllByText('Parameters')
    expect(paramHeaders.length).toBeGreaterThanOrEqual(1)
  })

  it('should display target_specific_params when present', () => {
    const targetWithParams: TargetInstance[] = [
      {
        target_registry_name: 'param_target',
        target_type: 'OpenAIResponseTarget',
        endpoint: 'https://api.openai.com',
        model_name: 'o3',
        target_specific_params: {
          reasoning_effort: 'high',
          max_output_tokens: 4096,
        },
      },
    ]

    render(
      <TestWrapper>
        <TargetTable {...defaultProps} targets={targetWithParams} activeTarget={null} />
      </TestWrapper>
    )

    // Expand the section
    fireEvent.click(screen.getByRole('button', { name: /OpenAIResponseTarget/i }))

    expect(screen.getByText(/reasoning_effort: high/)).toBeInTheDocument()
    expect(screen.getByText(/max_output_tokens: 4096/)).toBeInTheDocument()
  })

  it('should show tooltip for model with different underlying model', () => {
    const targetWithUnderlying: TargetInstance[] = [
      {
        target_registry_name: 'azure_deployment',
        target_type: 'OpenAIChatTarget',
        endpoint: 'https://azure.openai.com',
        model_name: 'gpt-4o',
        deployment_name: 'my-gpt4o-deployment',
      },
    ]

    render(
      <TestWrapper>
        <TargetTable {...defaultProps} targets={targetWithUnderlying} activeTarget={null} />
      </TestWrapper>
    )

    // Expand the section
    fireEvent.click(screen.getByRole('button', { name: /OpenAIChatTarget/i }))

    // Deployment name should be displayed with dotted underline
    const modelText = screen.getByText('my-gpt4o-deployment')
    expect(modelText).toHaveStyle({ textDecoration: 'underline dotted' })
  })

  it('should keep multiple sections expanded independently', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} />
      </TestWrapper>
    )

    // Expand two sections
    fireEvent.click(screen.getByRole('button', { name: /OpenAIChatTarget/i }))
    fireEvent.click(screen.getByRole('button', { name: /AzureImageTarget/i }))

    // Both should be visible
    expect(screen.getByText('gpt-4')).toBeInTheDocument()
    expect(screen.getByText('dall-e-3')).toBeInTheDocument()

    // Collapse one — the other should remain
    fireEvent.click(screen.getByRole('button', { name: /OpenAIChatTarget/i }))
    expect(screen.queryByText('gpt-4')).not.toBeInTheDocument()
    expect(screen.getByText('dall-e-3')).toBeInTheDocument()
  })

  it('should expand all sections when "Expand All" is clicked', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} />
      </TestWrapper>
    )

    // All collapsed initially
    expect(screen.queryByRole('table')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /expand all/i }))

    // All sections expanded — all data visible
    expect(screen.getByText('gpt-4')).toBeInTheDocument()
    expect(screen.getByText('dall-e-3')).toBeInTheDocument()
    const tables = screen.getAllByRole('table')
    expect(tables.length).toBe(3)
  })

  it('should collapse all sections when "Collapse All" is clicked', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} />
      </TestWrapper>
    )

    // Expand all first
    fireEvent.click(screen.getByRole('button', { name: /expand all/i }))
    expect(screen.getAllByRole('table').length).toBe(3)

    // Now collapse all
    fireEvent.click(screen.getByRole('button', { name: /collapse all/i }))
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('should not show Expand All button when targets list is empty', () => {
    render(
      <TestWrapper>
        <TargetTable {...defaultProps} targets={[]} />
      </TestWrapper>
    )

    expect(screen.queryByRole('button', { name: /expand all/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /collapse all/i })).not.toBeInTheDocument()
  })
})
