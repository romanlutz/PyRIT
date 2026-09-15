import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { ANALYTICS_OPERATION_FILTER, makeAnalyticsReport } from '@/test-utils/analyticsFixtures'

import AnalyticsGroups from './AnalyticsGroups'

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
)

describe('AnalyticsGroups', () => {
  beforeEach(() => { jest.clearAllMocks() })

  it('should drill into an actual outcome segment using its server predicates and outcome', async () => {
    const user = userEvent.setup()
    const onDrilldown = jest.fn()
    render(<TestWrapper><AnalyticsGroups report={makeAnalyticsReport()} successRate={false} allGroups={false} onDrilldown={onDrilldown} /></TestWrapper>)
    const segment = screen.getByRole('button', { name: 'Nightly: success segment; 4 results; filter dashboard' })
    segment.focus()
    await user.keyboard('{Enter}')
    expect(onDrilldown).toHaveBeenCalledWith([ANALYTICS_OPERATION_FILTER], 'success')
    await user.click(screen.getByRole('button', { name: 'Nightly: 4 success; filter dashboard' }))
    expect(onDrilldown).toHaveBeenCalledTimes(2)
  })

  it('should make the success-rate bar a keyboard-operable group drill-down', async () => {
    const user = userEvent.setup()
    const onDrilldown = jest.fn()
    render(<TestWrapper><AnalyticsGroups report={makeAnalyticsReport()} successRate allGroups={false} onDrilldown={onDrilldown} /></TestWrapper>)
    const bar = screen.getByRole('button', { name: /Inspect Nightly success rate: 10 results; ASR 66.7%/ })
    bar.focus()
    await user.keyboard(' ')
    expect(onDrilldown).toHaveBeenCalledWith([ANALYTICS_OPERATION_FILTER])
  })
})
