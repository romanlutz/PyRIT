import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { ScenarioQueueSnapshot } from '@/types'

import ScenarioQueue from './ScenarioQueue'

const SNAPSHOT: ScenarioQueueSnapshot = {
  revision: 3,
  snapshot_at: '2026-01-01T00:00:03Z',
  active: {
    scenario_result_id: 'run-active',
    scenario_name: 'ActiveScenario',
    scenario_registry_name: 'active.scenario',
    state: 'IN_PROGRESS',
    created_at: '2026-01-01T00:00:00Z',
    enqueued_at: '2026-01-01T00:00:00Z',
    started_at: '2026-01-01T00:00:01Z',
  },
  queued: [{
    scenario_result_id: 'run-waiting',
    scenario_name: 'WaitingScenario',
    scenario_registry_name: 'waiting.scenario',
    state: 'QUEUED',
    position: 1,
    created_at: '2026-01-01T00:00:02Z',
    enqueued_at: '2026-01-01T00:00:02Z',
  }],
}

function renderQueue(snapshot: ScenarioQueueSnapshot | null, currentScenarioResultId?: string) {
  return render(
    <FluentProvider theme={webLightTheme}>
      <ScenarioQueue
        snapshot={snapshot}
        loading={false}
        stale={false}
        error={null}
        currentScenarioResultId={currentScenarioResultId}
      />
    </FluentProvider>,
  )
}

describe('ScenarioQueue', () => {
  it('renders active and FIFO queued entries as native deep links', () => {
    renderQueue(SNAPSHOT, 'run-waiting')

    const activeLink = screen.getByRole('link', { name: /active\.scenario/i })
    const waitingLink = screen.getByRole('link', { name: /waiting\.scenario/i })
    expect(activeLink).toHaveAttribute('href', '/scenario-history/run-active')
    expect(waitingLink).toHaveAttribute('href', '/scenario-history/run-waiting')
    expect(waitingLink).toHaveAttribute('aria-current', 'page')
    expect(screen.getByText('Active')).toBeInTheDocument()
    expect(screen.getByText('Position 1')).toBeInTheDocument()
  })

  it('renders a concise empty state', () => {
    renderQueue({ revision: 0, snapshot_at: '2026-01-01T00:00:00Z', active: null, queued: [] })

    expect(screen.getByText('No active or queued scenarios.')).toBeInTheDocument()
  })

  it('keeps queue links keyboard reachable', async () => {
    const user = userEvent.setup()
    renderQueue(SNAPSHOT)

    await user.tab()

    expect(screen.getByRole('link', { name: /active\.scenario/i })).toHaveFocus()
  })
})
