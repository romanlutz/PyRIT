import type { ComponentProps, ReactNode } from 'react'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import ChatInputArea from './ChatInputArea'

function TestWrapper({ children }: { children: ReactNode }) {
  return <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
}

const defaultProps: ComponentProps<typeof ChatInputArea> = {
  onSend: jest.fn(),
  onNewConversation: jest.fn(),
  onUseAsTemplate: jest.fn(),
  onConfigureTarget: jest.fn(),
  onToggleConverterPanel: jest.fn(),
  isConverterPanelOpen: false,
  onInputChange: jest.fn(),
  onAttachmentsChange: jest.fn(),
  onClearConversion: jest.fn(),
  onConvertedValueChange: jest.fn(),
  onClearMediaConversion: jest.fn(),
}

describe('ChatInputArea multi-send', () => {
  beforeEach(() => jest.clearAllMocks())

  it.each(['click', 'enter'])('submits the same multi-send options with %s and resets n', async (method: string) => {
    const user = userEvent.setup()
    render(<TestWrapper><ChatInputArea {...defaultProps} /></TestWrapper>)

    await user.click(screen.getByRole('button', { name: 'Repetitions: 1' }))
    await user.click(screen.getByRole('button', { name: 'Increase repetitions' }))
    await user.click(screen.getByRole('radio', { name: 'Convert independently for each' }))
    await user.keyboard('{Escape}')
    await user.type(screen.getByRole('textbox'), 'Repeat this prompt')
    if (method === 'click') {
      await user.click(screen.getByRole('button', { name: 'Send in 2 conversations' }))
    } else {
      await user.keyboard('{Enter}')
    }

    expect(defaultProps.onSend).toHaveBeenCalledWith(
      'Repeat this prompt', undefined, [], { count: 2, requestConverterMode: 'per_branch' },
    )
    expect(screen.getByRole('button', { name: 'Repetitions: 1' })).toBeInTheDocument()
    expect(screen.getByRole('textbox')).toHaveValue('')
    await user.click(screen.getByRole('button', { name: 'Repetitions: 1' }))
    expect(screen.getByRole('radio', { name: 'Convert independently for each' })).toBeChecked()
  })

  it('preserves the default single-send callback', async () => {
    const user = userEvent.setup()
    render(<TestWrapper><ChatInputArea {...defaultProps} /></TestWrapper>)
    await user.type(screen.getByRole('textbox'), 'One request')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    expect(defaultProps.onSend).toHaveBeenCalledWith('One request', undefined, [])
  })

  it('does not submit an empty draft or reset its selected count', async () => {
    const user = userEvent.setup()
    render(<TestWrapper><ChatInputArea {...defaultProps} /></TestWrapper>)
    await user.click(screen.getByRole('button', { name: 'Repetitions: 1' }))
    await user.click(screen.getByRole('button', { name: 'Increase repetitions' }))
    await user.keyboard('{Escape}')
    expect(screen.getByRole('button', { name: 'Send in 2 conversations' })).toBeDisabled()
    await user.click(screen.getByRole('textbox'))
    await user.keyboard('{Enter}')
    expect(defaultProps.onSend).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Repetitions: 2' })).toBeInTheDocument()
  })
})
