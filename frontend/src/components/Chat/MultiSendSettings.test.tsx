import { useState } from 'react'
import type { ReactNode } from 'react'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { MultiSendOptions } from '@/types'

import MultiSendSettings from './MultiSendSettings'

function TestWrapper({ children }: { children: ReactNode }) {
  return <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
}

function Settings({ disabled = false }: { disabled?: boolean }) {
  const [options, setOptions] = useState<MultiSendOptions>({ count: 1, requestConverterMode: 'shared' })
  return <MultiSendSettings options={options} onChange={setOptions} disabled={disabled} />
}

describe('MultiSendSettings', () => {
  it('bounds the count between one and ten', async () => {
    const user = userEvent.setup()
    render(<TestWrapper><Settings /></TestWrapper>)
    await user.click(screen.getByRole('button', { name: 'Repetitions: 1' }))

    expect(screen.getByRole('button', { name: 'Decrease repetitions' })).toBeDisabled()
    for (let index = 1; index < 10; index++) {
      await user.click(screen.getByRole('button', { name: 'Increase repetitions' }))
    }
    expect(screen.getByRole('button', { name: 'Repetitions: 10' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Increase repetitions' })).toBeDisabled()
    expect(screen.getByText(/create 9 copies of its history/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Decrease repetitions' }))
    expect(screen.getByRole('button', { name: 'Repetitions: 9' })).toBeInTheDocument()
  })

  it('defaults to shared conversion and supports independent conversion', async () => {
    const user = userEvent.setup()
    render(<TestWrapper><Settings /></TestWrapper>)
    await user.click(screen.getByRole('button', { name: 'Repetitions: 1' }))
    expect(screen.getByRole('radio', { name: 'Convert once, reuse for all' })).toBeChecked()
    await user.click(screen.getByRole('radio', { name: 'Convert independently for each' }))
    expect(screen.getByRole('radio', { name: 'Convert independently for each' })).toBeChecked()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('radio')).not.toBeInTheDocument()
  })

  it('cannot change settings when sending is disabled', () => {
    render(<TestWrapper><Settings disabled /></TestWrapper>)
    expect(screen.getByRole('button', { name: 'Repetitions: 1' })).toBeDisabled()
  })
})
