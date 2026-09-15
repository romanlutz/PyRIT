import { StrictMode } from 'react'
import { act, render, screen } from '@testing-library/react'

import { useTreeDialogFocus } from './useTreeDialogFocus'

function Preview({ opener }: { opener: HTMLElement }) {
  useTreeDialogFocus(opener)
  return <button autoFocus>Close preview</button>
}

describe('useTreeDialogFocus', () => {
  it('keeps focus in a Strict Mode dialog until it actually unmounts', async () => {
    render(<button>Open preview</button>)
    const opener = screen.getByRole('button', { name: 'Open preview' })
    opener.focus()
    const preview = render(<StrictMode><Preview opener={opener} /></StrictMode>)
    await act(async () => { await Promise.resolve() })
    expect(screen.getByRole('button', { name: 'Close preview' })).toHaveFocus()
    preview.unmount()
    await act(async () => { await Promise.resolve() })
    expect(opener).toHaveFocus()
  })

  it('does not focus an opener that has been removed', async () => {
    const background = render(<button>Open preview</button>)
    const opener = screen.getByRole('button', { name: 'Open preview' })
    const focus = jest.spyOn(opener, 'focus')
    const preview = render(<Preview opener={opener} />)
    background.unmount()
    preview.unmount()
    await act(async () => { await Promise.resolve() })
    expect(focus).not.toHaveBeenCalled()
    focus.mockRestore()
  })
})
