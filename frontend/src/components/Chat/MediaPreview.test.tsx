import { StrictMode } from 'react'
import { act, render, screen } from '@testing-library/react'

import { MediaWithFallback } from './MediaPreview'

describe('MediaWithFallback lifecycle', () => {
  afterEach(() => jest.restoreAllMocks())

  it.each(['audio', 'video'] as const)('preserves %s sources during Strict Mode replay but releases them on unmount', async (type: 'audio' | 'video') => {
    const pause = jest.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    const load = jest.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
    const result = render(
      <StrictMode>
        <MediaWithFallback type={type} src="/test-clip" preload="metadata" stopOnUnmount />
      </StrictMode>,
    )
    const player = screen.getByTestId(`${type}-player`)
    await act(async () => { await Promise.resolve() })
    expect(player).toHaveAttribute('src', '/test-clip')
    expect(load).not.toHaveBeenCalled()
    result.unmount()
    await act(async () => { await Promise.resolve() })
    expect(pause).toHaveBeenCalled()
    expect(load).toHaveBeenCalledTimes(1)
    expect(player).not.toHaveAttribute('src')
  })
})
