import { useEffect, useRef } from 'react'

export function useTreeDialogFocus(opener: HTMLElement | null): void {
  const lifecycle = useRef({ generation: 0 })
  useEffect(() => {
    const state = lifecycle.current
    const mountedGeneration = ++state.generation
    return () => {
      queueMicrotask(() => {
        // Strict Mode replays effects while the dialog is still open.
        if (state.generation === mountedGeneration && opener?.isConnected) {
          opener.focus({ preventScroll: true })
        }
      })
    }
  }, [opener])
}
