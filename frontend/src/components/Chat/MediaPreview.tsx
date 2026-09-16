import { useCallback, useLayoutEffect, useRef, useState } from 'react'

import { Spinner, Text } from '@fluentui/react-components'

interface ImageWithSpinnerProps {
  src: string
  alt: string
  className: string
  hiddenClassName: string
  containerClassName: string
  spinnerClassName: string
}

export function ImageWithSpinner({
  src,
  alt,
  className,
  hiddenClassName,
  containerClassName,
  spinnerClassName,
}: ImageWithSpinnerProps) {
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState(false)
  const onLoad = useCallback(() => { setLoaded(true) }, [])
  const onError = useCallback(() => { setError(true); setLoaded(true) }, [])

  return (
    <div className={containerClassName}>
      {!loaded && <Spinner size="small" className={spinnerClassName} />}
      {error
        ? <Text size={200} italic>Image failed to load</Text>
        : <img
            src={src}
            alt={alt}
            className={loaded ? className : hiddenClassName}
            onLoad={onLoad}
            onError={onError}
          />
      }
    </div>
  )
}

interface MediaWithFallbackProps {
  type: 'video' | 'audio'
  src: string
  className?: string
  preload?: 'none' | 'metadata' | 'auto'
  stopOnUnmount?: boolean
  showLoadingStatus?: boolean
}

export function MediaWithFallback({
  type,
  src,
  className,
  preload,
  stopOnUnmount = false,
  showLoadingStatus = false,
}: MediaWithFallbackProps) {
  const [error, setError] = useState(false)
  const [metadataSource, setMetadataSource] = useState<string | null>(null)
  const player = useRef<HTMLMediaElement | null>(null)
  const cleanupLifecycle = useRef({ generation: 0 })
  const handleError = useCallback(() => { setError(true) }, [])
  const handleMetadata = useCallback(() => { setMetadataSource(src) }, [src])
  const setPlayer = useCallback((element: HTMLMediaElement | null) => { player.current = element }, [])

  useLayoutEffect(() => {
    const element = player.current
    const lifecycle = cleanupLifecycle.current
    const generation = ++lifecycle.generation
    return () => {
      if (stopOnUnmount && element) {
        element.pause()
        // Strict Mode replays layout effects without removing the player.
        queueMicrotask(() => {
          if (lifecycle.generation !== generation) return
          element.removeAttribute('src')
          element.load()
        })
      }
    }
  }, [stopOnUnmount, src])

  if (error) {
    return <Text size={200} italic data-testid={`${type}-error`}>{type === 'video' ? 'Video' : 'Audio'} failed to load</Text>
  }

  return (
    <>
      {showLoadingStatus && metadataSource !== src && <Spinner size="small" label={`Loading ${type}...`} />}
      {type === 'video'
        ? <video ref={setPlayer} src={src} controls preload={preload} className={className} onLoadedMetadata={handleMetadata} onError={handleError} data-testid="video-player" />
        : <audio ref={setPlayer} src={src} controls preload={preload} className={className} onLoadedMetadata={handleMetadata} onError={handleError} data-testid="audio-player" />}
    </>
  )
}
