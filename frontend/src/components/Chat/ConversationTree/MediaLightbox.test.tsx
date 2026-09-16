import React, { useState } from 'react'

import { Button, FluentProvider, webLightTheme } from '@fluentui/react-components'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { ConversationTreeNode } from '@/types'

import { treeNode, treePiece, treePreview } from './__fixtures__/treeFixtures'
import MediaLightbox from './MediaLightbox'
import type { PreviewResult } from './treeReadCoordinator'

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <FluentProvider theme={webLightTheme}>{children}</FluentProvider>
)

interface MediaHarnessProps {
  node: ConversationTreeNode
  result?: PreviewResult
  onRetry?: () => void
}

function MediaHarness({ node, result, onRetry = jest.fn() }: MediaHarnessProps) {
  const [opener, setOpener] = useState<HTMLElement | null>(null)
  return (
    <>
      <Button onClick={(event) => { setOpener(event.currentTarget) }}>Inspect media</Button>
      {opener && <MediaLightbox node={node} pieceIndex={0} result={result} opener={opener} onClose={() => { setOpener(null) }} onRetry={onRetry} />}
    </>
  )
}

describe('MediaLightbox', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    jest.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    jest.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
  })

  afterEach(() => { jest.restoreAllMocks() })

  it('should show a local loading state and close with X while restoring focus', async () => {
    const user = userEvent.setup()
    const node = treeNode('image', { piece_types: ['image_path'] })
    render(<TestWrapper><MediaHarness node={node} result={{ loading: true }} /></TestWrapper>)
    const opener = screen.getByRole('button', { name: /inspect media/i })
    await user.click(opener)
    expect(screen.getByText(/loading full image/i)).toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /close media/i }))
    await waitFor(() => { expect(screen.queryByRole('dialog')).not.toBeInTheDocument(); expect(opener).toHaveFocus() })
  })

  it('should display an image and close with Escape without replacing its opener', async () => {
    const user = userEvent.setup()
    const node = treeNode('image', { piece_types: ['image_path'] })
    const result = { loading: false, preview: treePreview(node.message, [treePiece({
      data_type: 'image_path', filename: 'garden.png', text: null, media_url: 'https://example.test/garden.png',
    })]) }
    render(<TestWrapper><MediaHarness node={node} result={result} /></TestWrapper>)
    const opener = screen.getByRole('button', { name: /inspect media/i })
    await user.click(opener)
    fireEvent.load(screen.getByAltText('garden.png'))
    expect(screen.getByRole('img', { name: 'garden.png' })).toHaveAttribute('src', 'https://example.test/garden.png')
    await user.keyboard('{Escape}')
    await waitFor(() => { expect(screen.queryByRole('dialog')).not.toBeInTheDocument(); expect(opener).toHaveFocus() })
  })

  it.each(['audio', 'video'] as const)('should never autoplay %s and stop playback and downloads on close', async (type: 'audio' | 'video') => {
    const user = userEvent.setup()
    const node = treeNode(type, { piece_types: [`${type}_path`] })
    const result = { loading: false, preview: treePreview(node.message, [treePiece({
      data_type: `${type}_path`, filename: `garden.${type}`, text: null, media_url: `https://example.test/${type}`,
    })]) }
    render(<TestWrapper><MediaHarness node={node} result={result} /></TestWrapper>)
    await user.click(screen.getByRole('button', { name: /inspect media/i }))
    const player = screen.getByTestId(`${type}-player`)
    expect(player).toHaveAttribute('controls')
    expect(player).not.toHaveAttribute('autoplay')
    expect(player).toHaveAttribute('preload', 'metadata')
    expect(screen.getByText(`Loading ${type}...`)).toBeInTheDocument()
    fireEvent.loadedMetadata(player)
    expect(screen.queryByText(`Loading ${type}...`)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /close media/i }))
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalledTimes(1)
    expect(HTMLMediaElement.prototype.load).toHaveBeenCalledTimes(1)
    expect(player).not.toHaveAttribute('src')
  })

  it('should expose a recoverable request error without mounting media', async () => {
    const user = userEvent.setup()
    const retry = jest.fn()
    render(<TestWrapper><MediaHarness node={treeNode('image', { piece_types: ['image_path'] })} result={{ loading: false, error: 'Media temporarily unavailable' }} onRetry={retry} /></TestWrapper>)
    await user.click(screen.getByRole('button', { name: /inspect media/i }))
    expect(screen.getByText(/media temporarily unavailable/i)).toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /reload media/i }))
    expect(retry).toHaveBeenCalledTimes(1)
  })

  it('should reuse image loading errors and permit reloading the same URL', async () => {
    const user = userEvent.setup()
    const retry = jest.fn()
    const node = treeNode('image', { piece_types: ['image_path'] })
    const result = { loading: false, preview: treePreview(node.message, [treePiece({
      data_type: 'image_path', filename: 'garden.png', text: null, media_url: 'https://example.test/garden.png',
    })]) }
    render(<TestWrapper><MediaHarness node={node} result={result} onRetry={retry} /></TestWrapper>)
    await user.click(screen.getByRole('button', { name: /inspect media/i }))
    fireEvent.error(screen.getByAltText('garden.png'))
    expect(screen.getByText(/image failed to load/i)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /reload media/i }))
    expect(retry).toHaveBeenCalledTimes(1)
    fireEvent.load(screen.getByAltText('garden.png'))
    expect(screen.getByRole('img', { name: 'garden.png' })).toBeInTheDocument()
  })

  it.each(['audio', 'video'] as const)('should reuse the %s error presentation', async (type: 'audio' | 'video') => {
    const user = userEvent.setup()
    const node = treeNode(type, { piece_types: [`${type}_path`] })
    const result = { loading: false, preview: treePreview(node.message, [treePiece({
      data_type: `${type}_path`, text: null, media_url: `https://example.test/${type}`,
    })]) }
    render(<TestWrapper><MediaHarness node={node} result={result} /></TestWrapper>)
    await user.click(screen.getByRole('button', { name: /inspect media/i }))
    fireEvent.error(screen.getByTestId(`${type}-player`))
    expect(screen.getByText(new RegExp(`${type} failed to load`, 'i'))).toBeInTheDocument()
  })

  it('should label unavailable or unsupported media instead of showing a broken player', async () => {
    const user = userEvent.setup()
    const node = treeNode('file', { piece_types: ['binary'] })
    const result = { loading: false, preview: treePreview(node.message, [treePiece({ data_type: 'binary', text: null })]) }
    render(<TestWrapper><MediaHarness node={node} result={result} /></TestWrapper>)
    await user.click(screen.getByRole('button', { name: /inspect media/i }))
    expect(screen.getByText(/this media is unavailable/i)).toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(screen.queryByTestId('video-player')).not.toBeInTheDocument()
  })
})
