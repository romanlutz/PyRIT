// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

;(function () {
  'use strict'
  const data = JSON.parse(document.getElementById('wrapped-data').textContent)
  const sections = Array.from(document.querySelectorAll('.slide'))
  const chapters = Array.from(document.querySelectorAll('.chapter-button'))
  const previous = document.getElementById('previous-slide')
  const next = document.getElementById('next-slide')
  const toggle = document.getElementById('music-toggle')
  const replay = document.getElementById('replay-cue')
  const surface = document.getElementById('player-surface')
  const placeholder = document.getElementById('player-placeholder')
  const status = document.getElementById('music-status')
  let index = 0
  let ratio = 0
  let apiPromise = null
  let controller

  function loadYouTube() {
    if (window.YT && window.YT.Player) return Promise.resolve()
    if (apiPromise) return apiPromise
    apiPromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        apiPromise = null
        script.remove()
        reject(new Error('YouTube did not load. Check your connection and press Enable music again.'))
      }, 15000)
      const script = document.createElement('script')
      script.src = 'https://www.youtube.com/iframe_api'
      script.onerror = () => {
        clearTimeout(timeout)
        apiPromise = null
        script.remove()
        reject(new Error('The YouTube API could not load. Check your network or content blocker.'))
      }
      window.onYouTubeIframeAPIReady = () => {
        clearTimeout(timeout)
        resolve()
      }
      document.head.appendChild(script)
    })
    return apiPromise
  }

  async function createPlayer() {
    if (!/^https?:$/.test(window.location.protocol)) {
      throw new Error('YouTube needs the local HTTP preview. The slide deck still works from this file.')
    }
    await loadYouTube()
    return new Promise((resolve, reject) => {
      let ready = false
      function destroy() {
        clearTimeout(timeout)
        player.destroy()
        if (!document.getElementById('youtube-player')) {
          const replacement = document.createElement('div')
          replacement.id = 'youtube-player'
          surface.appendChild(replacement)
        }
        placeholder.hidden = false
        replay.disabled = true
      }
      const timeout = setTimeout(() => {
        if (!ready) {
          destroy()
          reject(new Error('The YouTube player did not become ready. Press Enable music to retry.'))
        }
      }, 15000)
      const player = new YT.Player('youtube-player', {
        width: '100%',
        height: '100%',
        playerVars: { autoplay: 0, controls: 1, playsinline: 1, origin: window.location.origin },
        events: {
          onReady: () => {
            ready = true
            clearTimeout(timeout)
            placeholder.hidden = true
            const frame = player.getIframe()
            frame.title = 'YouTube music video for the current chapter'
            frame.setAttribute('allow', 'autoplay; encrypted-media; picture-in-picture')
            replay.disabled = false
            resolve({
              load: track => player.loadVideoById({
                videoId: track.youtube_id,
                startSeconds: track.start_seconds,
                endSeconds: track.end_seconds,
              }),
              play: () => player.playVideo(),
              pause: () => player.pauseVideo(),
              destroy,
            })
          },
          onStateChange: event => controller.playerState(event.data),
          onAutoplayBlocked: () => controller.blocked(),
          onError: event => {
            if (!ready) {
              clearTimeout(timeout)
              destroy()
              reject(new Error(`YouTube player startup failed (${event.data}). Try the Watch on YouTube link.`))
            } else controller.failed(event.data)
          },
        },
      })
    })
  }

  controller = new WrappedPlayback.PlaybackController({
    createPlayer,
    status: (state, message) => {
      status.textContent = message
      status.classList.toggle('error', state === 'error')
      toggle.textContent = controller.enabled ? 'Disable music' : 'Enable music'
      toggle.setAttribute('aria-pressed', String(controller.enabled))
    },
  })

  function observeVisibility() {
    const bounds = surface.getBoundingClientRect()
    controller.setVisibility({
      ratio, width: bounds.width, height: bounds.height,
      pageVisible: document.visibilityState === 'visible',
    })
  }

  function showSlide(value, focus) {
    index = WrappedPlayback.clampSlide(value, sections.length)
    sections.forEach((section, position) => { section.hidden = position !== index })
    chapters.forEach((button, position) => {
      if (position === index) button.setAttribute('aria-current', 'step')
      else button.removeAttribute('aria-current')
    })
    previous.disabled = index === 0
    next.textContent = index === sections.length - 1 ? 'Restart' : 'Next \u2192'
    document.getElementById('chapter-position').textContent = `${index + 1} / ${sections.length}`
    const track = data.slides[index].track
    document.getElementById('track-title').textContent = track ? track.title : 'No selected track'
    document.getElementById('track-artist').textContent = track ? track.artist : ''
    document.getElementById('watch-video').href = track && track.youtube_id
      ? `https://www.youtube.com/watch?v=${track.youtube_id}` : 'https://www.youtube.com/'
    controller.setTrack(track)
    history.replaceState(null, '', `#slide-${index + 1}`)
    if (focus) sections[index].querySelector('h1').focus({ preventScroll: true })
  }

  previous.addEventListener('click', () => showSlide(index - 1, true))
  next.addEventListener('click', () => showSlide(index === sections.length - 1 ? 0 : index + 1, true))
  chapters.forEach((button, position) => button.addEventListener('click', () => showSlide(position, true)))
  document.querySelector('.wordmark').addEventListener('click', event => {
    event.preventDefault()
    showSlide(0, true)
  })
  document.addEventListener('keydown', event => {
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName) || event.target.isContentEditable) return
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault()
      showSlide(index + (event.key === 'ArrowRight' ? 1 : -1), true)
    }
  })
  toggle.addEventListener('click', () => {
    const enabling = !controller.enabled
    controller.setEnabled(enabling)
    toggle.textContent = enabling ? 'Disable music' : 'Enable music'
    toggle.setAttribute('aria-pressed', String(enabling))
    if (enabling) surface.scrollIntoView({ block: 'nearest', behavior: 'auto' })
  })
  replay.addEventListener('click', () => controller.replay())
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      ratio = entries[0].intersectionRatio
      observeVisibility()
    }, { threshold: [0, 0.5, 0.51, 1] }).observe(surface)
  } else {
    toggle.disabled = true
    status.textContent = 'This browser cannot verify player visibility. Use Watch on YouTube for music.'
  }
  document.addEventListener('visibilitychange', observeVisibility)
  window.addEventListener('resize', observeVisibility)
  window.addEventListener('hashchange', () => {
    showSlide(WrappedPlayback.slideFromHash(window.location.hash, sections.length), false)
  })
  showSlide(WrappedPlayback.slideFromHash(window.location.hash, sections.length), false)
})()
