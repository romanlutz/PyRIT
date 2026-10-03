// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

;(function () {
  'use strict'
  const data = JSON.parse(document.getElementById('wrapped-data').textContent)
  const sections = Array.from(document.querySelectorAll('.slide'))
  const chapters = Array.from(document.querySelectorAll('.chapter-button'))
  const previous = document.getElementById('previous-slide')
  const next = document.getElementById('next-slide')
  const start = document.getElementById('start-take')
  const pause = document.getElementById('pause-take')
  const finish = document.getElementById('finish-take')
  const duration = document.getElementById('cue-duration')
  const advance = document.getElementById('auto-advance')
  const status = document.getElementById('recording-status')
  const motion = document.getElementById('motion-toggle')
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
  let displayed = -1
  let recording

  function showSlide(index, focus = false) {
    const changed = index !== displayed
    displayed = index
    sections.forEach((section, position) => { section.hidden = position !== index })
    chapters.forEach((button, position) => {
      if (position === index) button.setAttribute('aria-current', 'step')
      else button.removeAttribute('aria-current')
    })
    document.getElementById('chapter-position').textContent = `${index + 1} / ${sections.length}`
    document.body.dataset.chapter = data.slides[index].type
    if (changed) {
      const track = data.slides[index].track
      document.getElementById('track-title').textContent = track ? track.title : 'No selected track'
      document.getElementById('track-artist').textContent = track ? track.artist : ''
      const link = document.getElementById('spotify-link')
      link.hidden = !track
      if (track) link.href = `https://open.spotify.com/search/${encodeURIComponent(`${track.title} ${track.artist}`)}`
      history.replaceState(null, '', `#slide-${index + 1}`)
    }
    if (focus) sections[index].querySelector('h1').focus({ preventScroll: true })
  }

  function update(view) {
    const active = ['countdown', 'running', 'paused'].includes(view.state)
    document.body.classList.toggle('recording', active)
    showSlide(view.index)
    start.disabled = active
    duration.disabled = active
    advance.disabled = active
    pause.disabled = !['running', 'paused'].includes(view.state) || view.remaining === 0
    pause.textContent = view.state === 'paused' ? 'Resume' : 'Pause'
    finish.disabled = !active
    previous.disabled = view.index === 0 || view.state === 'countdown'
    next.disabled = view.state === 'countdown'
    next.textContent = view.index === sections.length - 1 ? (active ? 'Finish take' : 'Restart') : 'Next \u2192'
    chapters.forEach(button => { button.disabled = view.state === 'countdown' })
    const clock = document.getElementById('cue-clock')
    clock.textContent = view.state === 'countdown' ? String(view.countdown)
      : active ? `${view.remaining}s` : 'Ready'
    const overlay = document.getElementById('countdown')
    overlay.hidden = view.state !== 'countdown'
    document.getElementById('countdown-number').textContent = String(view.countdown)
    const messages = {
      idle: view.reason === 'countdown-cancelled' ? 'Countdown cancelled. Start a new take when ready.'
        : 'Queue the songs in Spotify. You control the music; this page only times the slides.',
      countdown: 'Get ready. The first slide starts after the countdown.',
      running: 'Take running. Change songs manually in Spotify at each chapter.',
      paused: view.reason === 'background' ? 'Paused because the tab was hidden. Resume when ready.'
        : view.reason === 'cue-complete' ? 'Cue complete. Change the song, then choose Next.'
        : 'Take paused. Spotify is not paused by this page.',
      finished: 'Take finished. Download the cue sheet for the actual slide timestamps.',
    }
    if (status.textContent !== messages[view.state]) status.textContent = messages[view.state]
  }

  recording = new WrappedRecording.RecordingController({
    count: sections.length, now: () => performance.now(), update,
  })
  recording.index = WrappedRecording.slideFromHash(window.location.hash, sections.length)
  recording.notify()
  const timer = window.setInterval(() => {
    if (recording.state === 'running' || recording.state === 'countdown') recording.tick()
  }, 100)
  window.addEventListener('pagehide', () => window.clearInterval(timer))

  function navigate(index) {
    recording.goTo(index)
    showSlide(recording.index, true)
  }

  previous.addEventListener('click', () => navigate(recording.index - 1))
  next.addEventListener('click', () => {
    if (recording.index === sections.length - 1) {
      if (['running', 'paused'].includes(recording.state)) recording.finish()
      else navigate(0)
    } else navigate(recording.index + 1)
  })
  chapters.forEach((button, index) => button.addEventListener('click', () => navigate(index)))
  document.querySelector('.wordmark').addEventListener('click', event => {
    event.preventDefault()
    navigate(0)
  })
  document.addEventListener('keydown', event => {
    if (/^(INPUT|TEXTAREA|SELECT|BUTTON|A|SUMMARY)$/.test(event.target.tagName) || event.target.isContentEditable) return
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault()
      navigate(recording.index + (event.key === 'ArrowRight' ? 1 : -1))
    } else if (event.code === 'Space' && ['running', 'paused'].includes(recording.state)) {
      event.preventDefault()
      if (recording.state === 'paused') recording.resume()
      else recording.pause()
    }
  })
  start.addEventListener('click', () => {
    if (!duration.reportValidity()) return
    recording.start({ seconds: Number(duration.value), autoAdvance: advance.checked })
    document.getElementById('slide-content').scrollIntoView({ block: 'start' })
  })
  pause.addEventListener('click', () => {
    if (recording.state === 'paused') recording.resume()
    else recording.pause()
  })
  finish.addEventListener('click', () => recording.finish())
  document.addEventListener('visibilitychange', () => {
    document.body.classList.toggle('page-hidden', document.hidden)
    if (document.hidden) recording.pause('background')
  })
  window.addEventListener('hashchange', () => {
    navigate(WrappedRecording.slideFromHash(window.location.hash, sections.length))
    history.replaceState(null, '', `#slide-${recording.index + 1}`)
  })

  function setMotion(enabled) {
    document.body.classList.toggle('motion-disabled', !enabled)
    motion.setAttribute('aria-pressed', String(enabled))
    motion.textContent = enabled ? 'Pause animation' : 'Resume animation'
  }
  setMotion(!reducedMotion.matches)
  motion.addEventListener('click', () => {
    setMotion(motion.getAttribute('aria-pressed') !== 'true')
  })
  reducedMotion.addEventListener('change', event => setMotion(!event.matches))
  const confetti = document.getElementById('confetti')
  for (let i = 0; i < 24; i += 1) {
    const piece = document.createElement('span')
    piece.style.cssText = `--x:${(i * 43) % 100}%;--delay:${-(i % 8)}s;--duration:${5 + i % 4}s;--spin:${180 + i * 31}deg`
    confetti.appendChild(piece)
  }
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      document.getElementById('slide-content').classList.toggle('offscreen', !entries[0].isIntersecting)
    }).observe(document.getElementById('slide-content'))
  }
  document.getElementById('fullscreen-toggle').addEventListener('click', async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen()
      else if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen()
      else status.textContent = 'Fullscreen is unavailable here. Open this page in your browser for recording.'
    } catch (error) {
      status.textContent = `Fullscreen could not start: ${error.message}. Use your browser's fullscreen control.`
    }
  })
  document.addEventListener('fullscreenchange', () => {
    document.getElementById('fullscreen-toggle').textContent = document.fullscreenElement ? 'Exit fullscreen' : 'Fullscreen'
  })
  document.getElementById('download-cues').addEventListener('click', () => {
    const seconds = Number(duration.value)
    if (!duration.reportValidity()) return
    const sheet = {
      title: data.title,
      music_control: 'manual; no audio is captured or played by the deck',
      timeline: 'Actual timestamps are relative to countdown completion and include pauses.',
      planned: data.slides.map((slide, index) => ({
        slide: index + 1, title: slide.title, track: slide.track,
        start_seconds: index * seconds, duration_seconds: seconds,
      })),
      actual: recording.entries,
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify(sheet, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'pyrit-wrapped-cues.json'
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  })
})()
