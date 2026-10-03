// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { PlaybackController, clampSlide, slideFromHash } = require(path.resolve(
  __dirname, '..', '..', '..', '..', 'build_scripts', 'pyrit_wrapped', 'web', 'playback.js'
))

function fixture(createOverride) {
  const calls = []
  const states = []
  const player = {
    load: track => calls.push(['load', track.youtube_id, track.start_seconds, track.end_seconds]),
    play: () => calls.push(['play']),
    pause: () => calls.push(['pause']),
    destroy: () => calls.push(['destroy']),
  }
  const controller = new PlaybackController({
    createPlayer: createOverride || (async () => player),
    status: (state, message) => states.push([state, message]),
  })
  const track = { youtube_id: '3GwjfUFyY6M', start_seconds: 0, end_seconds: 10 }
  return { controller, player, calls, states, track }
}

async function visible(controller, ratio = 1, width = 360, height = 203, pageVisible = true) {
  await controller.setVisibility({ ratio, width, height, pageVisible })
}

test('no player or network startup before enable', async () => {
  let created = 0
  const f = fixture(async () => { created += 1; return f.player })
  await f.controller.setTrack(f.track)
  await visible(f.controller)
  assert.equal(created, 0)
  assert.deepEqual(f.calls, [])
})

test('enabled visible player loads the exact bounded cue', async () => {
  const f = fixture()
  await f.controller.setTrack(f.track)
  await visible(f.controller)
  await f.controller.setEnabled(true)
  assert.deepEqual(f.calls, [['load', '3GwjfUFyY6M', 0, 10]])
})

test('half-visible, offscreen, or undersized players never load', async () => {
  for (const dimensions of [[0.5, 360, 203], [0, 360, 203], [1, 199, 203], [1, 360, 199]]) {
    let created = 0
    const f = fixture(async () => { created += 1; return f.player })
    await f.controller.setTrack(f.track)
    await visible(f.controller, ...dimensions)
    await f.controller.setEnabled(true)
    assert.equal(created, 0)
    assert.deepEqual(f.calls, [])
  }
})

test('navigation pauses the old cue before loading the next', async () => {
  const f = fixture()
  await f.controller.setTrack(f.track)
  await visible(f.controller)
  await f.controller.setEnabled(true)
  await f.controller.setTrack({ ...f.track, youtube_id: 'cJRw7rYOOzA' })
  assert.deepEqual(f.calls.slice(-2), [['pause'], ['load', 'cJRw7rYOOzA', 0, 10]])
})

test('scrolling or backgrounding pauses and visibility return resumes', async () => {
  const f = fixture()
  await f.controller.setTrack(f.track)
  await visible(f.controller)
  await f.controller.setEnabled(true)
  await visible(f.controller, 0)
  assert.equal(f.calls.at(-1)[0], 'pause')
  await visible(f.controller)
  assert.equal(f.calls.at(-1)[0], 'play')
  await visible(f.controller, 1, 360, 203, false)
  assert.equal(f.calls.at(-1)[0], 'pause')
})

test('cue end never silently restarts; replay is explicit', async () => {
  const f = fixture()
  await f.controller.setTrack(f.track)
  await visible(f.controller)
  await f.controller.setEnabled(true)
  f.controller.playerState(0)
  const before = f.calls.length
  await visible(f.controller)
  assert.equal(f.calls.length, before)
  await f.controller.replay()
  assert.equal(f.calls.at(-1)[0], 'load')
})

test('manual pause and autoplay blocking do not loop play attempts', async () => {
  const f = fixture()
  await f.controller.setTrack(f.track)
  await visible(f.controller)
  await f.controller.setEnabled(true)
  f.controller.blocked()
  const before = f.calls.length
  await visible(f.controller)
  assert.equal(f.calls.length, before)
  assert.equal(f.states.at(-1)[0], 'blocked')
  await f.controller.replay()
  f.controller.playerState(2)
  await visible(f.controller)
  assert.equal(f.states.at(-1)[0], 'paused')
})

test('late readiness uses only the latest slide', async () => {
  let resolve
  const f = fixture(() => new Promise(done => { resolve = done }))
  await f.controller.setTrack(f.track)
  await visible(f.controller)
  const first = f.controller.setEnabled(true)
  await Promise.resolve()
  const next = f.controller.setTrack({ ...f.track, youtube_id: 'cJRw7rYOOzA' })
  resolve(f.player)
  await Promise.all([first, next])
  assert.deepEqual(f.calls.filter(call => call[0] === 'load'), [['load', 'cJRw7rYOOzA', 0, 10]])
})

test('disable during loading destroys once and never starts', async () => {
  let resolve
  const f = fixture(() => new Promise(done => { resolve = done }))
  await f.controller.setTrack(f.track)
  await visible(f.controller)
  const loading = f.controller.setEnabled(true)
  await Promise.resolve()
  await f.controller.setEnabled(false)
  resolve(f.player)
  await loading
  assert.equal(f.calls.filter(call => call[0] === 'destroy').length, 1)
  assert.equal(f.calls.filter(call => call[0] === 'load').length, 0)
})

test('failed API is explicit and allows enabling again', async () => {
  const f = fixture(async () => { throw new Error('Network unavailable') })
  await f.controller.setTrack(f.track)
  await visible(f.controller)
  await f.controller.setEnabled(true)
  assert.equal(f.controller.enabled, false)
  assert.deepEqual(f.states.at(-1), ['error', 'Network unavailable'])
})

test('missing IDs and embed/referrer errors stay visible', async () => {
  const f = fixture()
  await f.controller.setTrack(null)
  await visible(f.controller)
  await f.controller.setEnabled(true)
  assert.equal(f.states.at(-1)[0], 'error')
  f.controller.failed(153)
  assert.match(f.states.at(-1)[1], /HTTP referrer/)
  f.controller.failed(150)
  assert.match(f.states.at(-1)[1], /does not allow/)
})

test('navigation clamps both boundaries', () => {
  assert.equal(clampSlide(-1, 8), 0)
  assert.equal(clampSlide(8, 8), 7)
  assert.equal(clampSlide(3, 8), 3)
})

test('deep links and same-document hash navigation use the correct chapter', () => {
  assert.equal(slideFromHash('#slide-7', 8), 6)
  assert.equal(slideFromHash('#slide-999', 8), 7)
  assert.equal(slideFromHash('#slide-0', 8), 0)
  assert.equal(slideFromHash('#not-a-chapter', 8), 0)
})
