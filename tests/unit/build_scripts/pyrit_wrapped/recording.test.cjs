// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { RecordingController, clampSlide, slideFromHash } = require(path.resolve(
  __dirname, '..', '..', '..', '..', 'build_scripts', 'pyrit_wrapped', 'web', 'recording.js'
))

function fixture(count = 3) {
  let now = 0
  const views = []
  const controller = new RecordingController({ count, now: () => now, update: view => views.push(view) })
  return {
    controller, views,
    advance: milliseconds => { now += milliseconds; controller.tick() },
    start: autoAdvance => controller.start({ seconds: 10, autoAdvance }),
  }
}

test('idle does not start recording or music', () => {
  const f = fixture()
  f.advance(30000)
  assert.equal(f.controller.state, 'idle')
  assert.deepEqual(f.controller.entries, [])
})

test('three second countdown starts a new take at the first slide', () => {
  const f = fixture()
  f.controller.goTo(2)
  f.start(false)
  assert.equal(f.controller.index, 0)
  assert.equal(f.views.at(-1).countdown, 3)
  f.advance(2000)
  assert.equal(f.views.at(-1).countdown, 1)
  f.controller.goTo(2)
  assert.equal(f.controller.index, 0)
  f.advance(1000)
  assert.equal(f.controller.state, 'running')
  assert.equal(f.views.at(-1).remaining, 10)
})

test('manual cues stop at the end and wait for navigation', () => {
  const f = fixture()
  f.start(false)
  f.advance(3000)
  f.advance(10000)
  assert.equal(f.controller.state, 'paused')
  assert.equal(f.views.at(-1).reason, 'cue-complete')
  assert.equal(f.controller.index, 0)
  f.controller.resume()
  assert.equal(f.controller.state, 'paused')
  f.advance(2000)
  f.controller.goTo(1)
  assert.equal(f.controller.state, 'running')
  assert.equal(f.views.at(-1).remaining, 10)
  assert.deepEqual(f.controller.entries[0], { slide: 1, start_seconds: 0, end_seconds: 12, active_seconds: 10 })
})

test('automatic cues advance and finish without looping', () => {
  const f = fixture()
  f.start(true)
  f.advance(3000)
  f.advance(10000)
  assert.equal(f.controller.index, 1)
  f.advance(10000)
  f.advance(10000)
  assert.equal(f.controller.index, 2)
  assert.equal(f.controller.state, 'finished')
  assert.equal(f.controller.entries.length, 3)
  assert.deepEqual(f.controller.entries.map(entry => entry.start_seconds), [0, 10, 20])
  assert.deepEqual(f.controller.entries.map(entry => entry.end_seconds), [10, 20, 30])
  f.advance(10000)
  assert.equal(f.controller.entries.length, 3)
})

test('pause and resume preserve remaining time and include pause in the timeline', () => {
  const f = fixture()
  f.start(false)
  f.advance(3000)
  f.advance(2500)
  f.controller.pause()
  assert.equal(f.views.at(-1).remaining, 8)
  f.advance(5000)
  assert.equal(f.views.at(-1).remaining, 8)
  f.controller.resume()
  f.advance(1000)
  f.controller.finish()
  assert.deepEqual(f.controller.entries[0], { slide: 1, start_seconds: 0, end_seconds: 8.5, active_seconds: 3.5 })
})

test('background pause never auto-resumes and countdown cancellation has no fake timeline', () => {
  const f = fixture()
  f.start(true)
  f.advance(3000)
  f.controller.pause('background')
  f.advance(60000)
  assert.equal(f.controller.state, 'paused')
  assert.equal(f.controller.index, 0)
  assert.equal(f.views.at(-1).reason, 'background')
  const countdown = fixture()
  countdown.start(false)
  countdown.controller.pause('background')
  assert.equal(countdown.controller.state, 'idle')
  assert.deepEqual(countdown.controller.entries, [])
})

test('a delayed timer never skips unseen slides or invents ideal timestamps', () => {
  const f = fixture()
  f.start(true)
  f.advance(5000)
  f.advance(45000)
  assert.equal(f.controller.index, 1)
  assert.equal(f.controller.entries[0].end_seconds, 45)
  assert.equal(f.views.at(-1).remaining, 10)
})

test('manual jumps record revisits in actual order, including during pauses', () => {
  const f = fixture()
  f.start(false)
  f.advance(3000)
  f.advance(1000)
  f.controller.goTo(2)
  f.advance(2000)
  f.controller.pause()
  f.controller.goTo(0)
  f.advance(3000)
  f.controller.finish()
  assert.deepEqual(f.controller.entries.map(entry => entry.slide), [1, 3, 1])
  assert.deepEqual(f.controller.entries.map(entry => entry.active_seconds), [1, 2, 3])
})

test('repeated finish is idempotent and a new take clears prior timestamps', () => {
  const f = fixture()
  f.start(false)
  f.advance(3000)
  f.controller.finish()
  f.controller.finish()
  assert.equal(f.controller.entries.length, 1)
  f.start(false)
  assert.deepEqual(f.controller.entries, [])
  f.controller.finish()
  assert.equal(f.controller.state, 'idle')
  assert.deepEqual(f.controller.entries, [])
})

test('starting an active take does not erase its progress', () => {
  const f = fixture()
  f.start(false)
  f.advance(3000)
  f.advance(5000)
  f.start(true)
  assert.equal(f.controller.state, 'running')
  assert.equal(f.views.at(-1).remaining, 5)
})

test('invalid slide counts and durations fail explicitly', () => {
  assert.throws(() => fixture(0), RangeError)
  for (const seconds of [0, 4, 121, NaN, Infinity]) {
    const f = fixture()
    assert.throws(() => f.controller.start({ seconds, autoAdvance: false }), RangeError)
  }
})

test('navigation clamps both boundaries and parses deep links', () => {
  assert.equal(clampSlide(-1, 8), 0)
  assert.equal(clampSlide(8, 8), 7)
  assert.equal(slideFromHash('#slide-7', 8), 6)
  assert.equal(slideFromHash('#slide-999', 8), 7)
  assert.equal(slideFromHash('#slide-0', 8), 0)
  assert.equal(slideFromHash('#not-a-chapter', 8), 0)
})
