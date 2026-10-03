// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

(function (root, factory) {
  const api = factory()
  if (typeof module === 'object' && module.exports) module.exports = api
  else root.WrappedRecording = api
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict'

  function clampSlide(index, count) {
    return Math.max(0, Math.min(count - 1, index))
  }

  function slideFromHash(hash, count) {
    const match = /^#slide-(\d+)$/.exec(hash)
    return clampSlide(match ? Number(match[1]) - 1 : 0, count)
  }

  class RecordingController {
    constructor({ count, now, update }) {
      if (!Number.isInteger(count) || count < 1) throw new RangeError('A take needs at least one slide.')
      this.count = count
      this.now = now
      this.update = update
      this.index = 0
      this.state = 'idle'
      this.seconds = 10
      this.autoAdvance = false
      this.entries = []
      this.elapsed = 0
      this.startedAt = null
      this.entryStart = null
      this.lastTick = null
      this.reason = ''
    }

    start({ seconds, autoAdvance }) {
      if (!Number.isFinite(seconds) || seconds < 5 || seconds > 120) {
        throw new RangeError('Choose a slide duration from 5 to 120 seconds.')
      }
      if (['countdown', 'running', 'paused'].includes(this.state)) return
      this.seconds = seconds
      this.autoAdvance = autoAdvance
      this.index = 0
      this.entries = []
      this.elapsed = 0
      this.startedAt = null
      this.entryStart = null
      this.countdownStart = this.now()
      this.state = 'countdown'
      this.reason = ''
      this.notify()
    }

    tick() {
      const now = this.now()
      if (this.state === 'countdown' && now - this.countdownStart >= 3000) {
        this.state = 'running'
        this.startedAt = now
        this.entryStart = now
        this.lastTick = now
      } else if (this.state === 'running') {
        this.elapsed += Math.max(0, now - this.lastTick)
        this.lastTick = now
        if (this.elapsed >= this.seconds * 1000) {
          if (this.autoAdvance && this.index === this.count - 1) this.finish()
          else if (this.autoAdvance) this.goTo(this.index + 1)
          else {
            this.state = 'paused'
            this.reason = 'cue-complete'
          }
        }
      }
      this.notify()
    }

    pause(reason = 'manual') {
      if (this.state === 'countdown') {
        this.state = 'idle'
        this.reason = 'countdown-cancelled'
      } else if (this.state === 'running') {
        this.elapsed += Math.max(0, this.now() - this.lastTick)
        this.state = 'paused'
        this.reason = reason
      }
      this.notify()
    }

    resume() {
      if (this.state !== 'paused' || this.elapsed >= this.seconds * 1000) return
      this.lastTick = this.now()
      this.state = 'running'
      this.reason = ''
      this.notify()
    }

    goTo(index) {
      if (this.state === 'countdown') return
      const next = clampSlide(index, this.count)
      if (next === this.index) return
      if (this.state === 'running' || this.state === 'paused') {
        this.closeEntry()
        this.entryStart = this.now()
        this.lastTick = this.now()
        this.elapsed = 0
        this.state = 'running'
        this.reason = ''
      }
      this.index = next
      this.notify()
    }

    finish() {
      if (this.state === 'countdown') {
        this.state = 'idle'
        this.reason = 'countdown-cancelled'
      } else if (this.state === 'running' || this.state === 'paused') {
        this.closeEntry()
        this.state = 'finished'
        this.reason = ''
      }
      this.notify()
    }

    closeEntry() {
      const now = this.now()
      const active = this.elapsed + (this.state === 'running' ? Math.max(0, now - this.lastTick) : 0)
      this.entries.push({
        slide: this.index + 1,
        start_seconds: (this.entryStart - this.startedAt) / 1000,
        end_seconds: (now - this.startedAt) / 1000,
        active_seconds: active / 1000,
      })
      this.entryStart = null
    }

    notify() {
      this.update({
        index: this.index,
        state: this.state,
        reason: this.reason,
        countdown: this.state === 'countdown'
          ? Math.max(1, Math.ceil(3 - (this.now() - this.countdownStart) / 1000)) : 0,
        remaining: Math.max(0, Math.ceil(this.seconds - this.elapsed / 1000)),
      })
    }
  }

  return { RecordingController, clampSlide, slideFromHash }
})
