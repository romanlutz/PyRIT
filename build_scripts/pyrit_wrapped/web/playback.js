// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

(function (root, factory) {
  const api = factory()
  if (typeof module === 'object' && module.exports) module.exports = api
  else root.WrappedPlayback = api
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict'

  class PlaybackController {
    constructor({ createPlayer, status }) {
      this.createPlayer = createPlayer
      this.status = status
      this.player = null
      this.pending = null
      this.track = null
      this.enabled = false
      this.visible = false
      this.pageVisible = true
      this.loadedId = null
      this.ended = false
      this.manualPause = false
      this.revision = 0
      this.playRequested = false
    }

    setTrack(track) {
      this.pausePlayer()
      this.track = track
      this.loadedId = null
      this.ended = false
      this.manualPause = false
      this.revision += 1
      return this.sync()
    }

    setVisibility({ ratio, width, height, pageVisible }) {
      this.visible = ratio > 0.5 && width >= 200 && height >= 200
      this.pageVisible = pageVisible
      if (!this.allowed()) this.pausePlayer()
      return this.sync()
    }

    setEnabled(enabled) {
      this.enabled = enabled
      this.manualPause = false
      this.revision += 1
      if (!enabled) {
        this.pausePlayer()
        if (this.player) this.player.destroy()
        this.player = null
        this.loadedId = null
        this.status('off', 'Music is off.')
        return Promise.resolve()
      }
      return this.sync()
    }

    allowed() {
      return this.enabled && this.visible && this.pageVisible && this.track && this.track.youtube_id
    }

    pausePlayer() {
      this.playRequested = false
      if (this.player) this.player.pause()
    }

    ensurePlayer() {
      if (this.player) return Promise.resolve(this.player)
      if (!this.pending) {
        this.pending = Promise.resolve()
          .then(() => this.createPlayer())
          .then(player => {
            if (!this.enabled) {
              player.destroy()
              return null
            }
            this.player = player
            return player
          })
          .finally(() => { this.pending = null })
      }
      return this.pending
    }

    async sync() {
      if (!this.enabled) return
      if (!this.track || !this.track.youtube_id) {
        this.status('error', 'No verified YouTube video is configured for this chapter.')
        return
      }
      if (!this.visible || !this.pageVisible) {
        this.status('waiting', 'Music pauses while the player is off-screen or the page is in the background.')
        return
      }
      const revision = this.revision
      try {
        const player = await this.ensurePlayer()
        if (!player) return
        if (!this.allowed() || revision !== this.revision) {
          if (!this.allowed()) this.pausePlayer()
          return
        }
        if (this.loadedId !== this.track.youtube_id) {
          this.loadedId = this.track.youtube_id
          this.playRequested = true
          this.player.load(this.track)
          this.status('loading', 'Loading the visible YouTube cue. Use the player controls if autoplay is blocked.')
        } else if (!this.ended && !this.manualPause && !this.playRequested) {
          this.playRequested = true
          this.player.play()
          this.status('playing', 'Playing the current cue.')
        }
      } catch (error) {
        if (revision !== this.revision) return
        this.pending = null
        this.loadedId = null
        this.enabled = false
        this.status('error', error.message || 'YouTube playback failed. Check your connection and try again.')
      }
    }

    replay() {
      this.ended = false
      this.manualPause = false
      this.loadedId = null
      this.revision += 1
      return this.sync()
    }

    playerState(state) {
      if (state === 1) {
        if (!this.visible || !this.pageVisible) this.pausePlayer()
        else this.status('playing', 'Playing the current cue.')
      } else if (state === 0) {
        this.playRequested = false
        this.ended = true
        this.status('ended', 'Cue finished. Replay it or go to the next chapter.')
      } else if (state === 2 && this.allowed()) {
        this.playRequested = false
        this.manualPause = true
        this.status('paused', 'Cue paused. Use Replay cue or the YouTube controls.')
      }
    }

    blocked() {
      this.playRequested = false
      this.manualPause = true
      this.status('blocked', 'Your browser blocked autoplay. Press Replay cue or Play in the visible YouTube player.')
    }

    failed(code) {
      this.pausePlayer()
      this.manualPause = true
      const messages = {
        2: 'YouTube rejected this video reference.',
        5: 'YouTube could not play this video in this browser.',
        100: 'This YouTube video is unavailable or private.',
        101: 'The owner does not allow this video to be embedded.',
        150: 'The owner does not allow this video to be embedded.',
        153: 'YouTube needs an HTTP referrer. Open the local preview, not file://.',
      }
      this.status('error', `${messages[code] || 'YouTube playback failed.'} Try the Watch on YouTube link.`)
    }
  }

  function clampSlide(index, count) {
    return Math.max(0, Math.min(count - 1, index))
  }

  function slideFromHash(hash, count) {
    const match = /^#slide-(\d+)$/.exec(hash)
    return clampSlide(match ? Number(match[1]) - 1 : 0, count)
  }

  return { PlaybackController, clampSlide, slideFromHash }
})
