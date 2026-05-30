import {
  ACTIVE_TARGET_STORAGE_KEY,
  clearActiveTargetName,
  readActiveTargetName,
  writeActiveTargetName,
} from './activeTargetStorage'

describe('activeTargetStorage', () => {
  beforeEach(() => {
    window.localStorage.clear()
    jest.restoreAllMocks()
  })

  describe('readActiveTargetName', () => {
    it('returns null when nothing has been stored', () => {
      expect(readActiveTargetName()).toBeNull()
    })

    it('returns the stored registry name', () => {
      window.localStorage.setItem(ACTIVE_TARGET_STORAGE_KEY, 'openai_chat_gpt4')
      expect(readActiveTargetName()).toBe('openai_chat_gpt4')
    })

    it('returns null when the stored value is an empty string', () => {
      window.localStorage.setItem(ACTIVE_TARGET_STORAGE_KEY, '')
      expect(readActiveTargetName()).toBeNull()
    })

    it('returns null when localStorage.getItem throws', () => {
      jest.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
        throw new Error('storage disabled')
      })
      expect(readActiveTargetName()).toBeNull()
    })
  })

  describe('writeActiveTargetName', () => {
    it('round-trips through readActiveTargetName', () => {
      writeActiveTargetName('my_target')
      expect(readActiveTargetName()).toBe('my_target')
    })

    it('overwrites a previously stored value', () => {
      writeActiveTargetName('first')
      writeActiveTargetName('second')
      expect(readActiveTargetName()).toBe('second')
    })

    it('does not throw when localStorage.setItem throws', () => {
      jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        throw new Error('quota exceeded')
      })
      expect(() => writeActiveTargetName('anything')).not.toThrow()
    })
  })

  describe('clearActiveTargetName', () => {
    it('removes the stored key', () => {
      writeActiveTargetName('to_be_cleared')
      clearActiveTargetName()
      expect(readActiveTargetName()).toBeNull()
      expect(window.localStorage.getItem(ACTIVE_TARGET_STORAGE_KEY)).toBeNull()
    })

    it('is a no-op when nothing is stored', () => {
      expect(() => clearActiveTargetName()).not.toThrow()
      expect(readActiveTargetName()).toBeNull()
    })

    it('does not throw when localStorage.removeItem throws', () => {
      jest.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => {
        throw new Error('storage disabled')
      })
      expect(() => clearActiveTargetName()).not.toThrow()
    })
  })
})
