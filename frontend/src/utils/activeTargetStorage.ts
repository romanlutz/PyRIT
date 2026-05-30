/**
 * localStorage wrapper for the active target selection.
 *
 * The active target is a per-browser UX preference: storing it server-side
 * would create wrong cross-user semantics (two operators sharing a backend
 * would fight over a single global "active"). All helpers tolerate
 * `Storage` access failures (private mode, disabled storage, quota errors)
 * silently — the UI gracefully degrades to an in-memory selection.
 */

export const ACTIVE_TARGET_STORAGE_KEY = 'copyrit:activeTargetRegistryName'

export function readActiveTargetName(): string | null {
  try {
    const value = window.localStorage.getItem(ACTIVE_TARGET_STORAGE_KEY)
    if (typeof value !== 'string' || value.length === 0) {
      return null
    }
    return value
  } catch {
    return null
  }
}

export function writeActiveTargetName(name: string): void {
  try {
    window.localStorage.setItem(ACTIVE_TARGET_STORAGE_KEY, name)
  } catch {
    // localStorage may be unavailable (e.g. quota, disabled, private mode).
    // The selection will simply not survive the next reload — that's
    // strictly better than crashing the app.
  }
}

export function clearActiveTargetName(): void {
  try {
    window.localStorage.removeItem(ACTIVE_TARGET_STORAGE_KEY)
  } catch {
    // See writeActiveTargetName.
  }
}
