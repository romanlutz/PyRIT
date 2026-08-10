import type { AttackOutcome } from '@/types'

export const ATTACK_OUTCOME_BADGE_COLORS: Record<
  AttackOutcome,
  'success' | 'danger' | 'warning' | 'informative'
> = {
  success: 'success',
  failure: 'danger',
  error: 'warning',
  undetermined: 'informative',
}

export function formatAttackOutcome(outcome: AttackOutcome | null | undefined): string {
  if (!outcome) return 'Unknown'
  return outcome.replace(/^\w/, (letter) => letter.toUpperCase())
}
