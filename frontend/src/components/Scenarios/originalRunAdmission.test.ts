import { formatOriginalRunReason } from './originalRunAdmission'

describe('formatOriginalRunReason', () => {
  it('should distinguish the closed validation scope from a busy concurrent slot', () => {
    expect(formatOriginalRunReason('validation_scope_exhausted')).toBe(
      'This validation preview has no remaining original-run admissions.',
    )
    expect(formatOriginalRunReason('capacity_busy')).toBe('The single original-run slot is busy.')
  })
})
