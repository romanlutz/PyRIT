import type { OriginalRunReason, OriginalRunStatus } from '@/types'

export const APPROVED_ORIGINAL_SCENARIO_NAME = 'benchmark.approved_original'

const REASON_MESSAGES: Record<OriginalRunReason, string> = {
  runner_not_configured: 'This server has no approved original runner.',
  operator_not_authorized: 'Your account is not admitted to this original run.',
  profile_not_admitted: 'The requested original-run profile is not approved.',
  admission_expired: 'This admission expired. Reload the approved scenario to request a new one.',
  model_route_unverified: 'The evaluated-model route is not verified.',
  capacity_busy: 'The single original-run slot is busy.',
  provider_unqualified: 'The original runner or host is not qualified yet.',
  cleanup_pending: 'The original worker has no verified cleanup receipt.',
  source_unverified: 'The original score or evidence could not be verified.',
}

const STATUS_MESSAGES: Record<OriginalRunStatus, string> = {
  unavailable: 'Unavailable',
  admission_pending: 'Admission pending',
  ready: 'Ready',
  running: 'Running',
  cancelling: 'Cancelling',
  completed: 'Original result verified',
  failed_ungraded: 'Failed without a verified original grade',
  failed_source_verified: 'Execution incomplete; original grade retained',
  cleanup_uncertain: 'Cleanup not verified',
}

export function formatOriginalRunReason(reason: string | null | undefined): string {
  if (!reason) return 'The original run is unavailable.'
  return REASON_MESSAGES[reason as OriginalRunReason] ?? 'The original run is unavailable.'
}

export function formatOriginalRunStatus(status: OriginalRunStatus): string {
  return STATUS_MESSAGES[status]
}
