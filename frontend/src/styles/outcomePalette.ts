import { tokens } from '@fluentui/react-components'

import type { AttackOutcome } from '@/types'

/** Shared categorical colors for saved attack outcomes, not application error messages. */
export const OUTCOME_PALETTE = {
  success: { color: tokens.colorPaletteGreenForeground1, tint: tokens.colorPaletteGreenBackground1 },
  failure: { color: tokens.colorPaletteRedForeground1, tint: tokens.colorPaletteRedBackground1 },
  error: { color: tokens.colorPaletteBlueForeground2, tint: tokens.colorPaletteBlueBackground2 },
  undetermined: { color: tokens.colorNeutralForeground3, tint: tokens.colorNeutralBackground3 },
} as const satisfies Record<AttackOutcome, { readonly color: string; readonly tint: string }>
