import { makeStyles, shorthands, tokens } from '@fluentui/react-components'

import { mobileTouchTargetHeight, NARROW_VIEWPORT_QUERY } from '@/styles/touchTargets'
import { OUTCOME_PALETTE } from '@/styles/outcomePalette'

function outcomeFill(color: string) {
  return {
    backgroundColor: color,
    ':hover': { backgroundColor: color },
    ':active': { backgroundColor: color },
  }
}

export const useAnalyticsGroupsStyles = makeStyles({
  root: { display: 'flex', flexDirection: 'column', gap: tokens.spacingVerticalM, minWidth: 0 },
  groups: {
    display: 'flex',
    flexDirection: 'column',
    overflowY: 'auto',
    maxHeight: '26rem',
    gap: tokens.spacingVerticalM,
    padding: tokens.spacingHorizontalXS,
  },
  group: {
    display: 'grid',
    gridTemplateColumns: 'minmax(8rem, 1fr) minmax(16rem, 3fr)',
    alignItems: 'center',
    gap: tokens.spacingHorizontalL,
    paddingBottom: tokens.spacingVerticalM,
    borderBottom: `1px solid ${tokens.colorNeutralStroke2}`,
    [NARROW_VIEWPORT_QUERY]: { gridTemplateColumns: 'minmax(0, 1fr)', gap: tokens.spacingVerticalS },
  },
  groupLabel: { display: 'flex', flexDirection: 'column', alignItems: 'flex-start', minWidth: 0 },
  groupButton: {
    maxWidth: '100%',
    overflowWrap: 'anywhere',
    justifyContent: 'flex-start',
    textAlign: 'start',
    ...mobileTouchTargetHeight,
  },
  visualization: { display: 'flex', flexDirection: 'column', gap: tokens.spacingVerticalXS, minWidth: 0 },
  bar: {
    display: 'flex',
    width: '100%',
    height: tokens.spacingVerticalL,
    backgroundColor: tokens.colorNeutralBackground4,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    ...mobileTouchTargetHeight,
  },
  segment: {
    display: 'block',
    height: '100%',
    '@media (forced-colors: active)': { borderRight: '1px solid currentColor', boxSizing: 'border-box' },
  },
  segmentButton: {
    minWidth: 0,
    height: '100%',
    padding: 0,
    border: 0,
    borderRadius: 0,
    '@media (forced-colors: active)': { borderRight: '1px solid currentColor' },
  },
  rateButton: {
    minWidth: 0,
    padding: 0,
    borderRadius: 0,
    justifyContent: 'flex-start',
  },
  success: outcomeFill(OUTCOME_PALETTE.success.color),
  failure: outcomeFill(OUTCOME_PALETTE.failure.color),
  error: outcomeFill(OUTCOME_PALETTE.error.color),
  undetermined: outcomeFill(OUTCOME_PALETTE.undetermined.color),
  rate: { backgroundColor: tokens.colorBrandBackground },
  unavailable: { ...shorthands.borderStyle('dashed'), backgroundColor: tokens.colorNeutralBackground2 },
  outcomes: { display: 'flex', flexWrap: 'wrap', gap: tokens.spacingHorizontalXS },
  swatch: { width: tokens.spacingHorizontalS, height: tokens.spacingVerticalS, border: '1px solid currentColor' },
  button: { columnGap: tokens.spacingHorizontalXS, ...mobileTouchTargetHeight },
  scroll: { overflowX: 'auto', minWidth: 0 },
  table: { minWidth: '50rem' },
  caption: {
    textAlign: 'start',
    paddingBlock: tokens.spacingVerticalS,
    fontWeight: tokens.fontWeightSemibold,
  },
  number: { fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' },
  legend: { display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: tokens.spacingHorizontalS },
})
