import { makeStyles, tokens } from '@fluentui/react-components'

import { mobileTouchTargetHeight } from '@/styles/touchTargets'

export const useAnalyticsResultsTableStyles = makeStyles({
  root: { display: 'flex', flexDirection: 'column', gap: tokens.spacingVerticalS, minWidth: 0 },
  heading: { margin: 0 },
  header: { display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: tokens.spacingHorizontalM },
  scroll: { overflow: 'auto', maxHeight: '28rem', minWidth: 0 },
  table: { minWidth: '65rem', tableLayout: 'auto' },
  objective: { minWidth: '15rem', maxWidth: '25rem', overflowWrap: 'anywhere' },
  metadata: { maxWidth: '18rem', overflowWrap: 'anywhere' },
  secondary: { display: 'block', color: tokens.colorNeutralForeground2, overflowWrap: 'anywhere' },
  date: { whiteSpace: 'nowrap' },
  button: { ...mobileTouchTargetHeight },
})
