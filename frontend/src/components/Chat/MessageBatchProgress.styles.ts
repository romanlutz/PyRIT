import { makeStyles, tokens } from '@fluentui/react-components'

import { mobileTouchTarget } from '@/styles/touchTargets'

export const useMessageBatchProgressStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    flexShrink: 0,
    gap: tokens.spacingVerticalXS,
    padding: `${tokens.spacingVerticalS} ${tokens.spacingHorizontalL}`,
    maxHeight: '30vh',
    overflowY: 'auto',
  },
  details: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
  },
  button: {
    ...mobileTouchTarget,
  },
})
