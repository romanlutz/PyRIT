import { makeStyles, tokens } from '@fluentui/react-components'

import { mobileTouchTarget } from '@/styles/touchTargets'

export const useMultiSendSettingsStyles = makeStyles({
  trigger: {
    minWidth: '40px',
    paddingInline: tokens.spacingHorizontalXS,
    ...mobileTouchTarget,
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
    width: '280px',
    maxWidth: 'calc(100vw - 48px)',
  },
  count: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalM,
  },
  countButton: {
    ...mobileTouchTarget,
  },
  explanation: {
    color: tokens.colorNeutralForeground2,
  },
})
