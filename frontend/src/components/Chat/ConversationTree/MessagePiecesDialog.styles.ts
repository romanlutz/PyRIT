import { makeStyles, tokens } from '@fluentui/react-components'

import { mobileTouchTarget, mobileTouchTargetHeight } from '@/styles/touchTargets'

export const useMessagePiecesDialogStyles = makeStyles({
  surface: {
    width: 'min(42rem, calc(100vw - 2rem))',
    maxWidth: 'calc(100vw - 2rem)',
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    rowGap: tokens.spacingVerticalM,
    maxHeight: '65dvh',
    overflowY: 'auto',
    minWidth: 0,
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    rowGap: tokens.spacingVerticalM,
    listStylePosition: 'inside',
    padding: 0,
    margin: 0,
  },
  piece: {
    paddingBottom: tokens.spacingVerticalM,
    borderBottom: `${tokens.strokeWidthThin} solid ${tokens.colorNeutralStroke2}`,
    overflowWrap: 'anywhere',
    whiteSpace: 'pre-wrap',
  },
  action: {
    ...mobileTouchTarget,
  },
  more: {
    ...mobileTouchTargetHeight,
  },
})
