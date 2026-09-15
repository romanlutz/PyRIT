import { makeStyles, tokens } from '@fluentui/react-components'

import { mobileTouchTarget, mobileTouchTargetHeight } from '@/styles/touchTargets'

export const useConversationChooserStyles = makeStyles({
  surface: {
    width: 'min(42rem, calc(100vw - 2rem))',
    maxWidth: 'calc(100vw - 2rem)',
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    rowGap: tokens.spacingVerticalM,
    minWidth: 0,
    maxHeight: '65dvh',
  },
  results: {
    minHeight: 0,
    overflowY: 'auto',
  },
  list: {
    listStyleType: 'none',
    padding: 0,
    margin: 0,
  },
  row: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'stretch',
    textAlign: 'left',
    width: '100%',
    padding: tokens.spacingHorizontalM,
    rowGap: tokens.spacingVerticalXS,
    borderBottom: `${tokens.strokeWidthThin} solid ${tokens.colorNeutralStroke2}`,
    borderRadius: tokens.borderRadiusNone,
    ...mobileTouchTargetHeight,
  },
  heading: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalS,
  },
  id: {
    fontFamily: tokens.fontFamilyMonospace,
  },
  context: {
    display: '-webkit-box',
    WebkitBoxOrient: 'vertical',
    WebkitLineClamp: 3,
    overflow: 'hidden',
    overflowWrap: 'anywhere',
    color: tokens.colorNeutralForeground2,
  },
  action: {
    ...mobileTouchTarget,
  },
  more: {
    width: '100%',
    ...mobileTouchTargetHeight,
  },
})
