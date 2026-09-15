import { makeStyles, tokens } from '@fluentui/react-components'

import { NARROW_VIEWPORT_QUERY, mobileTouchTarget, mobileTouchTargetHeight } from '@/styles/touchTargets'

export const useConversationTreeStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    flex: 1,
    width: '100%',
    height: '100%',
    minWidth: 0,
    minHeight: 0,
    backgroundColor: tokens.colorNeutralBackground2,
    color: tokens.colorNeutralForeground1,
  },
  inactive: {
    display: 'none',
  },
  toolbar: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: tokens.spacingHorizontalM,
    padding: tokens.spacingHorizontalM,
    borderBottom: `${tokens.strokeWidthThin} solid ${tokens.colorNeutralStroke2}`,
    backgroundColor: tokens.colorNeutralBackground1,
    [NARROW_VIEWPORT_QUERY]: {
      columnGap: tokens.spacingHorizontalXS,
      padding: tokens.spacingHorizontalS,
    },
  },
  controls: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    columnGap: tokens.spacingHorizontalXS,
  },
  status: {
    padding: `${tokens.spacingVerticalXS} ${tokens.spacingHorizontalM}`,
    color: tokens.colorNeutralForeground2,
    fontSize: tokens.fontSizeBase200,
  },
  pane: {
    position: 'relative',
    flex: 1,
    minHeight: '16rem',
    minWidth: 0,
    '& .react-flow__pane': {
      backgroundColor: tokens.colorNeutralBackground2,
    },
    '& .react-flow__edge-path': {
      stroke: tokens.colorNeutralStroke1,
    },
    '& .react-flow__attribution': {
      backgroundColor: tokens.colorNeutralBackground1,
      color: tokens.colorNeutralForeground2,
    },
  },
  pathEdge: {
    '& .react-flow__edge-path': {
      stroke: tokens.colorBrandStroke1,
      strokeWidth: tokens.strokeWidthThick,
    },
  },
  empty: {
    padding: tokens.spacingHorizontalXL,
  },
  emptyConversations: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
    padding: `${tokens.spacingVerticalXS} ${tokens.spacingHorizontalM}`,
    backgroundColor: tokens.colorNeutralBackground1,
  },
  button: {
    ...mobileTouchTarget,
  },
  textButton: {
    ...mobileTouchTargetHeight,
  },
})
