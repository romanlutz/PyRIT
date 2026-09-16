import { makeStyles, tokens } from '@fluentui/react-components'

export const useSequenceLanesStyles = makeStyles({
  root: {
    position: 'absolute',
    top: 0,
    left: 0,
    pointerEvents: 'none',
    userSelect: 'none',
    zIndex: -1,
  },
  lane: {
    position: 'absolute',
    boxSizing: 'border-box',
    borderTopStyle: 'dashed',
    borderTopWidth: tokens.strokeWidthThin,
    borderTopColor: tokens.colorNeutralStroke2,
    borderBottomStyle: 'dashed',
    borderBottomWidth: tokens.strokeWidthThin,
    borderBottomColor: tokens.colorNeutralStroke2,
    pointerEvents: 'none',
  },
  label: {
    position: 'absolute',
    top: tokens.spacingVerticalXXS,
    left: tokens.spacingHorizontalS,
    color: tokens.colorNeutralForeground3,
    whiteSpace: 'nowrap',
    transformOrigin: 'top left',
  },
})
