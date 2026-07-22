import { makeStyles, tokens } from '@fluentui/react-components'

export const useTourTooltipStyles = makeStyles({
  wrapper: {
    boxSizing: 'border-box',
    display: 'flex',
    flexDirection: 'column',
    width: '420px',
    maxWidth: `calc(100vw - ${tokens.spacingHorizontalXXL})`,
    position: 'relative',
  },
  container: {
    boxSizing: 'border-box',
    backgroundColor: tokens.colorNeutralBackground1,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    borderRadius: tokens.borderRadiusLarge,
    boxShadow: tokens.shadow16,
    padding: tokens.spacingHorizontalL,
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
    minWidth: 0,
  },
  closeRow: {
    display: 'flex',
    justifyContent: 'flex-end',
    marginBottom: `calc(${tokens.spacingVerticalS} * -1)`,
    marginTop: `calc(${tokens.spacingVerticalXS} * -1)`,
  },
  mascot: {
    width: '72px',
    height: '72px',
    flexShrink: 0,
    objectFit: 'contain',
    pointerEvents: 'none',
  },
  content: {
    color: tokens.colorNeutralForeground1,
    lineHeight: tokens.lineHeightBase300,
  },
  footer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalM,
    minWidth: 0,
  },
  stepCounter: {
    color: tokens.colorNeutralForeground3,
    whiteSpace: 'nowrap',
    marginRight: 'auto',
  },
  actions: {
    display: 'flex',
    flexWrap: 'wrap',
    justifyContent: 'flex-end',
    gap: tokens.spacingHorizontalS,
    marginLeft: 'auto',
    minWidth: 0,
  },
})
