import { makeStyles, tokens } from '@fluentui/react-components'

export const useScenarioRunEstimateStyles = makeStyles({
  summary: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: tokens.spacingVerticalXXS,
    minWidth: 0,
  },
  summaryHeader: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalXS,
  },
  total: {
    color: tokens.colorNeutralForeground1,
  },
  muted: {
    color: tokens.colorNeutralForeground3,
  },
  details: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
    minWidth: 0,
  },
  termGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
  },
  termList: {
    display: 'grid',
    gap: tokens.spacingVerticalXS,
    margin: 0,
  },
  termRow: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto',
    columnGap: tokens.spacingHorizontalM,
    alignItems: 'baseline',
  },
  termLabel: {
    minWidth: 0,
    overflowWrap: 'anywhere',
  },
  termValue: {
    margin: 0,
    fontWeight: tokens.fontWeightSemibold,
    fontVariantNumeric: 'tabular-nums',
  },
  termDetail: {
    gridColumn: '1 / -1',
    margin: 0,
    color: tokens.colorNeutralForeground3,
  },
  formula: {
    display: 'block',
    padding: `${tokens.spacingVerticalXS} ${tokens.spacingHorizontalS}`,
    overflowWrap: 'anywhere',
    fontFamily: tokens.fontFamilyMonospace,
    fontSize: tokens.fontSizeBase200,
    backgroundColor: tokens.colorNeutralBackground3,
    borderRadius: tokens.borderRadiusSmall,
  },
  caveat: {
    color: tokens.colorNeutralForeground3,
  },
})
