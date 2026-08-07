import { makeStyles, tokens } from '@fluentui/react-components'

export const useParameterFieldStyles = makeStyles({
  checkboxGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXXS,
  },
  fieldHint: {
    color: tokens.colorNeutralForeground3,
    marginTop: tokens.spacingVerticalXXS,
  },
})
