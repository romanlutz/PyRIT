import { makeStyles, tokens } from '@fluentui/react-components'

import { NARROW_VIEWPORT_QUERY, mobileTouchTarget } from '@/styles/touchTargets'

export const useMediaLightboxStyles = makeStyles({
  surface: {
    width: 'min(72rem, calc(100vw - 2rem))',
    maxWidth: 'calc(100vw - 2rem)',
    maxHeight: 'calc(100dvh - 2rem)',
    [NARROW_VIEWPORT_QUERY]: {
      width: 'calc(100vw - 1rem)',
      maxWidth: 'calc(100vw - 1rem)',
      padding: tokens.spacingHorizontalM,
    },
  },
  body: {
    minHeight: 0,
    minWidth: 0,
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    rowGap: tokens.spacingVerticalM,
    minWidth: 0,
    maxHeight: '70dvh',
    overflowY: 'auto',
  },
  title: {
    overflowWrap: 'anywhere',
  },
  imageContainer: {
    position: 'relative',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: '8rem',
    width: '100%',
  },
  image: {
    display: 'block',
    width: 'auto',
    height: 'auto',
    maxWidth: '100%',
    maxHeight: '64dvh',
    objectFit: 'contain',
  },
  hiddenImage: {
    visibility: 'hidden',
    width: 0,
    height: 0,
  },
  spinner: {
    position: 'absolute',
    inset: 0,
  },
  player: {
    display: 'block',
    width: '100%',
    maxHeight: '64dvh',
  },
  action: {
    ...mobileTouchTarget,
  },
})
