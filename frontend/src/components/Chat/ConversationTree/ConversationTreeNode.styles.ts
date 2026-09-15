import { makeStyles, tokens } from '@fluentui/react-components'

import { mobileTouchTarget, mobileTouchTargetHeight } from '@/styles/touchTargets'

import { COMPACT_PIECE_COUNT, TREE_NODE_WIDTH, treeNodeHeight } from './treeGraph'

export const useConversationTreeNodeStyles = makeStyles({
  root: {
    width: `${TREE_NODE_WIDTH}px`,
    height: `${treeNodeHeight(COMPACT_PIECE_COUNT)}px`,
    boxSizing: 'border-box',
    display: 'flex',
    // React Flow disables pointer events on non-selectable, non-draggable wrappers.
    pointerEvents: 'auto',
    flexDirection: 'column',
    rowGap: tokens.spacingVerticalXS,
    padding: tokens.spacingHorizontalM,
    border: `${tokens.strokeWidthThin} solid ${tokens.colorNeutralStroke1}`,
    borderRadius: tokens.borderRadiusLarge,
    backgroundColor: tokens.colorNeutralBackground1,
    color: tokens.colorNeutralForeground1,
    boxShadow: tokens.shadow2,
    '&:focus-within': {
      outline: `${tokens.strokeWidthThick} solid ${tokens.colorStrokeFocus2}`,
      outlineOffset: tokens.strokeWidthThick,
    },
  },
  singlePiece: {
    height: `${treeNodeHeight(1)}px`,
  },
  twoPieces: {
    height: `${treeNodeHeight(2)}px`,
  },
  currentPath: {
    border: `${tokens.strokeWidthThin} solid ${tokens.colorBrandStroke1}`,
  },
  user: {
    backgroundColor: tokens.colorBrandBackground2,
    color: tokens.colorNeutralForeground1,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    columnGap: tokens.spacingHorizontalS,
    minWidth: 0,
  },
  title: {
    display: 'flex',
    flex: 1,
    alignItems: 'baseline',
    flexWrap: 'wrap',
    columnGap: tokens.spacingHorizontalS,
    minWidth: 0,
  },
  detail: {
    color: tokens.colorNeutralForeground2,
  },
  markers: {
    display: 'flex',
    flexWrap: 'wrap',
    columnGap: tokens.spacingHorizontalXS,
    minHeight: tokens.lineHeightBase200,
  },
  pieces: {
    display: 'flex',
    flex: 1,
    flexDirection: 'column',
    rowGap: tokens.spacingVerticalXS,
    minHeight: 0,
    overflow: 'hidden',
  },
  piece: {
    display: 'flex',
    alignItems: 'center',
    columnGap: tokens.spacingHorizontalS,
    flex: 1,
    minHeight: 0,
    maxHeight: '3.5rem',
    minWidth: 0,
  },
  pieceText: {
    display: '-webkit-box',
    WebkitLineClamp: 2,
    WebkitBoxOrient: 'vertical',
    overflow: 'hidden',
    overflowWrap: 'anywhere',
    whiteSpace: 'pre-wrap',
    minWidth: 0,
    flex: 1,
  },
  skeleton: {
    flex: 1,
    minWidth: 0,
  },
  button: {
    flexShrink: 0,
    ...mobileTouchTarget,
  },
  textButton: {
    justifyContent: 'flex-start',
    minWidth: 0,
    ...mobileTouchTargetHeight,
  },
  footer: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXS,
  },
  endpoint: {
    minWidth: 0,
    flex: 1,
    ...mobileTouchTargetHeight,
  },
  thumbnailContainer: {
    position: 'relative',
    flex: '0 0 2.75rem',
    width: '2.75rem',
    height: '2.75rem',
    overflow: 'hidden',
  },
  thumbnail: {
    width: '100%',
    height: '100%',
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
  error: {
    color: tokens.colorPaletteRedForeground1,
  },
  retryRow: {
    display: 'flex',
    alignItems: 'center',
    columnGap: tokens.spacingHorizontalXS,
    minWidth: 0,
  },
  handle: {
    visibility: 'hidden',
  },
})
