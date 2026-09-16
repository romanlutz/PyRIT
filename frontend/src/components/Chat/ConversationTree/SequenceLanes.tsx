import { Text } from '@fluentui/react-components'
import { ViewportPortal, useViewport } from '@xyflow/react'

import { useSequenceLanesStyles } from './SequenceLanes.styles'
import { visibleSequenceLanes, type SequenceLane } from './treeSequenceLanes'
import type { TreePaneSize } from './useTreeViewport'

interface SequenceLanesProps {
  readonly lanes: SequenceLane[]
  readonly size: TreePaneSize
}

/** Subscribe here, not in the data-owning pane, so moving lane decorations never schedules previews. */
export default function SequenceLanes({ lanes, size }: SequenceLanesProps) {
  const styles = useSequenceLanesStyles()
  const viewport = useViewport()
  const visible = visibleSequenceLanes(lanes, viewport, size)
  return (
    <ViewportPortal>
      <div className={styles.root} aria-hidden="true" data-testid="tree-sequence-lanes">
        {visible.map((lane: SequenceLane) => (
          <div
            key={lane.sequence}
            className={styles.lane}
            data-testid={`tree-sequence-lane-${lane.sequence}`}
            data-sequence={lane.sequence}
            style={{
              left: -viewport.x / viewport.zoom,
              top: lane.top,
              width: size.width / viewport.zoom,
              height: lane.bottom - lane.top,
            }}
          >
            <Text size={100} className={styles.label} style={{ transform: `scale(${1 / viewport.zoom})` }}>
              Sequence {lane.sequence}
            </Text>
          </div>
        ))}
      </div>
    </ViewportPortal>
  )
}
