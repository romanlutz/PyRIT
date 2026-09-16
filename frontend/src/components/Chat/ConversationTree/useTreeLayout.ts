import { useEffect, useMemo, useState, useSyncExternalStore } from 'react'

import type { LayoutNode } from './treeGraph'
import LayoutWorker from './treeLayout.worker?worker'
import { TreeLayoutCoordinator } from './treeLayoutCoordinator'

const UNARRANGED_NODE_IDS: ReadonlySet<string> = new Set()

export function useTreeLayout(nodes: LayoutNode[], anchorId: string | null, active: boolean) {
  const [layout] = useState(() => new TreeLayoutCoordinator(() => new LayoutWorker()))
  const snapshot = useSyncExternalStore(layout.subscribe, layout.getSnapshot)
  const graphSignature = useMemo(() => JSON.stringify(nodes), [nodes])

  useEffect(() => { layout.setGraph(nodes, anchorId) }, [layout, nodes, anchorId])
  useEffect(() => {
    layout.setActive(active)
    return () => { layout.setActive(false) }
  }, [layout, active])

  return {
    ...snapshot,
    // A topology render can precede setGraph's effect; never draw old routes over its new rows.
    arrangedNodeIds: graphSignature === snapshot.graphSignature ? snapshot.arrangedNodeIds : UNARRANGED_NODE_IDS,
    retry: layout.retry,
  }
}
