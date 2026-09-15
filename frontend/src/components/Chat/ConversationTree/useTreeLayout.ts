import { useEffect, useState, useSyncExternalStore } from 'react'

import type { LayoutNode } from './treeGraph'
import LayoutWorker from './treeLayout.worker?worker'
import { TreeLayoutCoordinator } from './treeLayoutCoordinator'

export function useTreeLayout(nodes: LayoutNode[], anchorId: string | null, active: boolean) {
  const [layout] = useState(() => new TreeLayoutCoordinator(() => new LayoutWorker()))
  const snapshot = useSyncExternalStore(layout.subscribe, layout.getSnapshot)

  useEffect(() => { layout.setGraph(nodes, anchorId) }, [layout, nodes, anchorId])
  useEffect(() => {
    layout.setActive(active)
    return () => { layout.setActive(false) }
  }, [layout, active])

  return { ...snapshot, retry: layout.retry }
}
