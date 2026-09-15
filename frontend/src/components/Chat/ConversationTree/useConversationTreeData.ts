import { useEffect, useState, useSyncExternalStore } from 'react'

import { TreeReadCoordinator } from './treeReadCoordinator'

export function useConversationTreeData(
  attackResultId: string,
  activeConversationId: string | null,
  active: boolean,
  refreshKey: number,
) {
  const [reads] = useState(() => new TreeReadCoordinator(attackResultId, activeConversationId, refreshKey))
  const snapshot = useSyncExternalStore(reads.subscribe, reads.getSnapshot)

  useEffect(() => {
    reads.setActiveConversation(activeConversationId)
    reads.setActive(active)
    reads.refresh(refreshKey)
  }, [reads, activeConversationId, active, refreshKey])

  useEffect(() => () => { reads.setActive(false) }, [reads])

  return { reads, snapshot }
}
