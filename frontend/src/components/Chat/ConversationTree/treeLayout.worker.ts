import { layoutTree } from './treeLayout'
import type { TreeLayoutReply, TreeLayoutRequest } from './treeLayout.types'

self.onmessage = (event: MessageEvent<TreeLayoutRequest>): void => {
  const { requestId, nodes } = event.data
  let reply: TreeLayoutReply
  try {
    reply = { requestId, positions: layoutTree(nodes) }
  } catch (error: unknown) {
    reply = { requestId, error: error instanceof Error ? error.message : 'Unable to arrange this tree.' }
  }
  self.postMessage(reply)
}
