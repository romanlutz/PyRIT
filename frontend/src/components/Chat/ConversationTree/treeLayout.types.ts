import type { LayoutNode, TreePosition } from './treeGraph'

export interface TreeLayoutRequest {
  readonly requestId: number
  readonly nodes: LayoutNode[]
}

export type TreeLayoutReply =
  | { readonly requestId: number; readonly positions: Array<[string, TreePosition]> }
  | { readonly requestId: number; readonly error: string }

export interface TreeLayoutWorkerPort {
  onmessage: ((event: MessageEvent<TreeLayoutReply>) => void) | null
  onerror: ((event: ErrorEvent) => void) | null
  postMessage: (request: TreeLayoutRequest) => void
  terminate: () => void
}
