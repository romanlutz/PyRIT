import type { LayoutNode, TreePosition } from './treeGraph'

export interface TreeLayoutRequest {
  readonly requestId: number
  readonly nodes: LayoutNode[]
}

export interface TreeLayoutResult {
  readonly positions: Array<[string, TreePosition]>
  readonly edgeRoutes: Array<[string, TreePosition[]]>
}

export type TreeLayoutReply =
  | (TreeLayoutResult & { readonly requestId: number })
  | { readonly requestId: number; readonly error: string }

export interface TreeLayoutWorkerPort {
  onmessage: ((event: MessageEvent<TreeLayoutReply>) => void) | null
  onerror: ((event: ErrorEvent) => void) | null
  postMessage: (request: TreeLayoutRequest) => void
  terminate: () => void
}
