import dagre from '@dagrejs/dagre'

import {
  TREE_COLUMN_GAP,
  TREE_NODE_HEIGHT,
  TREE_NODE_WIDTH,
  TREE_ROW_GAP,
  type LayoutNode,
  type TreePosition,
} from './treeGraph'

export function layoutTree(nodes: LayoutNode[]): Array<[string, TreePosition]> {
  const graph = new dagre.graphlib.Graph()
    .setGraph({ rankdir: 'TB', nodesep: TREE_COLUMN_GAP, ranksep: TREE_ROW_GAP, ranker: 'longest-path' })
    .setDefaultEdgeLabel(() => ({}))
  for (const node of nodes) {
    graph.setNode(node.id, { width: TREE_NODE_WIDTH, height: node.height ?? TREE_NODE_HEIGHT })
  }
  for (const node of nodes) {
    if (node.parentId && graph.hasNode(node.parentId)) graph.setEdge(node.parentId, node.id)
  }
  dagre.layout(graph)
  return nodes.map((node: LayoutNode) => {
    const position = graph.node(node.id)
    return [node.id, { x: position.x - TREE_NODE_WIDTH / 2, y: position.y - (node.height ?? TREE_NODE_HEIGHT) / 2 }]
  })
}
