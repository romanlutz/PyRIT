import { TREE_ROW_GAP, type LayoutNode, type TreePosition } from './treeGraph'

/** Route every connection into a row above its tallest message, not through a neighboring card. */
export function treeEdgeLanes(
  nodes: LayoutNode[],
  positions: ReadonlyMap<string, TreePosition>,
): Map<string, number> {
  const rowLanes = new Map<number, number>()
  for (const node of nodes) {
    const position = positions.get(node.id)
    if (!position) continue
    const lane = position.y - TREE_ROW_GAP / 2
    rowLanes.set(node.sequence, Math.min(rowLanes.get(node.sequence) ?? lane, lane))
  }
  const lanes = new Map<string, number>()
  for (const node of nodes) {
    const lane = rowLanes.get(node.sequence)
    if (positions.has(node.id) && lane !== undefined) lanes.set(node.id, lane)
  }
  return lanes
}

export function simplifyTreeRoute(points: TreePosition[]): TreePosition[] {
  const route: TreePosition[] = []
  for (const point of points) {
    const previous = route[route.length - 1]
    if (previous?.x === point.x && previous.y === point.y) continue
    const before = route[route.length - 2]
    if (before && previous
      && ((before.x === previous.x && previous.x === point.x)
        || (before.y === previous.y && previous.y === point.y))) route.pop()
    route.push(point)
  }
  return route
}

interface RoutingSpan {
  readonly points: TreePosition[]
  readonly index: number
}

interface RoutingGroup {
  readonly sourceX: number
  minTargetX: number
  maxTargetX: number
  readonly spans: RoutingSpan[]
}

/** Stagger overlapping horizontal runs without leaving the clear space between sequence bands. */
export function separateTreeRoutes(edges: Array<[string, TreePosition[]]>): Array<[string, TreePosition[]]> {
  const routes = edges.map(([id, points]: [string, TreePosition[]]): [string, TreePosition[]] => [id, [...points]])
  const gaps = new Map<number, Map<number, RoutingGroup>>()
  for (const [, points] of routes) {
    for (let index = 0; index < points.length - 1; index++) {
      const source = points[index], target = points[index + 1]
      if (source.y !== target.y || source.x === target.x) continue
      const gap = gaps.get(source.y) ?? new Map<number, RoutingGroup>()
      gaps.set(source.y, gap)
      const group = gap.get(source.x) ?? {
        sourceX: source.x, minTargetX: target.x, maxTargetX: target.x, spans: [],
      }
      group.minTargetX = Math.min(group.minTargetX, target.x)
      group.maxTargetX = Math.max(group.maxTargetX, target.x)
      group.spans.push({ points, index })
      gap.set(source.x, group)
    }
  }
  for (const [centerY, gap] of gaps) {
    const groups = [...gap.values()].sort((a: RoutingGroup, b: RoutingGroup) => a.sourceX - b.sourceX)
    const next = groups.map((): number[] => [])
    const incoming = groups.map(() => 0)
    const levels = groups.map(() => 0)
    for (let left = 0; left < groups.length; left++) {
      for (let right = left + 1; right < groups.length; right++) {
        const goesRight = groups[left].maxTargetX >= groups[right].sourceX
        const goesLeft = groups[right].minTargetX <= groups[left].sourceX
        if (!goesRight && !goesLeft) continue
        const before = goesRight ? right : left
        const after = goesRight ? left : right
        next[before].push(after)
        incoming[after]++
      }
    }
    const ready = incoming.flatMap((count: number, index: number) => count === 0 ? [index] : [])
    for (const index of ready) {
      for (const after of next[index]) {
        levels[after] = Math.max(levels[after], levels[index] + 1)
        incoming[after]--
        if (incoming[after] === 0) ready.push(after)
      }
    }
    if (ready.length !== groups.length) throw new Error('Conversation connections have conflicting sequence-row order.')
    const levelCount = Math.max(...levels) + 1
    for (const [index, group] of groups.entries()) {
      const y = centerY - TREE_ROW_GAP / 4 + (TREE_ROW_GAP / 2) * (levels[index] + 1) / (levelCount + 1)
      for (const span of group.spans) {
        span.points[span.index] = { ...span.points[span.index], y }
        span.points[span.index + 1] = { ...span.points[span.index + 1], y }
      }
    }
  }
  return routes.map(([id, points]: [string, TreePosition[]]) => [id, simplifyTreeRoute(points)])
}

const EDGE_CORNER_RADIUS = TREE_ROW_GAP / 8

export function treeEdgePath(points: TreePosition[]): string {
  const route = simplifyTreeRoute(points)
  if (route.length < 2) throw new Error('A conversation connection needs at least two distinct points.')
  let path = `M ${route[0].x} ${route[0].y}`
  for (let index = 1; index < route.length - 1; index++) {
    const before = route[index - 1], point = route[index], after = route[index + 1]
    const incomingLength = Math.hypot(point.x - before.x, point.y - before.y)
    const outgoingLength = Math.hypot(after.x - point.x, after.y - point.y)
    const radius = Math.min(EDGE_CORNER_RADIUS, incomingLength / 2, outgoingLength / 2)
    const entry = {
      x: point.x - (point.x - before.x) / incomingLength * radius,
      y: point.y - (point.y - before.y) / incomingLength * radius,
    }
    const exit = {
      x: point.x + (after.x - point.x) / outgoingLength * radius,
      y: point.y + (after.y - point.y) / outgoingLength * radius,
    }
    path += ` L ${entry.x} ${entry.y} Q ${point.x} ${point.y} ${exit.x} ${exit.y}`
  }
  const end = route[route.length - 1]
  return `${path} L ${end.x} ${end.y}`
}
