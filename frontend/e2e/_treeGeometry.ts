import { expect } from '@playwright/test'
import type { Page } from '@playwright/test'

/** Inspect actual rendered geometry, including SVG routing and card dimensions. */
export async function expectClearTreeGeometry(page: Page): Promise<void> {
  await expect(page.getByRole('status').filter({ hasText: /All \d+ conversations loaded/ })).toBeVisible()
  await expect(page.getByTestId('conversation-tree')).toHaveAttribute('data-layout-pending', 'false')
  await page.getByRole('button', { name: 'Fit to view' }).click()
  const geometry = await page.getByTestId('conversation-tree').evaluate((tree: HTMLElement) => {
    interface Point { x: number; y: number }
    type Segment = readonly [Point, Point]
    const epsilon = 0.1
    const nodes = [...tree.querySelectorAll<HTMLElement>('.react-flow__node')].map((element: HTMLElement) => {
      const card = element.querySelector('article')
      if (!card) throw new Error('A graph node has no message card')
      const box = card.getBoundingClientRect()
      return { id: element.dataset.id ?? '', x: box.x, y: box.y, width: box.width, height: box.height }
    })
    const edges = [...tree.querySelectorAll<SVGPathElement>('.react-flow__edge-path')].map((element: SVGPathElement) => {
      const id = element.closest<SVGGElement>('.react-flow__edge')?.dataset.id
      if (!id) throw new Error('A connection has no source/target identity')
      const [source, target] = id.split(':')
      const matrix = element.getScreenCTM()
      if (!matrix) throw new Error('A connection has no screen transform')
      const length = element.getTotalLength()
      const points: Point[] = []
      for (let distance = 0; distance < length; distance += 8) {
        const point = element.getPointAtLength(distance).matrixTransform(matrix)
        points.push({ x: point.x, y: point.y })
      }
      const end = element.getPointAtLength(length).matrixTransform(matrix)
      points.push({ x: end.x, y: end.y })
      const corners = points.filter((point: Point, index: number) => {
        if (index === 0 || index === points.length - 1) return true
        const before = points[index - 1], after = points[index + 1]
        return Math.abs((point.x - before.x) * (after.y - point.y)
          - (point.y - before.y) * (after.x - point.x)) > epsilon
      })
      return {
        id, source, target, points,
        segments: corners.slice(1).map((point: Point, index: number): Segment => [corners[index], point]),
      }
    })
    const overlaps: string[] = []
    const blocked: string[] = []
    const crossings: string[] = []
    const unevenSiblings: string[] = []
    const cross = (a: Point, b: Point): number => a.x * b.y - a.y * b.x
    const subtract = (a: Point, b: Point): Point => ({ x: a.x - b.x, y: a.y - b.y })
    const intersects = ([a, b]: Segment, [c, d]: Segment): boolean => {
      const first = subtract(b, a), second = subtract(d, c), offset = subtract(c, a)
      const denominator = cross(first, second)
      if (Math.abs(denominator) <= epsilon) {
        if (Math.abs(cross(offset, first)) > epsilon) return false
        const axis = Math.abs(first.x) > Math.abs(first.y) ? 'x' : 'y'
        return Math.max(Math.min(a[axis], b[axis]), Math.min(c[axis], d[axis]))
          < Math.min(Math.max(a[axis], b[axis]), Math.max(c[axis], d[axis])) - epsilon
      }
      const t = cross(offset, second) / denominator
      const u = cross(offset, first) / denominator
      return t > epsilon && t < 1 - epsilon && u > epsilon && u < 1 - epsilon
    }
    for (let index = 0; index < nodes.length; index++) {
      const a = nodes[index]
      for (const b of nodes.slice(index + 1)) {
        if (a.x < b.x + b.width - epsilon && a.x + a.width > b.x + epsilon
          && a.y < b.y + b.height - epsilon && a.y + a.height > b.y + epsilon) {
          overlaps.push(`${a.id}/${b.id}`)
        }
      }
    }
    for (const edge of edges) {
      for (const node of nodes) {
        if (node.id === edge.source || node.id === edge.target) continue
        if (edge.points.some((point: Point) =>
          point.x > node.x + epsilon && point.x < node.x + node.width - epsilon
          && point.y > node.y + epsilon && point.y < node.y + node.height - epsilon,
        )) blocked.push(`${edge.id}/${node.id}`)
      }
    }
    const byId = new Map(nodes.map((node) => [node.id, node]))
    for (let index = 0; index < edges.length; index++) {
      const first = edges[index]
      for (const second of edges.slice(index + 1)) {
        if (first.source === second.source) {
          const a = byId.get(first.target), b = byId.get(second.target)
          if (a && b && Math.abs(a.y + a.height / 2 - b.y - b.height / 2) > epsilon) {
            unevenSiblings.push(`${first.target}/${second.target}`)
          }
          continue
        }
        if (first.segments.some((segment: Segment) =>
          second.segments.some((other: Segment) => intersects(segment, other)),
        )) crossings.push(`${first.id}/${second.id}`)
      }
    }
    return { nodeCount: nodes.length, edgeCount: edges.length, overlaps, blocked, crossings, unevenSiblings }
  })
  expect(geometry.nodeCount).toBeGreaterThan(2)
  expect(geometry.edgeCount).toBe(geometry.nodeCount - 1)
  expect(geometry.overlaps).toEqual([])
  expect(geometry.blocked).toEqual([])
  expect(geometry.crossings).toEqual([])
  expect(geometry.unevenSiblings).toEqual([])
}
