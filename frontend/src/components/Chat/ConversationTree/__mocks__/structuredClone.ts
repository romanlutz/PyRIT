/// <reference types="node" />

import { deserialize, serialize } from 'node:v8'

/** jsdom lacks the native structuredClone used by Dagre 3; preserve Maps and Sets in tests. */
export function installStructuredClone(): void {
  if (typeof globalThis.structuredClone === 'function') return
  Object.defineProperty(globalThis, 'structuredClone', {
    configurable: true,
    writable: true,
    value: (value: unknown): unknown => deserialize(serialize(value)),
  })
}
