import type { ConversationView } from '@/types'

const SCENARIO_RESULT_ID_QUERY_KEY = 'scenarioResultId'
const CONVERSATION_VIEW_QUERY_KEY = 'view'
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

/**
 * Returns the original value represented by a React Router path parameter.
 *
 * React Router already decodes each path segment, but re-escapes decoded
 * slashes as `%2F` so they remain inside one parameter. Undo only that
 * re-escaping; calling `decodeURIComponent` again would corrupt literal `%`
 * sequences and can throw for malformed user-entered URLs.
 */
export function routerPathParamValue(value: string | undefined): string {
  return (value ?? '').replace(/%2F/gi, '/')
}

/** Returns one validated scenario-run provenance UUID from a route query. */
export function scenarioRunProvenance(searchParams: URLSearchParams): string | null {
  const values = searchParams.getAll(SCENARIO_RESULT_ID_QUERY_KEY)
  if (values.length !== 1 || !UUID_PATTERN.test(values[0])) {
    return null
  }
  return values[0]
}

export function conversationViewFromSearchParams(searchParams: URLSearchParams): ConversationView {
  const values = searchParams.getAll(CONVERSATION_VIEW_QUERY_KEY)
  return values.length === 1 && values[0] === 'tree' ? 'tree' : 'chat'
}

/** Change only the presentation mode, preserving provenance and other query parameters. */
export function searchParamsWithConversationView(
  searchParams: URLSearchParams,
  view: ConversationView,
): URLSearchParams {
  const next = new URLSearchParams(searchParams)
  if (view === 'tree') {
    next.set(CONVERSATION_VIEW_QUERY_KEY, view)
  } else {
    next.delete(CONVERSATION_VIEW_QUERY_KEY)
  }
  return next
}

/** Builds an attack-detail route with optional provenance and presentation mode. */
export function attackRoutePath(
  attackResultId: string,
  scenarioResultId?: string | null,
  view: ConversationView = 'chat',
): string {
  return appendAttackSearchParams(
    `/attacks/${encodeURIComponent(attackResultId)}`,
    scenarioResultId,
    view,
  )
}

/** Builds an attack-conversation route with optional bounded scenario-run provenance. */
export function attackConversationRoutePath(
  attackResultId: string,
  conversationId: string,
  scenarioResultId?: string | null,
  view: ConversationView = 'chat',
): string {
  return appendAttackSearchParams(
    `/attacks/${encodeURIComponent(attackResultId)}/conversations/${encodeURIComponent(conversationId)}`,
    scenarioResultId,
    view,
  )
}

/** Builds the route for one scenario run. Callers must pass a trusted persisted ID. */
export function scenarioRunRoutePath(scenarioResultId: string): string {
  return `/scanner-history/${encodeURIComponent(scenarioResultId)}`
}

/** Builds the route for one attack result within a scenario run. */
export function scenarioRunAttackRoutePath(scenarioResultId: string, attackResultId: string): string {
  return `${scenarioRunRoutePath(scenarioResultId)}/${encodeURIComponent(attackResultId)}`
}

function appendAttackSearchParams(
  path: string,
  scenarioResultId: string | null | undefined,
  view: ConversationView,
): string {
  const searchParams = new URLSearchParams()
  if (scenarioResultId && UUID_PATTERN.test(scenarioResultId)) {
    searchParams.set(SCENARIO_RESULT_ID_QUERY_KEY, scenarioResultId)
  }
  const query = searchParamsWithConversationView(searchParams, view).toString()
  return query ? `${path}?${query}` : path
}
