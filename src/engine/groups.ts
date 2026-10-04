// Groups (Transformer Block / MHA / SwiGLU) stay invisible to the engine (PLAN.md §2.3).
//
// A group is a container: its children carry `parentId = <group id>`. Inside every group sit two
// pass-through proxy parts: `group_input` (bridges the group's outer input port) and
// `group_output` (bridges its outer output port). Canvas edges attach to the group's outer ports
// (the group's own id + handle 'in' / 'out'); `flattenGroups` redirects those edges onto the
// proxies so `inferShapes` only ever sees a flat graph of parts.
import type { GraphEdge, GraphModel } from './types'

export const GROUP_INPUT = 'group_input'
export const GROUP_OUTPUT = 'group_output'

export function isProxyType(type: string): boolean {
  return type === GROUP_INPUT || type === GROUP_OUTPUT
}

/**
 * Redirect edges that touch a group's outer ports:
 *   … → group (in)    becomes  … → group_input proxy (in)
 *   group (out) → …   becomes  group_output proxy (out) → …
 * Edge ids are kept, so edge shapes from inference still map to the canvas edges.
 * Group ids are not part nodes, so a group without a proxy just leaves a dangling edge
 * (which inference ignores).
 */
export function flattenGroups(graph: GraphModel): GraphModel {
  const inProxy = new Map<string, string>()
  const outProxy = new Map<string, string>()
  for (const n of graph.nodes) {
    if (!n.parentId) continue
    if (n.type === GROUP_INPUT) inProxy.set(n.parentId, n.id)
    else if (n.type === GROUP_OUTPUT) outProxy.set(n.parentId, n.id)
  }
  if (inProxy.size === 0 && outProxy.size === 0) return graph

  const edges = graph.edges.map((e): GraphEdge => {
    const source = outProxy.get(e.source)
    const target = inProxy.get(e.target)
    if (!source && !target) return e
    return {
      ...e,
      ...(source ? { source, sourceHandle: 'out' } : {}),
      ...(target ? { target, targetHandle: 'in' } : {}),
    }
  })
  return { nodes: graph.nodes, edges }
}
