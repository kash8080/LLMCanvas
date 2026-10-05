// Shape inference over the part graph (PLAN.md §2.2).
//
// Nodes are evaluated in topological order. For each node:
//   - an unconnected input            → status 'error' ("Input 'q' is not connected")
//   - an input from a node w/o shape   → status 'unknown' (downstream of an error: grey, not red)
//   - otherwise the def's `infer` runs → 'ok' or 'error' with readable messages
// Nodes on a cycle get an error; nodes only downstream of a cycle are 'unknown'.
//
// The engine works on a flat node list. Phase 3 groups are meant to stay flat too: a group is a
// container (`parentId`), and its outer ports are bridged by ordinary pass-through proxy nodes
// (see docs/PROGRESS.md, Session 3).
import { resolveParams, validateParams } from './resolve'
import { NO_TYING, type TyingCheck } from './tying'
import type { GraphModel, Hyperparams, InferenceResult, NodeDef, NodeResult, Shape } from './types'

const EMPTY_COUNT = { total: 0, tensors: [] }

/**
 * `tying` (engine/tying.ts, from `checkTying`): LM heads that share the Embedding's weight get a param
 * count of 0 (with `tied` set); tying problems become errors on those parts.
 */
export function inferShapes(graph: GraphModel, hp: Hyperparams, defs: Record<string, NodeDef>, tying: TyingCheck = NO_TYING): InferenceResult {
  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]))
  // Only edges between known nodes take part.
  const edges = graph.edges.filter((e) => nodeById.has(e.source) && nodeById.has(e.target))

  // Topological order (Kahn).
  const indegree = new Map(graph.nodes.map((n) => [n.id, 0]))
  const successors = new Map<string, string[]>(graph.nodes.map((n) => [n.id, []]))
  for (const e of edges) {
    indegree.set(e.target, indegree.get(e.target)! + 1)
    successors.get(e.source)!.push(e.target)
  }
  const queue = graph.nodes.filter((n) => indegree.get(n.id) === 0).map((n) => n.id)
  const order: string[] = []
  while (queue.length > 0) {
    const id = queue.shift()!
    order.push(id)
    for (const next of successors.get(id)!) {
      indegree.set(next, indegree.get(next)! - 1)
      if (indegree.get(next) === 0) queue.push(next)
    }
  }
  const unordered = graph.nodes.map((n) => n.id).filter((id) => !order.includes(id))
  const onCycle = new Set(unordered.filter((id) => reachesItself(id, successors)))

  const results: Record<string, NodeResult> = {}

  for (const id of [...order, ...unordered]) {
    const node = nodeById.get(id)!
    const def = defs[node.type]
    if (!def) {
      results[id] = {
        status: 'error',
        errors: [`Unknown part type "${node.type}"`],
        inputShapes: [],
        outputShapes: [],
        resolved: {},
        paramCount: EMPTY_COUNT,
      }
      continue
    }

    const resolved = resolveParams(def, node.params, hp)
    const own = def.paramCount(resolved, hp)
    const tiedTo = tying.tied[id]
    const paramCount = tiedTo ? { ...own, total: 0, tied: { to: tiedTo, params: own.total } } : own
    const errors: string[] = [...validateParams(def, node.params), ...(tying.errors[id] ?? [])]
    const inputShapes: (Shape | null)[] = []
    let upstreamUnknown = false

    for (const port of def.inputs) {
      const incoming = edges.filter((e) => e.target === id && e.targetHandle === port.id)
      if (incoming.length === 0) {
        errors.push(`Input '${port.label}' is not connected`)
        inputShapes.push(null)
        continue
      }
      if (incoming.length > 1) errors.push(`Input '${port.label}' has ${incoming.length} connections (only one allowed)`)
      const shape = outputShapeOf(incoming[0].source, incoming[0].sourceHandle, results, defs, nodeById)
      if (!shape) upstreamUnknown = true
      inputShapes.push(shape)
    }

    if (onCycle.has(id)) errors.unshift('Part of a cycle: data must flow one way (top → bottom)')

    const base = { inputShapes, resolved, paramCount }
    if (errors.length > 0) {
      results[id] = { ...base, status: 'error', errors, outputShapes: def.outputs.map(() => null) }
    } else if (upstreamUnknown || unordered.includes(id)) {
      results[id] = { ...base, status: 'unknown', errors: [], outputShapes: def.outputs.map(() => null) }
    } else {
      let r
      try {
        r = def.infer({ inputs: inputShapes as Shape[], p: resolved, hp })
      } catch (err) {
        r = { outputs: [], errors: [`Internal error: ${err instanceof Error ? err.message : String(err)}`] }
      }
      results[id] =
        r.errors.length > 0
          ? { ...base, status: 'error', errors: r.errors, outputShapes: def.outputs.map(() => null) }
          : { ...base, status: 'ok', errors: [], outputShapes: def.outputs.map((_, i) => r.outputs[i] ?? null) }
    }
  }

  const edgeShapes: Record<string, Shape | null> = {}
  for (const e of graph.edges) edgeShapes[e.id] = outputShapeOf(e.source, e.sourceHandle, results, defs, nodeById)

  return { nodes: results, edges: edgeShapes }
}

function outputShapeOf(
  nodeId: string,
  handle: string | null,
  results: Record<string, NodeResult>,
  defs: Record<string, NodeDef>,
  nodeById: Map<string, { type: string }>,
): Shape | null {
  const r = results[nodeId]
  const def = defs[nodeById.get(nodeId)?.type ?? '']
  if (!r || !def) return null
  const idx = handle == null ? 0 : def.outputs.findIndex((o) => o.id === handle)
  return idx >= 0 ? (r.outputShapes[idx] ?? null) : null
}

function reachesItself(start: string, successors: Map<string, string[]>): boolean {
  const seen = new Set<string>()
  const stack = [...(successors.get(start) ?? [])]
  while (stack.length > 0) {
    const id = stack.pop()!
    if (id === start) return true
    if (seen.has(id)) continue
    seen.add(id)
    stack.push(...(successors.get(id) ?? []))
  }
  return false
}

/** Would adding source → target create a cycle? (i.e. can target already reach source?) */
export function wouldCreateCycle(edges: { source: string; target: string }[], source: string, target: string): boolean {
  if (source === target) return true
  const successors = new Map<string, string[]>()
  for (const e of edges) {
    if (!successors.has(e.source)) successors.set(e.source, [])
    successors.get(e.source)!.push(e.target)
  }
  const seen = new Set<string>()
  const stack = [target]
  while (stack.length > 0) {
    const id = stack.pop()!
    if (id === source) return true
    if (seen.has(id)) continue
    seen.add(id)
    stack.push(...(successors.get(id) ?? []))
  }
  return false
}
