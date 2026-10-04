// Connection rules (PLAN.md §2.4): output → input only, one edge per input (new edge replaces
// the old one), no cycles. Pure functions so Canvas and tests can share them.
import { wouldCreateCycle } from '../engine/infer'
import { getNodeDef } from '../nodes/registry'
import type { AppEdge, AppNode } from './types'

export interface ConnectionLike {
  source: string
  sourceHandle?: string | null
  target: string
  targetHandle?: string | null
}

/** null if the connection is allowed, otherwise a short human-readable reason. */
export function connectionProblem(nodes: AppNode[], edges: AppEdge[], c: ConnectionLike): string | null {
  if (c.source === c.target) return "A part can't connect to itself."
  const src = nodes.find((n) => n.id === c.source)
  const tgt = nodes.find((n) => n.id === c.target)
  if (src?.type !== 'part' || tgt?.type !== 'part') return 'Only model parts have ports.'
  const sdef = getNodeDef(src.data.partType)
  const tdef = getNodeDef(tgt.data.partType)
  if (!sdef?.outputs.some((o) => o.id === c.sourceHandle) || !tdef?.inputs.some((i) => i.id === c.targetHandle))
    return 'Connect an output (green, bottom) to an input (blue, top).'
  // The edge currently on this input would be replaced, so leave it out of the cycle check.
  const others = edges.filter((e) => !(e.target === c.target && e.targetHandle === c.targetHandle))
  if (wouldCreateCycle(others, c.source, c.target)) return 'That would create a cycle — data must flow one way.'
  return null
}

/**
 * Dropping a connection on a node body: pick the first free input (dragging from an output) or the
 * first output (dragging from an input). Returns the connection, or a reason it can't be made.
 */
export function connectionToBody(
  nodes: AppNode[],
  edges: AppEdge[],
  from: { nodeId: string; handleId: string | null; type: 'source' | 'target' },
  targetNodeId: string,
): { connection: ConnectionLike } | { problem: string } {
  const node = nodes.find((n) => n.id === targetNodeId)
  if (node?.type !== 'part') return { problem: 'Only model parts have ports.' }
  const def = getNodeDef(node.data.partType)
  if (!def) return { problem: 'Unknown part.' }

  if (from.type === 'source') {
    let free = def.inputs.filter((p) => !edges.some((e) => e.target === targetNodeId && e.targetHandle === p.id))
    if (def.inputs.length === 0) return { problem: `${node.data.title || def.label} has no inputs.` }
    // A single occupied input is unambiguous: replace its edge (same as dropping on the port).
    if (free.length === 0 && def.inputs.length === 1) free = def.inputs
    if (free.length === 0) return { problem: `All inputs of ${node.data.title || def.label} are connected — drop on a specific input to replace it.` }
    let problem = ''
    for (const p of free) {
      const connection = { source: from.nodeId, sourceHandle: from.handleId, target: targetNodeId, targetHandle: p.id }
      const why = connectionProblem(nodes, edges, connection)
      if (!why) return { connection }
      problem = why
    }
    return { problem }
  }

  const out = def.outputs[0]
  if (!out) return { problem: `${node.data.title || def.label} has no outputs.` }
  const connection = { source: targetNodeId, sourceHandle: out.id, target: from.nodeId, targetHandle: from.handleId }
  const why = connectionProblem(nodes, edges, connection)
  return why ? { problem: why } : { connection }
}
