// Connection rules (PLAN.md §2.4): output → input only, one edge per input (new edge replaces
// the old one), no cycles. Groups (PLAN.md §2.3) behave like parts with one input and one output;
// an edge may only join nodes in the same container (both top-level, or both inside the same
// group), so things outside a group connect to its outer ports, never to the parts inside.
// Pure functions so Canvas and tests can share them.
import { flattenGroups, GROUP_INPUT, GROUP_OUTPUT } from '../engine/groups'
import { wouldCreateCycle } from '../engine/infer'
import type { PortDef } from '../engine/types'
import { GROUP_PORTS } from '../nodes/groups'
import { getNodeDef } from '../nodes/registry'
import { nodeTitle, toGraphModel } from '../store/inference'
import type { AppEdge, AppNode } from './types'

export interface ConnectionLike {
  source: string
  sourceHandle?: string | null
  target: string
  targetHandle?: string | null
}

/**
 * The ports a node shows on the canvas (null = no ports). Proxies only show their inner side:
 * the group input proxy has just an output, the group output proxy just an input.
 */
export function nodePorts(node: AppNode | undefined): { inputs: PortDef[]; outputs: PortDef[] } | null {
  if (node?.type === 'group') return GROUP_PORTS
  if (node?.type !== 'part') return null
  const def = getNodeDef(node.data.partType)
  if (!def) return null
  if (def.type === GROUP_INPUT) return { inputs: [], outputs: def.outputs }
  if (def.type === GROUP_OUTPUT) return { inputs: def.inputs, outputs: [] }
  return { inputs: def.inputs, outputs: def.outputs }
}

/** null if the connection is allowed, otherwise a short human-readable reason. */
export function connectionProblem(nodes: AppNode[], edges: AppEdge[], c: ConnectionLike): string | null {
  if (c.source === c.target) return "A part can't connect to itself."
  const src = nodes.find((n) => n.id === c.source)
  const tgt = nodes.find((n) => n.id === c.target)
  const sp = nodePorts(src)
  const tp = nodePorts(tgt)
  if (!sp || !tp) return 'Only model parts have ports.'
  if (!sp.outputs.some((o) => o.id === c.sourceHandle) || !tp.inputs.some((i) => i.id === c.targetHandle))
    return 'Connect an output (green, bottom) to an input (blue, top).'
  if ((src!.parentId ?? null) !== (tgt!.parentId ?? null))
    return 'Parts inside a group only connect to each other — from outside, use the group’s in/out ports.'
  // The edge currently on this input would be replaced, so leave it out of the cycle check.
  // Check on the flattened graph so paths through groups (in → … → out) count.
  const others = edges.filter((e) => !(e.target === c.target && e.targetHandle === c.targetHandle))
  const candidate = { id: '__candidate', source: c.source, target: c.target, sourceHandle: c.sourceHandle ?? null, targetHandle: c.targetHandle ?? null }
  const flat = flattenGroups(toGraphModel(nodes, [...others, candidate])).edges
  const cand = flat.find((e) => e.id === candidate.id)!
  if (wouldCreateCycle(flat.filter((e) => e !== cand), cand.source, cand.target)) return 'That would create a cycle — data must flow one way.'
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
  const ports = nodePorts(node)
  if (!node || !ports) return { problem: 'Only model parts have ports.' }
  const name = nodeTitle(node)

  if (from.type === 'source') {
    let free = ports.inputs.filter((p) => !edges.some((e) => e.target === targetNodeId && e.targetHandle === p.id))
    if (ports.inputs.length === 0) return { problem: `${name} has no inputs.` }
    // A single occupied input is unambiguous: replace its edge (same as dropping on the port).
    if (free.length === 0 && ports.inputs.length === 1) free = ports.inputs
    if (free.length === 0) return { problem: `All inputs of ${name} are connected — drop on a specific input to replace it.` }
    let problem = ''
    for (const p of free) {
      const connection = { source: from.nodeId, sourceHandle: from.handleId, target: targetNodeId, targetHandle: p.id }
      const why = connectionProblem(nodes, edges, connection)
      if (!why) return { connection }
      problem = why
    }
    return { problem }
  }

  const out = ports.outputs[0]
  if (!out) return { problem: `${name} has no outputs.` }
  const connection = { source: targetNodeId, sourceHandle: out.id, target: from.nodeId, targetHandle: from.handleId }
  const why = connectionProblem(nodes, edges, connection)
  return why ? { problem: why } : { connection }
}
