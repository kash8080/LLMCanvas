// Bridges React Flow nodes/edges to the pure engine (no React).
import type { AppEdge, AppNode, PartNode } from '../canvas/types'
import { inferShapes } from '../engine/infer'
import type { GraphModel, Hyperparams, InferenceResult } from '../engine/types'
import { nodeRegistry } from '../nodes/registry'

export function toGraphModel(nodes: AppNode[], edges: AppEdge[]): GraphModel {
  const parts = nodes.filter((n): n is PartNode => n.type === 'part')
  return {
    nodes: parts.map((n) => ({ id: n.id, type: n.data.partType, params: n.data.params, ...(n.parentId ? { parentId: n.parentId } : {}) })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      sourceHandle: e.sourceHandle ?? null,
      target: e.target,
      targetHandle: e.targetHandle ?? null,
    })),
  }
}

// Tiny memo: dragging a node changes `nodes` but not any node's `data`, so we can skip re-inferring.
let last: { nodes: AppNode[]; edges: AppEdge[]; hp: Hyperparams; result: InferenceResult } | null = null

function sameGraphInputs(a: AppNode[], b: AppNode[]): boolean {
  return a.length === b.length && a.every((n, i) => n.id === b[i].id && n.data === b[i].data && n.parentId === b[i].parentId)
}

export function inferCanvas(nodes: AppNode[], edges: AppEdge[], hp: Hyperparams): InferenceResult {
  if (last && last.hp === hp && last.edges === edges && sameGraphInputs(last.nodes, nodes)) {
    last = { ...last, nodes }
    return last.result
  }
  const result = inferShapes(toGraphModel(nodes, edges), hp, nodeRegistry)
  last = { nodes, edges, hp, result }
  return result
}
