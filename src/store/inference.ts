// Bridges React Flow nodes/edges to the pure engine (no React).
import type { AppEdge, AppNode, PartNode } from '../canvas/types'
import { flattenGroups, GROUP_INPUT, GROUP_OUTPUT, isProxyType } from '../engine/groups'
import { inferShapes } from '../engine/infer'
import { estimateMemory, type MemoryInput, type MemoryMode, type MemoryReport } from '../engine/memory'
import { accountParams, type GroupInfo, type ParamReport } from '../engine/params'
import type { GraphModel, Hyperparams, InferenceResult, NodeStatus, Shape } from '../engine/types'
import { GROUP_DEFS } from '../nodes/groups'
import { nodeRegistry } from '../nodes/registry'

/** What a group frame shows: computed from its descendants (groups have no logic of their own). */
export interface GroupSummary {
  /** Sum of all descendant parts' params. */
  params: number
  /** 'error' if any descendant has an error, 'unknown' if any is unknown, else 'ok'. */
  status: NodeStatus
  /** Descendant errors, prefixed with the part's title. */
  errors: string[]
  /** Shapes at the outer ports (= the proxies' shapes). */
  inShape: Shape | null
  outShape: Shape | null
  /** Titles of the direct children (proxies left out), in layout order. */
  contains: string[]
  /** Param count per direct child (nested groups summed), for the drawer. */
  breakdown: { id: string; title: string; params: number }[]
}

/**
 * Shapes + group summaries + parameter accounting (engine/params.ts: connected model, categories, layers).
 * `graph` (flattened) and `groupInfos` are kept for the memory estimate, which also depends on UI state (mode).
 */
export type CanvasInference = InferenceResult & { groups: Record<string, GroupSummary>; params: ParamReport; graph: GraphModel; groupInfos: GroupInfo[] }

/** Group frames are not parts: only parts (incl. proxies) go to the engine, with their parentId. */
export function toGraphModel(nodes: AppNode[], edges: Pick<AppEdge, 'id' | 'source' | 'target' | 'sourceHandle' | 'targetHandle'>[]): GraphModel {
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

/** Shape inference for the canvas: flatten groups onto their proxies, infer, summarise groups, count params. */
export function inferGraph(nodes: AppNode[], edges: AppEdge[], hp: Hyperparams): CanvasInference {
  const graph = flattenGroups(toGraphModel(nodes, edges))
  const result = inferShapes(graph, hp, nodeRegistry)
  const groups = nodes.flatMap((n) => (n.type === 'group' ? [{ id: n.id, type: n.data.groupType, ...(n.parentId ? { parentId: n.parentId } : {}) }] : []))
  return {
    ...result,
    groups: summarizeGroups(nodes, result),
    params: accountParams({ graph, groups, inference: result, defs: nodeRegistry, hp }),
    graph,
    groupInfos: groups,
  }
}

export function nodeTitle(n: AppNode): string {
  if (n.type === 'part') return n.data.title || nodeRegistry[n.data.partType]?.label || n.data.partType
  if (n.type === 'group') return n.data.title || GROUP_DEFS[n.data.groupType].label
  if (n.type === 'frame') return n.data.title || 'Frame'
  return n.type
}

export function summarizeGroups(nodes: AppNode[], result: InferenceResult): Record<string, GroupSummary> {
  const children = new Map<string, AppNode[]>()
  for (const n of nodes) {
    if (!n.parentId) continue
    if (!children.has(n.parentId)) children.set(n.parentId, [])
    children.get(n.parentId)!.push(n)
  }

  const summaries: Record<string, GroupSummary> = {}
  const summarize = (groupId: string): GroupSummary => {
    if (summaries[groupId]) return summaries[groupId]
    const s: GroupSummary = { params: 0, status: 'ok', errors: [], inShape: null, outShape: null, contains: [], breakdown: [] }
    let unknown = false
    for (const c of children.get(groupId) ?? []) {
      if (c.type === 'group') {
        const inner = summarize(c.id)
        s.params += inner.params
        s.errors.push(...inner.errors.map((e) => `${nodeTitle(c)} › ${e}`))
        unknown ||= inner.status === 'unknown'
        s.contains.push(nodeTitle(c))
        s.breakdown.push({ id: c.id, title: nodeTitle(c), params: inner.params })
      } else if (c.type === 'part') {
        const r = result.nodes[c.id]
        if (!r) continue
        s.params += r.paramCount.total
        if (c.data.partType === GROUP_INPUT) s.inShape = r.outputShapes[0] ?? null
        if (c.data.partType === GROUP_OUTPUT) s.outShape = r.outputShapes[0] ?? null
        if (r.status === 'error') s.errors.push(...r.errors.map((e) => (c.data.partType === GROUP_INPUT ? `Input not connected` : `${nodeTitle(c)}: ${e}`)))
        unknown ||= r.status === 'unknown'
        if (!isProxyType(c.data.partType)) {
          s.contains.push(nodeTitle(c))
          s.breakdown.push({ id: c.id, title: nodeTitle(c), params: r.paramCount.total })
        }
      }
    }
    s.status = s.errors.length > 0 ? 'error' : unknown ? 'unknown' : 'ok'
    summaries[groupId] = s
    return s
  }
  for (const n of nodes) if (n.type === 'group') summarize(n.id)
  return summaries
}

// Tiny memo: dragging a node changes `nodes` but not any node's `data`, so we can skip re-inferring.
let last: { nodes: AppNode[]; edges: AppEdge[]; hp: Hyperparams; result: CanvasInference } | null = null

function sameGraphInputs(a: AppNode[], b: AppNode[]): boolean {
  return a.length === b.length && a.every((n, i) => n.id === b[i].id && n.data === b[i].data && n.parentId === b[i].parentId)
}

export function inferCanvas(nodes: AppNode[], edges: AppEdge[], hp: Hyperparams): CanvasInference {
  if (last && last.hp === hp && last.edges === edges && sameGraphInputs(last.nodes, nodes)) {
    last = { ...last, nodes }
    return last.result
  }
  const result = inferGraph(nodes, edges, hp)
  last = { nodes, edges, hp, result }
  return result
}

/** Engine input for the memory estimate (engine/memory.ts). */
export function memoryInput(inference: CanvasInference, hp: Hyperparams, mode: MemoryMode, checkpointing: boolean): MemoryInput {
  return { graph: inference.graph, groups: inference.groupInfos, inference, params: inference.params, defs: nodeRegistry, hp, mode, checkpointing }
}

// Memo for the memory estimate: recomputed only when the inference, hyperparams, mode or checkpointing change.
let lastMemory: { inference: CanvasInference; hp: Hyperparams; mode: MemoryMode; checkpointing: boolean; report: MemoryReport } | null = null

export function memoryFor(inference: CanvasInference, hp: Hyperparams, mode: MemoryMode, checkpointing: boolean): MemoryReport {
  const m = lastMemory
  if (m && m.inference === inference && m.hp === hp && m.mode === mode && m.checkpointing === checkpointing) return m.report
  const report = estimateMemory(memoryInput(inference, hp, mode, checkpointing))
  lastMemory = { inference, hp, mode, checkpointing, report }
  return report
}
