// Parameter accounting (PLAN.md §4, R7). Pure TS, no React.
//
// Rule for "the model": a part counts towards the model total when it feeds (directly or through
// other parts) a Logits or Loss part — i.e. it is on the path to the output. Parts that don't
// (a stray Linear, a block that isn't wired in yet) are reported separately as "unconnected" so
// they never silently inflate the total. If the canvas has no Logits / Loss part at all, every
// part counts.
//
// Categories (for the breakdown bar):
//   Embedding  — Embedding parts
//   Attention  — parts inside a Multi-Head Self-Attention group (q/k/v/output projections)
//   FFN        — parts inside a SwiGLU or non-gated FFN group (w1/w2/w3, or w1/w2)
//   Norms      — RMSNorm / LayerNorm parts
//   LM head    — a Linear whose output goes straight into a Logits part (0 params when its weight is
//                tied to the embedding — hyperparam `tie_embeddings`, engine/tying.ts)
//   Other      — anything else with weights (e.g. an extra Linear between blocks)
import { formatCount } from './format'
import type { Category, GraphModel, Hyperparams, InferenceResult, NodeDef } from './types'

export type ParamCategory = 'embedding' | 'attention' | 'ffn' | 'norm' | 'lm_head' | 'other'
export const PARAM_CATEGORIES: ParamCategory[] = ['embedding', 'attention', 'ffn', 'norm', 'lm_head', 'other']

/** What can be highlighted on the canvas from the analysis panel. */
export type HighlightKey = ParamCategory | 'unconnected'

/** Part types that mark the model's output: everything feeding them is "the model". */
export const OUTPUT_PART_TYPES = ['logits', 'loss']

/** Group containers (React Flow parents); `type` is a GroupType ('transformer_block' | 'mha' | 'swiglu' | 'ffn'). */
export interface GroupInfo {
  id: string
  type: string
  parentId?: string
}

/** A part that has weights. */
export interface PartParams {
  id: string
  params: number
  category: ParamCategory
  /** Feeds a Logits / Loss part (counted in the model total). */
  connected: boolean
  /** Row in the per-layer breakdown: the part's top-level group (e.g. a Transformer Block) or the part itself. */
  row: string
  /** Weight tying: the Embedding part whose matrix this LM head reuses (`params` is then 0). */
  tiedTo?: string
}

/** One row of the per-layer breakdown, in data-flow order. */
export interface ParamRow {
  /** Group id (a Transformer Block = a layer) or part id (embedding, ln_final, lm_head, …). */
  id: string
  isLayer: boolean
  params: number
  byCategory: Record<ParamCategory, number>
}

export interface ParamFormula {
  /** Connected Transformer Blocks. */
  L: number
  symbolic: string
  substituted: string
  /** Value of the standard CS336 formula for the current hyperparams and L. */
  value: number
  /** Expected value per category under the standard formula. */
  expected: Record<ParamCategory, number>
  /** True when every category on the canvas equals the formula's term (standard structure). */
  matches: boolean
  /** Categories where the canvas differs from the formula. */
  diffs: { category: ParamCategory; actual: number; expected: number }[]
  /** Weight tying is on: the `+ d·V` LM head term is dropped (the embedding matrix is counted once). */
  tied: boolean
  /** Why the canvas differs from the standard terms (non-gated FFN, LayerNorm, …), when we can tell. */
  notes: string[]
}

export interface ParamReport {
  /** Params of the connected model. */
  total: number
  /** Params in parts that don't feed Logits / Loss. */
  unconnected: number
  unconnectedIds: string[]
  /** False when there is no Logits / Loss part (then every part counts as connected). */
  hasOutput: boolean
  /** Ids of every part (with or without weights) that belongs to the model: feeds Logits / Loss, or all parts when there is none. Memory uses it too. */
  connectedIds: Set<string>
  byCategory: Record<ParamCategory, number>
  /** Every part with params > 0. */
  parts: Record<string, PartParams>
  rows: ParamRow[]
  /** Connected LM heads whose weight is tied to an Embedding (they add 0); `params` = what they would add untied. */
  tied: { id: string; to: string; params: number }[]
  /** Per group: highlight keys of the weighted parts inside, and connected / unconnected params. */
  groups: Record<string, { keys: HighlightKey[]; connected: number; unconnected: number }>
  formula: ParamFormula
}

export interface ParamAccountingInput {
  /** Flattened graph (edges redirected onto group proxies, see engine/groups.ts); nodes keep `parentId`. */
  graph: GraphModel
  groups: GroupInfo[]
  inference: InferenceResult
  defs: Record<string, NodeDef>
  hp: Hyperparams
}

export function emptyByCategory(): Record<ParamCategory, number> {
  return { embedding: 0, attention: 0, ffn: 0, norm: 0, lm_head: 0, other: 0 }
}

export function accountParams({ graph, groups, inference, defs, hp }: ParamAccountingInput): ParamReport {
  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]))
  const groupById = new Map(groups.map((g) => [g.id, g]))
  const edges = graph.edges.filter((e) => nodeById.has(e.source) && nodeById.has(e.target))

  // Connected = can reach an output part (walk edges backwards from Logits / Loss).
  const outputs = graph.nodes.filter((n) => OUTPUT_PART_TYPES.includes(n.type)).map((n) => n.id)
  const hasOutput = outputs.length > 0
  const predecessors = new Map<string, string[]>()
  for (const e of edges) {
    if (!predecessors.has(e.target)) predecessors.set(e.target, [])
    predecessors.get(e.target)!.push(e.source)
  }
  const connected = new Set<string>()
  const stack = [...outputs]
  while (stack.length > 0) {
    const id = stack.pop()!
    if (connected.has(id)) continue
    connected.add(id)
    stack.push(...(predecessors.get(id) ?? []))
  }

  const feedsLogits = new Set(edges.filter((e) => nodeById.get(e.target)?.type === 'logits').map((e) => e.source))
  const ancestors = (parentId?: string): GroupInfo[] => {
    const out: GroupInfo[] = []
    for (let g = parentId ? groupById.get(parentId) : undefined; g; g = g.parentId ? groupById.get(g.parentId) : undefined) out.push(g)
    return out // innermost first
  }
  const categoryOf = (id: string, defCategory: Category, parentId?: string): ParamCategory => {
    if (defCategory === 'embedding') return 'embedding'
    if (defCategory === 'norm') return 'norm'
    const inside = ancestors(parentId).map((g) => g.type)
    if (inside.includes('mha') || defCategory === 'attention') return 'attention'
    if (inside.includes('swiglu') || inside.includes('ffn') || defCategory === 'ffn') return 'ffn'
    if (defCategory === 'linear' && feedsLogits.has(id)) return 'lm_head'
    return 'other'
  }

  const parts: Record<string, PartParams> = {}
  const byCategory = emptyByCategory()
  const groupStats: ParamReport['groups'] = {}
  for (const g of groups) groupStats[g.id] = { keys: [], connected: 0, unconnected: 0 }
  let total = 0
  let unconnected = 0
  const unconnectedIds: string[] = []
  const tied: ParamReport['tied'] = []
  // For the formula notes: connected non-gated FFN groups and LayerNorm parts.
  const nonGatedFfns = new Set<string>()
  let layerNorms = 0

  for (const n of graph.nodes) {
    const count = inference.nodes[n.id]?.paramCount
    const params = count?.total ?? 0
    const def = defs[n.type]
    // A tied LM head has 0 params of its own but is still listed (as "tied").
    if ((params <= 0 && !count?.tied) || !def) continue
    const isConnected = !hasOutput || connected.has(n.id)
    const up = ancestors(n.parentId)
    const p: PartParams = {
      id: n.id,
      params,
      category: categoryOf(n.id, def.category, n.parentId),
      connected: isConnected,
      row: up.length > 0 ? up[up.length - 1].id : n.id,
      ...(count?.tied ? { tiedTo: count.tied.to } : {}),
    }
    parts[n.id] = p
    if (isConnected && count?.tied) tied.push({ id: n.id, to: count.tied.to, params: count.tied.params })
    if (isConnected) {
      const ffn = up.find((g) => g.type === 'ffn')
      if (ffn) nonGatedFfns.add(ffn.id)
      if (n.type === 'layernorm') layerNorms += 1
    }
    const key: HighlightKey = isConnected ? p.category : 'unconnected'
    for (const g of up) {
      const s = groupStats[g.id]
      if (!s.keys.includes(key)) s.keys.push(key)
      if (isConnected) s.connected += params
      else s.unconnected += params
    }
    if (isConnected) {
      total += params
      byCategory[p.category] += params
    } else {
      unconnected += params
      unconnectedIds.push(n.id)
    }
  }

  // Rows in data-flow (topological) order of their first weighted part.
  const rows: ParamRow[] = []
  const rowIndex = new Map<string, ParamRow>()
  for (const id of topoOrder(graph.nodes.map((n) => n.id), edges)) {
    const p = parts[id]
    if (!p || !p.connected) continue
    let row = rowIndex.get(p.row)
    if (!row) {
      row = { id: p.row, isLayer: groupById.get(p.row)?.type === 'transformer_block', params: 0, byCategory: emptyByCategory() }
      rowIndex.set(p.row, row)
      rows.push(row)
    }
    row.params += p.params
    row.byCategory[p.category] += p.params
  }

  const L = rows.filter((r) => r.isLayer).length
  const notes: string[] = []
  const s = (k: number) => (k === 1 ? '' : 's')
  if (nonGatedFfns.size > 0)
    notes.push(
      `${nonGatedFfns.size} non-gated FFN${s(nonGatedFfns.size)} (2·d·d_ff each: w1, w2) where the formula assumes SwiGLU (3·d·d_ff: w1, w2, w3). With its own d_ff = 4·d_model a non-gated FFN has about the same params as SwiGLU with d_ff ≈ 8/3·d_model.`,
    )
  if (hp.tie_embeddings && tied.length === 0 && byCategory.lm_head > 0)
    notes.push('Weight tying is on, but the LM head could not be tied (see its error), so it still counts its own d·V.')
  if (layerNorms > 0) notes.push(`${layerNorms} LayerNorm${s(layerNorms)} (2·d each: gain γ and bias β) where the formula assumes RMSNorm (d: gain only).`)
  return {
    total,
    unconnected,
    unconnectedIds,
    hasOutput,
    connectedIds: hasOutput ? connected : new Set(graph.nodes.map((n) => n.id)),
    byCategory,
    parts,
    rows,
    tied,
    groups: groupStats,
    formula: { ...standardFormula(hp, L, byCategory), notes },
  }
}

/**
 * The standard CS336 TransformerLM count (SwiGLU, RMSNorm, no biases, no weight tying):
 *   V·d + L·(4d² + 3d·d_ff + 2d) + d + d·V
 * With `tie_embeddings` the LM head reuses the embedding matrix, so the `+ d·V` term is dropped.
 * `actual` (the canvas's per-category sums) decides whether the canvas matches it.
 */
export function standardFormula(hp: Hyperparams, L: number, actual: Record<ParamCategory, number> = emptyByCategory()): ParamFormula {
  const { vocab_size: V, d_model: d, d_ff: F } = hp
  const tied = !!hp.tie_embeddings
  const expected: Record<ParamCategory, number> = {
    embedding: V * d,
    attention: L * 4 * d * d,
    ffn: L * 3 * d * F,
    norm: (2 * L + 1) * d,
    lm_head: tied ? 0 : d * V,
    other: 0,
  }
  const value = Object.values(expected).reduce((a, b) => a + b, 0)
  const diffs = PARAM_CATEGORIES.filter((c) => actual[c] !== expected[c]).map((c) => ({ category: c, actual: actual[c], expected: expected[c] }))
  return {
    L,
    symbolic: `V·d + L·(4d² + 3d·d_ff + 2d) + d${tied ? '' : ' + d·V'}`,
    substituted: `${V}·${d} + ${L}·(4·${d}² + 3·${d}·${F} + 2·${d}) + ${d}${tied ? '' : ` + ${d}·${V}`}`,
    value,
    expected,
    matches: diffs.length === 0,
    diffs,
    tied,
    notes: [],
  }
}

/** Short factual observations about where the parameters are (shown under the breakdown). */
export function paramInsights(report: ParamReport): string[] {
  const { total, byCategory: c, rows } = report
  if (total <= 0) return []
  const pct = (x: number) => {
    const v = (x / total) * 100
    return `${v > 0 && v < 1 ? v.toFixed(2) : Math.round(v)}%`
  }
  const out: string[] = []
  const vocab = c.embedding + c.lm_head
  const layers = rows.filter((r) => r.isLayer)
  const blocks = layers.reduce((a, r) => a + r.params, 0)
  const perBlock = layers.length > 0 ? blocks / layers.length : 0
  const tied = report.tied.length > 0

  if (tied) {
    const saved = report.tied.reduce((a, t) => a + t.params, 0)
    out.push(`Weight tying: the LM head reuses the embedding matrix — one V × d table instead of two, saving ${formatCount(saved)} parameters (untied: ${formatCount(total + saved)}).`)
  }
  if (vocab > 0 && vocab >= blocks) {
    let s = tied
      ? `Embedding (shared with the LM head) = ${pct(vocab)} of the parameters at this size: the V × d table outweighs the ${layers.length} block${layers.length === 1 ? '' : 's'}.`
      : `Embedding + LM head = ${pct(vocab)} of the parameters at this size: the two V × d tables outweigh the ${layers.length} block${layers.length === 1 ? '' : 's'}.`
    if (perBlock > 0) {
      const crossover = Math.floor(vocab / perBlock) + 1
      s += ` With ≥ ${crossover} blocks (or a larger d_model — blocks grow with d², the tables with d) the blocks would dominate.`
    }
    out.push(s)
  } else if (blocks > 0) {
    out.push(
      `Transformer blocks hold ${pct(blocks)} of the parameters (${layers.length} × ${formatCount(Math.round(perBlock))}); ${tied ? 'embedding (shared with the LM head)' : 'embedding + LM head'} = ${pct(vocab)}.`,
    )
  }
  if (c.attention > 0 && c.ffn > 0) {
    const terms = c.ffn === report.formula.expected.ffn && c.attention === report.formula.expected.attention ? ' (3·d·d_ff vs 4·d²)' : ''
    out.push(`FFN has ${(c.ffn / c.attention).toFixed(2)}× the weights of attention${terms}; norms are just ${pct(c.norm)}.`)
  }
  return out
}

/** Kahn's algorithm; nodes on cycles are appended at the end in input order. */
export function topoOrder(ids: string[], edges: { source: string; target: string }[]): string[] {
  const indegree = new Map(ids.map((id) => [id, 0]))
  const successors = new Map<string, string[]>(ids.map((id) => [id, []]))
  for (const e of edges) {
    indegree.set(e.target, indegree.get(e.target)! + 1)
    successors.get(e.source)!.push(e.target)
  }
  const queue = ids.filter((id) => indegree.get(id) === 0)
  const order: string[] = []
  while (queue.length > 0) {
    const id = queue.shift()!
    order.push(id)
    for (const next of successors.get(id)!) {
      indegree.set(next, indegree.get(next)! - 1)
      if (indegree.get(next) === 0) queue.push(next)
    }
  }
  const seen = new Set(order)
  return [...order, ...ids.filter((id) => !seen.has(id))]
}
