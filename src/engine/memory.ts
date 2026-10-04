// Memory estimation (PLAN.md §5, R8). Pure TS, no React. An educational estimate, not allocator-exact.
//
//   weights     = P · b                     (every mode)
//   gradients   = P · b                     (fwd_bwd, train)
//   optimizer   = 2 · P · b                 (train: AdamW m and v, `zeros_like(p)` → same dtype as p)
//   buffers     = RoPE cos + sin tables     (every mode; once per attention group, see below)
//   activations = forward: the peak "live" set — max over parts of (its inputs + outputs + internal
//                 temporaries like the attention probs), a simple approximation;
//                 fwd_bwd / train: every tensor saved for backward, each unique tensor counted once.
//
// P = connected params (engine/params.ts), b = bytes per element of the float dtype (fp32 4, bf16/fp16 2);
// int64 tensors (token ids, targets) are always 8 bytes. Only parts of the connected model count.
//
// Tensor identity: a tensor is its producing part + output port, after skipping pass-through parts
// (group in/out proxies, Logits). So ln1's output feeding q_proj, k_proj and v_proj is one tensor,
// stored once. Internal tensors (attention probs, RMSNorm's rms) are keyed by the part that creates them.
// Attribution: a tensor's bytes belong to the part that produced it (internal ones: the part that saves
// them); tensors coming from a source part without inputs (Data Batch token ids / targets) belong to
// the part that saves them.
//
// Activation checkpointing (CS336 `checkpoint_blocks`: one torch.utils.checkpoint per Transformer Block):
// tensors saved inside a block are not kept; instead each block's input is kept (B·T·d per block) and,
// during backward, one block at a time is recomputed — so the largest block's saved tensors are added once.
// Everything outside the blocks (embedding, ln_final, lm_head, loss) is saved as usual.
//
// RoPE buffers: CS336 builds one RotaryPositionalEmbedding per MultiHeadSelfAttention and uses it for both
// q and k. The canvas shows two RoPE parts per attention group, so RoPE parts inside the same attention
// group share one cos/sin pair (counted once); a RoPE part outside an attention group counts on its own.
import { GROUP_INPUT, GROUP_OUTPUT } from './groups'
import { bytesPerElement } from './hyperparams'
import { formatBytes } from './format'
import { topoOrder, type GroupInfo, type ParamReport } from './params'
import { num } from './resolve'
import { numel } from './shape'
import type { GraphModel, Hyperparams, InferenceResult, NodeDef, Shape } from './types'

export type MemoryMode = 'forward' | 'fwd_bwd' | 'train'
export const MEMORY_MODES: MemoryMode[] = ['forward', 'fwd_bwd', 'train']

export type MemoryComponent = 'weights' | 'gradients' | 'optimizer' | 'activations' | 'buffers'
export const MEMORY_COMPONENTS: MemoryComponent[] = ['weights', 'gradients', 'optimizer', 'activations', 'buffers']

/** Activation buckets. Attention probs get their own bucket: B·H·T² is usually the big one. */
export type MemoryCategory = 'attn_probs' | 'attention' | 'ffn' | 'norm' | 'embedding' | 'logits' | 'residual'
export const MEMORY_CATEGORIES: MemoryCategory[] = ['attn_probs', 'attention', 'ffn', 'norm', 'embedding', 'logits', 'residual']

/** Canvas highlight key for an activation category (the Parameters tab uses the plain param categories). */
export type MemoryHighlightKey = `mem:${MemoryCategory}`
export const memKey = (c: MemoryCategory): MemoryHighlightKey => `mem:${c}`
export const isMemKey = (k: string): k is MemoryHighlightKey => k.startsWith('mem:')

/** Parts that only forward their input (no new tensor). */
const PASS_THROUGH = new Set([GROUP_INPUT, GROUP_OUTPUT, 'logits'])

/** One activation tensor (unique). */
export interface MemTensor {
  /** `<producer>:<port>` or `<part>#<internal name>`. */
  key: string
  /** Part that produced it (internal tensors: the part that creates them). */
  producer: string
  /** Part the bytes are attributed to (see the header comment). */
  owner: string
  /** Producing port id ('out', 'targets', …) or the internal tensor's name ('attention probs', 'rms'). */
  label: string
  internal: boolean
  shape: Shape
  elements: number
  bytes: number
  category: MemoryCategory
  /** Layer row: the owner's top-level group (a Transformer Block) or the owner itself. */
  row: string
  /** fwd_bwd / train: parts that save it for backward (several = shared, counted once). forward: the part using it. */
  savedBy: string[]
  /** Counted in the activation total (false = dropped by activation checkpointing; recomputed in backward). */
  counted: boolean
  /** Checkpointing: 'block_input' = kept as a checkpointed block's input; 'recompute' = in the block recomputed at the peak. */
  role?: 'block_input' | 'recompute'
}

export interface MemoryRow {
  id: string
  isLayer: boolean
  bytes: number
  byCategory: Record<MemoryCategory, number>
}

export interface RopeBuffer {
  /** Attention group id, or the RoPE part id when it isn't inside one. */
  id: string
  ropeIds: string[]
  maxSeqLen: number
  headDim: number
  /** cos + sin: 2 · max_seq_len · head_dim/2. */
  elements: number
  bytes: number
}

export interface MemoryReport {
  mode: MemoryMode
  checkpointing: boolean
  /** Bytes per float element. */
  b: number
  /** Connected params. */
  P: number
  total: number
  byComponent: Record<MemoryComponent, number>
  activations: {
    total: number
    /** Every unique tensor considered, biggest first; `counted` says whether it is in the total. */
    tensors: MemTensor[]
    byCategory: Record<MemoryCategory, number>
    /** Counted bytes per owning part. forward mode: bytes live while that part runs. */
    byPart: Record<string, number>
    /** Per layer row (data-flow order). forward mode: the peak within the row. */
    rows: MemoryRow[]
    /** forward: the part with the largest live set. */
    peakPart?: string
    /** Checkpointing: the block whose saved tensors are counted (recomputed during backward), and its bytes. */
    recomputeBlock?: string
    recomputeBytes: number
    /** Checkpointing: number of checkpointed blocks and the bytes of their inputs. */
    checkpointedBlocks: number
    blockInputBytes: number
    /** Connected parts whose shapes are unknown / in error (not counted). */
    skipped: string[]
  }
  buffers: { total: number; items: RopeBuffer[] }
  /** Per group: counted activation bytes inside (forward: peak inside) and the categories present (for highlighting). */
  groups: Record<string, { activations: number; keys: MemoryCategory[] }>
  /** Per part: memory categories it owns (for highlighting). */
  partKeys: Record<string, MemoryCategory[]>
}

export interface MemoryInput {
  /** Flattened graph (engine/groups.ts); nodes keep `parentId`. */
  graph: GraphModel
  groups: GroupInfo[]
  inference: InferenceResult
  params: ParamReport
  defs: Record<string, NodeDef>
  hp: Hyperparams
  mode: MemoryMode
  checkpointing: boolean
}

export function emptyMemByCategory(): Record<MemoryCategory, number> {
  return { attn_probs: 0, attention: 0, ffn: 0, norm: 0, embedding: 0, logits: 0, residual: 0 }
}

export function estimateMemory(input: MemoryInput): MemoryReport {
  const { graph, groups, inference, params, defs, hp, mode } = input
  const checkpointing = input.checkpointing && mode !== 'forward'
  const b = bytesPerElement('float', hp)
  const P = params.total

  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]))
  const groupById = new Map(groups.map((g) => [g.id, g]))
  const edges = graph.edges.filter((e) => nodeById.has(e.source) && nodeById.has(e.target))
  const incoming = new Map<string, (typeof edges)[number]>()
  for (const e of edges) incoming.set(`${e.target}:${e.targetHandle}`, e)
  const intoNode = (id: string) => edges.find((e) => e.target === id)
  const isConnected = (id: string) => params.connectedIds.has(id)

  const ancestors = (id: string): GroupInfo[] => {
    const out: GroupInfo[] = []
    const start = nodeById.get(id)?.parentId ?? groupById.get(id)?.parentId
    for (let g = start ? groupById.get(start) : undefined; g; g = g.parentId ? groupById.get(g.parentId) : undefined) out.push(g)
    return out // innermost first
  }
  const rowOf = (id: string) => {
    const up = ancestors(id)
    return up.length > 0 ? up[up.length - 1].id : id
  }
  /** Outermost Transformer Block containing the part (checkpointing unit). */
  const blockOf = (id: string): string | undefined => {
    const blocks = ancestors(id).filter((g) => g.type === 'transformer_block')
    return blocks.length > 0 ? blocks[blocks.length - 1].id : undefined
  }
  const feedsLogits = new Set(edges.filter((e) => nodeById.get(e.target)?.type === 'logits').map((e) => e.source))

  /** Producer of what arrives at (node, port), skipping pass-through parts. */
  const resolveSource = (nodeId: string, port: string): { source: string; handle: string } | null => {
    const e = incoming.get(`${nodeId}:${port}`)
    if (!e) return null
    let source = e.source
    let handle = e.sourceHandle ?? 'out'
    for (let guard = 0; guard < 1000 && PASS_THROUGH.has(nodeById.get(source)?.type ?? ''); guard++) {
      const up = intoNode(source)
      if (!up) break
      source = up.source
      handle = up.sourceHandle ?? 'out'
    }
    return { source, handle }
  }

  const categoryOf = (owner: string, internal: string | null, saver: string): MemoryCategory => {
    const type = nodeById.get(owner)?.type ?? ''
    const def = defs[type]
    if ((type === 'sdpa' && internal === 'attention probs') || type === 'softmax') return 'attn_probs'
    const inside = ancestors(owner).map((g) => g.type)
    if (inside.includes('swiglu')) return 'ffn'
    if (inside.includes('mha') || def?.category === 'attention') return 'attention'
    if (def?.category === 'norm') return 'norm'
    if (def?.category === 'embedding') return 'embedding'
    if (def?.category === 'loss' || type === 'logits' || (def?.category === 'linear' && feedsLogits.has(owner))) return 'logits'
    if (def?.category === 'io' && owner !== saver) return categoryOf(saver, null, saver)
    return 'residual'
  }

  // Parts that take part: connected, real (not pass-through), with known shapes.
  const order = topoOrder(graph.nodes.map((n) => n.id), edges)
  const skipped: string[] = []
  const active: string[] = []
  for (const id of order) {
    const n = nodeById.get(id)!
    if (!isConnected(id) || PASS_THROUGH.has(n.type) || !defs[n.type]) continue
    if (inference.nodes[id]?.status !== 'ok') skipped.push(id)
    else active.push(id)
  }

  /** A tensor record (not yet registered anywhere). `user` = the part that saves / uses it. */
  const newTensor = (key: string, owner: string, label: string, internal: boolean, shape: Shape, user: string): MemTensor => {
    // A source part without inputs (Data Batch): attribute to the part that uses the tensor.
    const ownerDef = defs[nodeById.get(owner)?.type ?? '']
    const attributed = !internal && ownerDef && ownerDef.inputs.length === 0 ? user : owner
    const elements = numel(shape)
    return {
      key,
      producer: owner,
      owner: attributed,
      label,
      internal,
      shape,
      elements,
      bytes: elements * bytesPerElement(shape.dtype, hp),
      category: categoryOf(owner, internal ? label : null, user),
      row: rowOf(attributed),
      savedBy: [user],
      counted: true,
    }
  }
  /** Saved tensors, deduplicated by key: a second saver only adds itself to `savedBy`. */
  const tensorsMap = new Map<string, MemTensor>()
  const saveTensor = (key: string, owner: string, label: string, internal: boolean, shape: Shape, user: string): MemTensor => {
    const t = tensorsMap.get(key)
    if (!t) {
      const fresh = newTensor(key, owner, label, internal, shape, user)
      tensorsMap.set(key, fresh)
      return fresh
    }
    if (!t.savedBy.includes(user)) t.savedBy.push(user)
    return t
  }

  const byPart: Record<string, number> = {}
  let peakPart: string | undefined
  let recomputeBlock: string | undefined
  let recomputeBytes = 0
  let checkpointedBlocks = 0
  let blockInputBytes = 0
  let tensors: MemTensor[]

  if (mode === 'forward') {
    // Live while each part runs: its unique inputs + outputs + internal temporaries.
    const live: Record<string, MemTensor[]> = {}
    for (const id of active) {
      const def = defs[nodeById.get(id)!.type]
      const r = inference.nodes[id]
      const set = new Map<string, MemTensor>()
      const add = (t: MemTensor) => set.set(t.key, t)
      def.inputs.forEach((port, i) => {
        const src = resolveSource(id, port.id)
        const shape = r.inputShapes[i]
        if (src && shape) add(newTensor(`${src.source}:${src.handle}`, src.source, src.handle, false, shape, id))
      })
      def.outputs.forEach((port, i) => {
        const shape = r.outputShapes[i]
        if (shape) add(newTensor(`${id}:${port.id}`, id, port.id, false, shape, id))
      })
      for (const s of savedFor(def, r, hp)) if (s.which === 'internal') add(newTensor(`${id}#${s.name}`, id, s.name, true, s.shape, id))
      live[id] = [...set.values()]
      byPart[id] = live[id].reduce((a, t) => a + t.bytes, 0)
      if (peakPart === undefined || byPart[id] > byPart[peakPart]) peakPart = id
    }
    tensors = peakPart ? live[peakPart] : []
  } else {
    // Saved for backward, deduplicated by tensor key.
    for (const id of active) {
      const n = nodeById.get(id)!
      const def = defs[n.type]
      const r = inference.nodes[id]
      for (const s of savedFor(def, r, hp)) {
        if (s.which === 'internal') saveTensor(`${id}#${s.name}`, id, s.name, true, s.shape, id)
        else if (s.which === 'output') saveTensor(`${id}:${s.port ?? 'out'}`, id, s.port ?? 'out', false, s.shape, id)
        else {
          const src = resolveSource(id, s.port ?? def.inputs[0]?.id ?? 'in')
          if (src) saveTensor(`${src.source}:${src.handle}`, src.source, src.handle, false, s.shape, id)
        }
      }
    }
    tensors = [...tensorsMap.values()]

    if (checkpointing) {
      // A tensor belongs to block K when every part saving it is inside K (outermost Transformer Block).
      const blockOfTensor = (t: MemTensor): string | undefined => {
        const bs = t.savedBy.map(blockOf)
        return bs.every((x) => x !== undefined && x === bs[0]) ? bs[0] : undefined
      }
      const blockIds = groups.filter((g) => g.type === 'transformer_block' && !ancestors(g.id).some((a) => a.type === 'transformer_block'))
      const inputKeys = new Set<string>()
      for (const g of blockIds) {
        const proxy = graph.nodes.find((n) => n.parentId === g.id && n.type === GROUP_INPUT)
        if (!proxy || !isConnected(proxy.id)) continue
        const src = resolveSource(proxy.id, 'in')
        const shape = inference.nodes[proxy.id]?.outputShapes[0]
        if (!src || !shape) continue
        checkpointedBlocks += 1
        const t = saveTensor(`${src.source}:${src.handle}`, src.source, src.handle, false, shape, g.id)
        t.role = 'block_input'
        inputKeys.add(t.key)
      }
      tensors = [...tensorsMap.values()]
      const perBlock = new Map<string, number>()
      for (const t of tensors) {
        if (inputKeys.has(t.key)) continue
        const blk = blockOfTensor(t)
        if (!blk) continue
        t.counted = false
        perBlock.set(blk, (perBlock.get(blk) ?? 0) + t.bytes)
      }
      for (const [blk, bytes] of perBlock) if (recomputeBlock === undefined || bytes > recomputeBytes) [recomputeBlock, recomputeBytes] = [blk, bytes]
      for (const t of tensors) {
        if (t.counted || t.role) continue
        if (blockOfTensor(t) === recomputeBlock) {
          t.counted = true
          t.role = 'recompute'
        }
      }
      blockInputBytes = tensors.filter((t) => inputKeys.has(t.key)).reduce((a, t) => a + t.bytes, 0)
    }
    for (const t of tensors) if (t.counted) byPart[t.owner] = (byPart[t.owner] ?? 0) + t.bytes
  }

  tensors.sort((x, y) => y.bytes - x.bytes)
  const counted = tensors.filter((t) => t.counted)
  const byCategory = emptyMemByCategory()
  for (const t of counted) byCategory[t.category] += t.bytes
  const activationTotal = mode === 'forward' ? (peakPart ? byPart[peakPart] : 0) : counted.reduce((a, t) => a + t.bytes, 0)

  // Highlight keys per part / group, and per-group activation bytes.
  const partKeys: Record<string, MemoryCategory[]> = {}
  const groupStats: MemoryReport['groups'] = {}
  for (const g of groups) groupStats[g.id] = { activations: 0, keys: [] }
  for (const t of counted) {
    for (const id of [t.owner, ...ancestors(t.owner).map((g) => g.id)]) {
      const keys = groupStats[id]?.keys ?? (partKeys[id] ??= [])
      if (!keys.includes(t.category)) keys.push(t.category)
    }
  }
  for (const [id, bytes] of Object.entries(byPart)) {
    for (const g of ancestors(id)) {
      const s = groupStats[g.id]
      s.activations = mode === 'forward' ? Math.max(s.activations, bytes) : s.activations + bytes
    }
  }

  // Rows in data-flow order.
  const rows: MemoryRow[] = []
  const rowIndex = new Map<string, MemoryRow>()
  const rowFor = (id: string) => {
    let row = rowIndex.get(id)
    if (!row) {
      row = { id, isLayer: groupById.get(id)?.type === 'transformer_block', bytes: 0, byCategory: emptyMemByCategory() }
      rowIndex.set(id, row)
    }
    return row
  }
  if (mode === 'forward') {
    for (const id of active) {
      const row = rowFor(rowOf(id))
      if (byPart[id] > row.bytes) row.bytes = byPart[id]
    }
  } else {
    for (const t of counted) {
      const row = rowFor(t.row)
      row.bytes += t.bytes
      row.byCategory[t.category] += t.bytes
    }
  }
  const rank = new Map(order.map((id, i) => [id, i]))
  const firstPart = (rowId: string) => {
    let best = Infinity
    for (const id of order) if (id === rowId || rowOf(id) === rowId) best = Math.min(best, rank.get(id)!)
    return best
  }
  for (const row of [...rowIndex.values()].sort((x, y) => firstPart(x.id) - firstPart(y.id))) if (row.bytes > 0) rows.push(row)

  // RoPE buffers: one cos/sin pair per attention group (CS336: one RotaryPositionalEmbedding per MHA).
  const buffers = new Map<string, RopeBuffer>()
  for (const id of order) {
    const n = nodeById.get(id)!
    if (n.type !== 'rope' || !isConnected(id)) continue
    const r = inference.nodes[id]
    if (!r) continue
    const maxSeqLen = num(r.resolved, 'max_seq_len')
    const headDim = num(r.resolved, 'head_dim')
    const mha = ancestors(id).find((g) => g.type === 'mha')
    const key = mha?.id ?? id
    const elements = 2 * maxSeqLen * Math.floor(headDim / 2)
    const prev = buffers.get(key)
    if (!prev) buffers.set(key, { id: key, ropeIds: [id], maxSeqLen, headDim, elements, bytes: elements * b })
    else {
      prev.ropeIds.push(id)
      if (elements > prev.elements) Object.assign(prev, { maxSeqLen, headDim, elements, bytes: elements * b })
    }
  }
  const bufferItems = [...buffers.values()]
  const bufferTotal = bufferItems.reduce((a, x) => a + x.bytes, 0)

  const byComponent: Record<MemoryComponent, number> = {
    weights: P * b,
    gradients: mode === 'forward' ? 0 : P * b,
    optimizer: mode === 'train' ? 2 * P * b : 0,
    activations: activationTotal,
    buffers: bufferTotal,
  }
  return {
    mode,
    checkpointing,
    b,
    P,
    total: MEMORY_COMPONENTS.reduce((a, c) => a + byComponent[c], 0),
    byComponent,
    activations: {
      total: activationTotal,
      tensors,
      byCategory,
      byPart,
      rows,
      peakPart,
      recomputeBlock,
      recomputeBytes: checkpointing ? counted.filter((t) => t.role === 'recompute').reduce((a, t) => a + t.bytes, 0) : 0,
      checkpointedBlocks,
      blockInputBytes,
      skipped,
    },
    buffers: { total: bufferTotal, items: bufferItems },
    groups: groupStats,
    partKeys,
  }
}

/** savedForBackward with the part's (known) shapes. */
function savedFor(def: NodeDef, r: InferenceResult['nodes'][string], hp: Hyperparams) {
  return def.savedForBackward({ inputs: r.inputShapes as Shape[], outputs: r.outputShapes as Shape[], p: r.resolved, hp })
}

/** The formula line for each component, with the numbers substituted. */
export function memoryFormulas(report: MemoryReport): Record<MemoryComponent, string> {
  const { P, b, mode, activations: a, buffers } = report
  const Pn = P.toLocaleString('en-US')
  const rope = buffers.items[0]
  const sameRope = buffers.items.every((x) => x.elements === rope?.elements)
  let activations: string
  if (mode === 'forward') activations = 'peak live tensors = max over parts of (inputs + outputs + temporaries)'
  else if (report.checkpointing)
    activations = `outside blocks + ${a.checkpointedBlocks} block input${a.checkpointedBlocks === 1 ? '' : 's'} (B·T·d each) + largest block, recomputed = ${fmt(
      a.total - a.blockInputBytes - a.recomputeBytes,
    )} + ${fmt(a.blockInputBytes)} + ${fmt(a.recomputeBytes)}`
  else activations = `Σ unique tensors saved for backward (${a.tensors.filter((t) => t.counted).length} tensors)`
  return {
    weights: `P · bytes = ${Pn} · ${b}`,
    gradients: mode === 'forward' ? 'none: no backward pass' : `P · bytes = ${Pn} · ${b}`,
    optimizer: mode === 'train' ? `2 · P · bytes = 2 · ${Pn} · ${b}   (AdamW m, v)` : 'none: no optimizer step',
    activations,
    buffers:
      buffers.items.length === 0
        ? 'no RoPE parts'
        : sameRope
          ? `RoPE cos + sin: ${buffers.items.length} · 2 · max_seq_len · d_head/2 · bytes = ${buffers.items.length} · 2 · ${rope.maxSeqLen} · ${rope.headDim / 2} · ${b}`
          : `RoPE cos + sin: Σ 2 · max_seq_len · d_head/2 · bytes over ${buffers.items.length} attention groups`,
  }
}

function fmt(n: number): string {
  return n.toLocaleString('en-US')
}

/** Recompute the estimate with some inputs changed (for "what if" insights). */
export function estimateWith(input: MemoryInput, patch: Partial<Pick<MemoryInput, 'mode' | 'checkpointing'>> & { hp?: Partial<Hyperparams> }): MemoryReport {
  return estimateMemory({ ...input, ...patch, hp: { ...input.hp, ...patch.hp } })
}

/** How each activation bucket scales (every one of them is linear in B). */
const CATEGORY_SCALING: Record<MemoryCategory, string> = {
  attn_probs: 'B·H·T² (quadratic in the context)',
  attention: 'B·T·d_model',
  ffn: 'B·T·d_ff',
  norm: 'B·T·d_model',
  embedding: 'B·T·d_model',
  logits: 'B·T·V (the vocabulary size)',
  residual: 'B·T·d_model',
}
const CATEGORY_TERM: Record<MemoryCategory, string> = {
  attn_probs: 'attention probs',
  attention: 'attention tensors (Q, K, V, merged heads)',
  ffn: 'SwiGLU tensors',
  norm: 'RMSNorm outputs',
  embedding: 'embedding tensors',
  logits: 'logits',
  residual: 'residual-stream tensors',
}

/**
 * Short, factual "where to optimise" lines, computed from the numbers. Each line is a pure
 * consequence of the formulas: attention probs ∝ B·H·T², logits ∝ B·T·V, every activation ∝ B,
 * optimizer = 2× weights.
 */
export function memoryInsights(input: MemoryInput, report: MemoryReport): string[] {
  const out: string[] = []
  const a = report.activations
  const pct = (x: number, of: number) => `${Math.round((x / of) * 100)}%`
  const { batch_size: B } = input.hp

  if (report.mode === 'forward') {
    if (a.total > 0) {
      out.push(`Without gradients nothing is kept for backward: only the biggest live set counts (${formatBytes(a.total)}), so activations stay small.`)
      const top = MEMORY_CATEGORIES.reduce((x, c) => (a.byCategory[c] > a.byCategory[x] ? c : x))
      out.push(
        `${pct(a.byCategory[top], a.total)} of the peak is ${CATEGORY_TERM[top]} (${formatBytes(a.byCategory[top])}); it grows with ${CATEGORY_SCALING[top]}.`,
      )
    }
  } else if (a.total > 0) {
    const probs = a.byCategory.attn_probs
    if (probs > 0) {
      const layers = a.tensors.filter((t) => t.counted && t.category === 'attn_probs').length
      out.push(
        `Attention probs are ${pct(probs, a.total)} of activations (${formatBytes(probs)}${layers > 1 ? `, ${layers} layers` : ''}). They are B·H·T² per layer: doubling the context quadruples them; fewer heads or a smaller batch shrinks them linearly (Q/K/V stay B·T·d). Fused (Flash) attention avoids storing them.`,
      )
    }
    const logits = a.byCategory.logits
    if (logits > 0) out.push(`Logits / loss are ${pct(logits, a.total)} of activations (${formatBytes(logits)}): B·T·V grows with the vocabulary and the context, not with depth.`)
    if (!report.checkpointing) {
      const ck = estimateWith(input, { checkpointing: true })
      const saved = a.total - ck.activations.total
      if (saved > 0) out.push(`Activation checkpointing would save ${formatBytes(saved)} (activations ${formatBytes(a.total)} → ${formatBytes(ck.activations.total)}), at the cost of re-running each block's forward during backward.`)
    } else {
      const full = estimateWith(input, { checkpointing: false })
      const saved = full.activations.total - a.total
      if (saved > 0) out.push(`Checkpointing saves ${formatBytes(saved)} here: only block inputs (B·T·d each) plus one block at a time are kept; the cost is an extra forward per block.`)
    }
    if (B > 1) out.push(`Activations scale linearly with batch size: B = ${Math.floor(B / 2)} would need about ${formatBytes(a.total * (Math.floor(B / 2) / B))} instead of ${formatBytes(a.total)}.`)
  }
  if (report.mode === 'train' && report.P > 0)
    out.push(`AdamW keeps m and v per parameter: optimizer = 2 × weights (${formatBytes(report.byComponent.optimizer)}), so weights + gradients + optimizer = 4·P·bytes.`)
  if (input.hp.dtype === 'fp32') {
    const half = estimateWith(input, { hp: { dtype: 'bf16' } })
    out.push(`bf16 would take the total from ${formatBytes(report.total)} to ${formatBytes(half.total)} (2 bytes instead of 4; int64 token ids stay 8 bytes).`)
  }
  return out
}
