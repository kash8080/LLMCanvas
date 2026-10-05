// Group templates (PLAN.md §2.3): build a Transformer Block / MHA / SwiGLU / non-gated FFN frame with all its
// children (parts bound to the global hyperparams, proxies) and internal edges.
//
// Child positions are relative to the frame. Every frame has a fixed size: big enough for its
// children expanded, and the same size when collapsed to a card (no re-layout of neighbours).
//
//   Transformer Block            MHA                                 SwiGLU
//   ┌───────────────────────┐    ┌───────────────────────────┐       ┌──────────────┐
//   │ in                    │    │            in             │       │      in      │
//   │ │        ln1          │    │ q_proj   k_proj   v_proj  │       │  w1      w3  │
//   │ │   ┌─── MHA ────┐    │    │ split q  split k  split v │       │ silu     │   │
//   │ │   └────────────┘    │    │ RoPE q   RoPE k     │     │       │    gate      │
//   │ add1 ←──┘             │    │         attention         │       │     w2       │
//   │ │        ln2          │    │        merge heads        │       │     out      │
//   │ │   ┌── SwiGLU ──┐    │    │        output_proj        │       └──────────────┘
//   │ │   └────────────┘    │    │            out            │
//   │ add2 ←──┘             │    └───────────────────────────┘
//   │ out                   │
//   └───────────────────────┘    FFN (non-gated): in → w1 → silu → w2 → out (one column)
import type { XYPosition } from '@xyflow/react'
import { GROUP_INPUT, GROUP_OUTPUT } from '../engine/groups'
import type { ParamValue } from '../engine/types'
import { GROUP_DEFS, type GroupType } from '../nodes/groups'
import type { SavedEdge } from '../store/persistence'
import { createPartNode, defaultSize, newId, PART_WIDTH, PROXY_HEIGHT, PROXY_WIDTH } from './nodeFactory'
import type { AppNode, GroupNode, PaletteItemId } from './types'

/** Height reserved for the frame's header bar. */
export const GROUP_HEADER = 40
const PAD = 20
const ROW = 110
const FIRST_ROW = 124 // y of the first part row (below the header and the input proxy)
const PROXY_Y = 48
const COL_GAP = 24
const PART_H = 64 // single-input part
const MULTI_H = 74 // part with 2–3 inputs (labels above the ports)
const V_GAP = 44 // vertical gap for an edge + its shape label

// ---- MHA: q / k / v columns ----
const MHA_COL = (i: number) => PAD + PART_WIDTH / 2 + i * (PART_WIDTH + COL_GAP)
const MHA_W = 2 * PAD + 3 * PART_WIDTH + 2 * COL_GAP
const MHA_LAST_ROW = FIRST_ROW + 5 * ROW
const MHA_H = MHA_LAST_ROW + PART_H + 40 + PROXY_HEIGHT + PAD

// ---- SwiGLU: w1 / w3 columns ----
const FFN_GAP = 40
const FFN_COL = (i: number) => PAD + PART_WIDTH / 2 + i * (PART_WIDTH + FFN_GAP)
const FFN_W = 2 * PAD + 2 * PART_WIDTH + FFN_GAP
const FFN_LAST_ROW = FIRST_ROW + 3 * ROW
const FFN_H = FFN_LAST_ROW + PART_H + 40 + PROXY_HEIGHT + PAD

// ---- FFN (non-gated, CS336 SiLU.py): one column w1 → act → w2 ----
const PLAIN_W = 2 * PAD + PART_WIDTH + 2 * FFN_GAP
const PLAIN_LAST_ROW = FIRST_ROW + 2 * ROW
const PLAIN_H = PLAIN_LAST_ROW + PART_H + 40 + PROXY_HEIGHT + PAD

// ---- Transformer Block: residual column on the left, sub-layers to the right ----
const BLK_RES = PAD + PART_WIDTH / 2 // residual stream column (centre x)
const BLK_BRANCH_X = PAD + PART_WIDTH + 56 // left edge of the MHA frame
const BLK_BRANCH = BLK_BRANCH_X + MHA_W / 2 // branch column (centre x)
const BLK_LN1 = FIRST_ROW
const BLK_MHA = BLK_LN1 + PART_H + V_GAP
const BLK_ADD1 = BLK_MHA + MHA_H + V_GAP
const BLK_LN2 = BLK_ADD1 + MULTI_H + 40
const BLK_FFN = BLK_LN2 + PART_H + V_GAP
const BLK_ADD2 = BLK_FFN + FFN_H + V_GAP
const BLK_OUT = BLK_ADD2 + MULTI_H + 40
const BLK_W = BLK_BRANCH_X + MHA_W + PAD
const BLK_H = BLK_OUT + PROXY_HEIGHT + PAD

/** Fixed frame size, and where the outer ports sit (fraction of the width, over the proxies). */
export const GROUP_LAYOUT: Record<GroupType, { width: number; height: number; portX: number }> = {
  transformer_block: { width: BLK_W, height: BLK_H, portX: BLK_RES / BLK_W },
  mha: { width: MHA_W, height: MHA_H, portX: 0.5 },
  swiglu: { width: FFN_W, height: FFN_H, portX: 0.5 },
  ffn: { width: PLAIN_W, height: PLAIN_H, portX: 0.5 },
}

export interface Built {
  nodes: AppNode[]
  edges: SavedEdge[]
}

export interface GroupPlacement {
  /** Id of the group node. */
  id: string
  /** Prefix for child ids (nested groups share their parent's prefix: `b1.q_proj`, `b1.attn`). */
  prefix: string
  position: XYPosition
  parentId?: string
  title?: string
}

export function edgeId(source: string, sourceHandle: string, target: string, targetHandle: string): string {
  return `e:${source}.${sourceHandle}->${target}.${targetHandle}`
}

/** Append a group of `type` (frame first, then its children, nested groups recursively) to `out`. */
export function buildGroup(type: GroupType, at: GroupPlacement, out: Built): void {
  const { id, prefix } = at
  const { width, height } = GROUP_LAYOUT[type]
  const frame: GroupNode = {
    id,
    type: 'group',
    position: at.position,
    width,
    height,
    data: { groupType: type, mode: 'auto', ...(at.title ? { title: at.title } : {}) },
    ...(at.parentId ? { parentId: at.parentId, extent: 'parent' as const } : {}),
  }
  out.nodes.push(frame)

  const child = { parentId: id, extent: 'parent' as const }
  /** A part centred on column x `cx` at row y. */
  const part = (localId: string, partType: string, cx: number, y: number, title?: string, params?: Record<string, ParamValue>) => {
    const node = createPartNode(partType, { x: cx - PART_WIDTH / 2, y }, prefix + localId, title)
    if (params) node.data.params = { ...node.data.params, ...params }
    out.nodes.push({ ...node, ...child })
  }
  const proxy = (kind: 'in' | 'out', cx: number, y: number) => {
    const node = createPartNode(kind === 'in' ? GROUP_INPUT : GROUP_OUTPUT, { x: cx - PROXY_WIDTH / 2, y }, `${id}.${kind}`)
    out.nodes.push({ ...node, ...child })
  }
  const edge = (source: string, sourceHandle: string, target: string, targetHandle: string) =>
    out.edges.push({ id: edgeId(source, sourceHandle, target, targetHandle), source, sourceHandle, target, targetHandle })
  const p = (localId: string) => prefix + localId
  const IN = `${id}.in`
  const OUT = `${id}.out`

  if (type === 'mha') {
    const center = MHA_COL(1)
    proxy('in', center, PROXY_Y)
    const row = (i: number) => FIRST_ROW + i * ROW
    for (const [name, col] of [['q', 0], ['k', 1], ['v', 2]] as const) {
      part(`${name}_proj`, 'linear', MHA_COL(col), row(0), `${name}_proj`)
      part(`split_${name}`, 'split_heads', MHA_COL(col), row(1), `split ${name}`)
      edge(IN, 'out', p(`${name}_proj`), 'in')
      edge(p(`${name}_proj`), 'out', p(`split_${name}`), 'in')
    }
    part('rope_q', 'rope', MHA_COL(0), row(2), 'RoPE q')
    part('rope_k', 'rope', MHA_COL(1), row(2), 'RoPE k')
    edge(p('split_q'), 'out', p('rope_q'), 'in')
    edge(p('split_k'), 'out', p('rope_k'), 'in')
    part('sdpa', 'sdpa', center, row(3), 'attention')
    edge(p('rope_q'), 'out', p('sdpa'), 'q')
    edge(p('rope_k'), 'out', p('sdpa'), 'k')
    edge(p('split_v'), 'out', p('sdpa'), 'v')
    part('merge', 'merge_heads', center, row(4), 'merge heads')
    edge(p('sdpa'), 'out', p('merge'), 'in')
    part('output_proj', 'linear', center, row(5), 'output_proj')
    edge(p('merge'), 'out', p('output_proj'), 'in')
    proxy('out', center, MHA_LAST_ROW + PART_H + 40)
    edge(p('output_proj'), 'out', OUT, 'in')
    return
  }

  if (type === 'swiglu') {
    const center = FFN_W / 2
    proxy('in', center, PROXY_Y)
    const row = (i: number) => FIRST_ROW + i * ROW
    part('w1', 'linear', FFN_COL(0), row(0), 'w1', { out_features: { bind: 'd_ff' } })
    part('w3', 'linear', FFN_COL(1), row(0), 'w3', { out_features: { bind: 'd_ff' } })
    edge(IN, 'out', p('w1'), 'in')
    edge(IN, 'out', p('w3'), 'in')
    part('silu', 'silu', FFN_COL(0), row(1), 'silu')
    edge(p('w1'), 'out', p('silu'), 'in')
    part('gate', 'multiply', center, row(2), 'gate')
    edge(p('silu'), 'out', p('gate'), 'a')
    edge(p('w3'), 'out', p('gate'), 'b')
    part('w2', 'linear', center, row(3), 'w2', { in_features: { bind: 'd_ff' } })
    edge(p('gate'), 'out', p('w2'), 'in')
    proxy('out', center, FFN_LAST_ROW + PART_H + 40)
    edge(p('w2'), 'out', OUT, 'in')
    return
  }

  if (type === 'ffn') {
    // Non-gated FFN (CS336 SiLU.py): w2(silu(w1 x)); the activation is an ordinary part (swap for GELU / ReLU).
    const center = PLAIN_W / 2
    proxy('in', center, PROXY_Y)
    const row = (i: number) => FIRST_ROW + i * ROW
    part('w1', 'linear', center, row(0), 'w1', { out_features: { bind: 'd_ff' } })
    edge(IN, 'out', p('w1'), 'in')
    part('act', 'silu', center, row(1), 'silu')
    edge(p('w1'), 'out', p('act'), 'in')
    part('w2', 'linear', center, row(2), 'w2', { in_features: { bind: 'd_ff' } })
    edge(p('act'), 'out', p('w2'), 'in')
    proxy('out', center, PLAIN_LAST_ROW + PART_H + 40)
    edge(p('w2'), 'out', OUT, 'in')
    return
  }

  // Transformer Block (pre-norm): y = x + attn(ln1(x)); out = y + ffn(ln2(y))
  proxy('in', BLK_RES, PROXY_Y)
  part('ln1', 'rmsnorm', BLK_BRANCH, BLK_LN1, 'ln1')
  edge(IN, 'out', p('ln1'), 'in')
  buildGroup('mha', { id: p('attn'), prefix, position: { x: BLK_BRANCH_X, y: BLK_MHA }, parentId: id, title: 'attn' }, out)
  edge(p('ln1'), 'out', p('attn'), 'in')
  part('add1', 'add', BLK_RES, BLK_ADD1, 'x + attn')
  edge(IN, 'out', p('add1'), 'a')
  edge(p('attn'), 'out', p('add1'), 'b')

  part('ln2', 'rmsnorm', BLK_BRANCH, BLK_LN2, 'ln2')
  edge(p('add1'), 'out', p('ln2'), 'in')
  buildGroup('swiglu', { id: p('ffn'), prefix, position: { x: BLK_BRANCH - FFN_W / 2, y: BLK_FFN }, parentId: id, title: 'ffn' }, out)
  edge(p('ln2'), 'out', p('ffn'), 'in')
  part('add2', 'add', BLK_RES, BLK_ADD2, 'x + ffn')
  edge(p('add1'), 'out', p('add2'), 'a')
  edge(p('ffn'), 'out', p('add2'), 'b')
  proxy('out', BLK_RES, BLK_OUT)
  edge(p('add2'), 'out', OUT, 'in')
}

/**
 * A fresh group from the palette / quick-add, with unique ids. New Transformer Blocks are numbered ("Block 3").
 * `parentId`: create it inside that group (a sub-layer inside a Transformer Block); `position` is then relative to it.
 */
export function instantiateGroup(type: GroupType, position: XYPosition, existing: AppNode[], parentId?: string): Built {
  const id = newId(type)
  const out: Built = { nodes: [], edges: [] }
  const title = type === 'transformer_block' ? nextBlockTitle(existing) : undefined
  buildGroup(type, { id, prefix: `${id}.`, position, title, ...(parentId ? { parentId } : {}) }, out)
  return out
}

const isBlock = (n: AppNode): n is GroupNode => n.type === 'group' && n.data.groupType === 'transformer_block'

/** num_layers is derived: the number of Transformer Block groups on the canvas (PLAN.md D1). */
export function numLayers(nodes: AppNode[]): number {
  return nodes.filter(isBlock).length
}

/** "Block N" with N one past both the block count and the highest existing "Block k" title. */
export function nextBlockTitle(nodes: AppNode[]): string {
  let n = numLayers(nodes)
  for (const node of nodes) {
    const m = isBlock(node) ? /^Block (\d+)$/.exec(node.data.title ?? '') : null
    if (m) n = Math.max(n, Number(m[1]))
  }
  return `Block ${n + 1}`
}

/** Size of a palette item (used to centre a dropped item on the cursor). */
export function paletteItemSize(item: PaletteItemId): { width: number; height: number } {
  if (item.startsWith('group:')) {
    const { width, height } = GROUP_LAYOUT[item.slice(6) as GroupType]
    return { width, height }
  }
  return defaultSize(item)
}

export function isGroupItem(item: PaletteItemId): item is `group:${GroupType}` {
  return item.startsWith('group:') && item.slice(6) in GROUP_DEFS
}
