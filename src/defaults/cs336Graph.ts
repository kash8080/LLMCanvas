// The default canvas: CS336 TransformerLM with 2 blocks, laid out flat (Phase 3 wraps blocks in groups).
//
// Layout: the residual stream runs straight down column R; each sub-layer branches out to the
// right (columns Q/K/V, C) and comes back into an Add on column R. Data Batch sits on a far-right
// lane so its `targets` edge drops straight down to Cross-Entropy without crossing anything.
import { createPartNode, PART_WIDTH } from '../canvas/nodeFactory'
import type { AppNode } from '../canvas/types'
import { DEFAULT_HYPERPARAMS } from '../engine/hyperparams'
import type { ParamValue } from '../engine/types'
import { DOC_VERSION, type CanvasDocument, type SavedEdge } from '../store/persistence'

const ROW = 110
const R = 0 // residual stream column (centre x)
const C = 420 // branch centre column
const Q = 190
const K = 420
const V = 650
const LANE = 900 // data batch / loss lane
export const NUM_DEFAULT_BLOCKS = 2
const BLOCK_ROWS = 14

export function cs336Document(): CanvasDocument {
  const nodes: AppNode[] = []
  const edges: SavedEdge[] = []

  const part = (id: string, type: string, cx: number, row: number, title?: string, params?: Record<string, ParamValue>) => {
    const node = createPartNode(type, { x: cx - PART_WIDTH / 2, y: row * ROW }, id, title)
    if (params) node.data.params = { ...node.data.params, ...params }
    nodes.push(node)
  }
  const edge = (source: string, sourceHandle: string, target: string, targetHandle: string) => {
    edges.push({ id: `e:${source}.${sourceHandle}->${target}.${targetHandle}`, source, sourceHandle, target, targetHandle })
  }

  nodes.push({
    id: 'welcome',
    type: 'sticky',
    position: { x: -520, y: -ROW - 60 },
    width: 340,
    height: 230,
    data: {
      text:
        'Welcome to LLM Canvas!\n\n' +
        'This is the CS336 TransformerLM (2 blocks). Shapes flow top → bottom.\n' +
        '• Click a part to see/edit its params (🔗 = follows a global hyperparam).\n' +
        '• Click a port dot for its shape and size.\n' +
        '• Drag from a green output to a blue input (or onto a part) to connect.\n' +
        '• Hyperparams ▾ in the toolbar changes the whole model.',
      bgColor: '#fef08a',
      textColor: '#1f2937',
      fontSize: 13,
    },
  })

  part('data', 'data_batch', LANE, -1)
  part('embed', 'embedding', R, 0, 'token_embeddings')
  edge('data', 'input_ids', 'embed', 'in')

  let residual = 'embed'
  for (let b = 1; b <= NUM_DEFAULT_BLOCKS; b++) {
    const p = `b${b}.`
    const r0 = 1 + (b - 1) * BLOCK_ROWS
    nodes.push({
      id: `${p}label`,
      type: 'textbox',
      position: { x: -360, y: r0 * ROW },
      width: 220,
      height: 60,
      data: { text: `Transformer Block ${b}`, bgColor: 'transparent', textColor: '#475569', fontSize: 20 },
    })

    // --- attention sub-layer: x + MHA(ln1(x)) ---
    part(`${p}ln1`, 'rmsnorm', C, r0, 'ln1')
    edge(residual, 'out', `${p}ln1`, 'in')
    for (const [name, cx] of [['q', Q], ['k', K], ['v', V]] as const) {
      part(`${p}${name}_proj`, 'linear', cx, r0 + 1, `${name}_proj`)
      part(`${p}split_${name}`, 'split_heads', cx, r0 + 2, `split ${name}`)
      edge(`${p}ln1`, 'out', `${p}${name}_proj`, 'in')
      edge(`${p}${name}_proj`, 'out', `${p}split_${name}`, 'in')
    }
    part(`${p}rope_q`, 'rope', Q, r0 + 3, 'RoPE q')
    part(`${p}rope_k`, 'rope', K, r0 + 3, 'RoPE k')
    edge(`${p}split_q`, 'out', `${p}rope_q`, 'in')
    edge(`${p}split_k`, 'out', `${p}rope_k`, 'in')
    part(`${p}sdpa`, 'sdpa', C, r0 + 4, 'attention')
    edge(`${p}rope_q`, 'out', `${p}sdpa`, 'q')
    edge(`${p}rope_k`, 'out', `${p}sdpa`, 'k')
    edge(`${p}split_v`, 'out', `${p}sdpa`, 'v')
    part(`${p}merge`, 'merge_heads', C, r0 + 5, 'merge heads')
    edge(`${p}sdpa`, 'out', `${p}merge`, 'in')
    part(`${p}output_proj`, 'linear', C, r0 + 6, 'output_proj')
    edge(`${p}merge`, 'out', `${p}output_proj`, 'in')
    part(`${p}add1`, 'add', R, r0 + 7, 'x + attn')
    edge(residual, 'out', `${p}add1`, 'a')
    edge(`${p}output_proj`, 'out', `${p}add1`, 'b')

    // --- FFN sub-layer: x + SwiGLU(ln2(x)) = x + w2(SiLU(w1 x) ⊙ w3 x) ---
    part(`${p}ln2`, 'rmsnorm', C, r0 + 8, 'ln2')
    edge(`${p}add1`, 'out', `${p}ln2`, 'in')
    part(`${p}w1`, 'linear', C - 115, r0 + 9, 'w1', { out_features: { bind: 'd_ff' } })
    part(`${p}w3`, 'linear', C + 115, r0 + 9, 'w3', { out_features: { bind: 'd_ff' } })
    edge(`${p}ln2`, 'out', `${p}w1`, 'in')
    edge(`${p}ln2`, 'out', `${p}w3`, 'in')
    part(`${p}silu`, 'silu', C - 115, r0 + 10, 'silu')
    edge(`${p}w1`, 'out', `${p}silu`, 'in')
    part(`${p}gate`, 'multiply', C, r0 + 11, 'gate')
    edge(`${p}silu`, 'out', `${p}gate`, 'a')
    edge(`${p}w3`, 'out', `${p}gate`, 'b')
    part(`${p}w2`, 'linear', C, r0 + 12, 'w2', { in_features: { bind: 'd_ff' } })
    edge(`${p}gate`, 'out', `${p}w2`, 'in')
    part(`${p}add2`, 'add', R, r0 + 13, 'x + ffn')
    edge(`${p}add1`, 'out', `${p}add2`, 'a')
    edge(`${p}w2`, 'out', `${p}add2`, 'b')

    residual = `${p}add2`
  }

  const r = 1 + NUM_DEFAULT_BLOCKS * BLOCK_ROWS
  part('ln_final', 'rmsnorm', R, r, 'ln_final')
  edge(residual, 'out', 'ln_final', 'in')
  part('lm_head', 'linear', R, r + 1, 'lm_head', { out_features: { bind: 'vocab_size' } })
  edge('ln_final', 'out', 'lm_head', 'in')
  part('logits', 'logits', R, r + 2)
  edge('lm_head', 'out', 'logits', 'in')
  part('xent', 'cross_entropy', LANE, r + 3)
  edge('logits', 'out', 'xent', 'logits')
  edge('data', 'targets', 'xent', 'targets')
  part('loss', 'loss', LANE, r + 4)
  edge('xent', 'out', 'loss', 'in')

  return { app: 'llm-canvas', version: DOC_VERSION, hyperparams: { ...DEFAULT_HYPERPARAMS }, nodes, edges }
}
