// The default canvas: CS336 TransformerLM with 2 Transformer Block groups (each with nested MHA and
// SwiGLU groups, built by src/canvas/groupTemplates.ts).
//
// Layout: the residual stream runs straight down column x = 0 (Embedding, the blocks' in/out ports,
// ln_final, lm_head, Logits). Inside a block the residual stream stays on the block's left column
// and the sub-layers sit to the right. Data Batch, Cross-Entropy and Loss sit on a lane right of
// the blocks so the `targets` edge drops straight down without crossing anything.
import { buildGroup, edgeId, GROUP_LAYOUT, type Built } from '../canvas/groupTemplates'
import { createPartNode, PART_WIDTH } from '../canvas/nodeFactory'
import { DEFAULT_HYPERPARAMS } from '../engine/hyperparams'
import type { ParamValue } from '../engine/types'
import { DOC_VERSION, type CanvasDocument } from '../store/persistence'

const ROW = 110
const R = 0 // residual stream column (centre x)
export const NUM_DEFAULT_BLOCKS = 2
const BLOCK = GROUP_LAYOUT.transformer_block
const BLOCK_X = R - BLOCK.portX * BLOCK.width // block's residual column lines up with R
const BLOCK_GAP = 60
const LANE = BLOCK_X + BLOCK.width + 180 // data batch / loss lane (centre x)

export function cs336Document(): CanvasDocument {
  const g: Built = { nodes: [], edges: [] }

  const part = (id: string, type: string, cx: number, y: number, title?: string, params?: Record<string, ParamValue>) => {
    const node = createPartNode(type, { x: cx - PART_WIDTH / 2, y }, id, title)
    if (params) node.data.params = { ...node.data.params, ...params }
    g.nodes.push(node)
  }
  const edge = (source: string, sourceHandle: string, target: string, targetHandle: string) => {
    g.edges.push({ id: edgeId(source, sourceHandle, target, targetHandle), source, sourceHandle, target, targetHandle })
  }

  g.nodes.push({
    id: 'welcome',
    type: 'sticky',
    position: { x: BLOCK_X - 480, y: -ROW - 60 },
    width: 400,
    height: 290,
    data: {
      text:
        'Welcome to LLM Canvas!\n\n' +
        'This is the CS336 TransformerLM with 2 Transformer Blocks. Shapes flow top → bottom.\n' +
        '• Zoom in (pinch / ⌘+scroll) to open the blocks, and further to open attention and SwiGLU. Zoom out to see each block as one card.\n' +
        '• Click a part to see/edit its params (🔗 = follows a global hyperparam).\n' +
        '• Click a port dot for its shape and size.\n' +
        '• Drag from a green output to a blue input (or onto a part) to connect.\n' +
        '• Hyperparams ▾ in the toolbar changes the whole model.',
      bgColor: '#fef08a',
      textColor: '#1f2937',
      fontSize: 14,
    },
  })

  part('data', 'data_batch', LANE, -ROW)
  part('embed', 'embedding', R, 0, 'token_embeddings')
  edge('data', 'input_ids', 'embed', 'in')

  let y = ROW
  let residual = 'embed'
  for (let b = 1; b <= NUM_DEFAULT_BLOCKS; b++) {
    const id = `b${b}`
    buildGroup('transformer_block', { id, prefix: `${id}.`, position: { x: BLOCK_X, y }, title: `Block ${b}` }, g)
    edge(residual, 'out', id, 'in')
    residual = id
    y += BLOCK.height + BLOCK_GAP
  }

  part('ln_final', 'rmsnorm', R, y, 'ln_final')
  edge(residual, 'out', 'ln_final', 'in')
  part('lm_head', 'linear', R, y + ROW, 'lm_head', { out_features: { bind: 'vocab_size' } })
  edge('ln_final', 'out', 'lm_head', 'in')
  part('logits', 'logits', R, y + 2 * ROW)
  edge('lm_head', 'out', 'logits', 'in')
  part('xent', 'cross_entropy', LANE, y + 3 * ROW)
  edge('logits', 'out', 'xent', 'logits')
  edge('data', 'targets', 'xent', 'targets')
  part('loss', 'loss', LANE, y + 4 * ROW)
  edge('xent', 'out', 'loss', 'in')

  return { app: 'llm-canvas', version: DOC_VERSION, hyperparams: { ...DEFAULT_HYPERPARAMS }, nodes: g.nodes, edges: g.edges }
}
