import type { NodeDef } from '../engine/types'
import { expectFloat, NO_PARAMS, NOTHING_SAVED } from './common'

export const mergeHeads: NodeDef = {
  type: 'merge_heads',
  label: 'Merge Heads',
  category: 'attention',
  inputs: [{ id: 'in', label: 'heads' }],
  outputs: [{ id: 'out', label: 'out' }],
  params: [],
  infer: ({ inputs: [x] }) => {
    const bad = expectFloat(x, 'Merge Heads input', 3)
    if (bad) return { outputs: [], errors: [bad] }
    const [H, T, dh] = x.dims.slice(-3)
    const merged = { size: H.size * dh.size, ...(H.label === 'H' && dh.label === 'd_head' ? { label: 'd_model' } : {}) }
    return { outputs: [{ dims: [...x.dims.slice(0, -3), T, merged], dtype: 'float' }], errors: [] }
  },
  paramCount: NO_PARAMS,
  savedForBackward: NOTHING_SAVED,
  docs: {
    overview:
      'Undoes Split Heads: moves the head axis back behind the sequence axis and concatenates the H head outputs into one d_model vector per token. The output projection that follows is what actually mixes information between heads.',
    formula: ['B × H × T × d_head → B × T × H × d_head → B × T × d_model', 'x.transpose(−3, −2).reshape(…, T, H · d_head)'],
    pointsToRemember: [
      'Only a transpose + reshape: no parameters and nothing saved for backward.',
      'Concatenation, not averaging: each head keeps its own d_head slice of the output vector.',
      'After the transpose the tensor isn’t contiguous, so reshape has to copy it (a .view() would fail).',
      'H · d_head equals d_model only when d_model divides evenly into heads.',
    ],
    cs336Ref: { file: 'MultiHeadSelfAttention.py', symbol: 'MultiHeadSelfAttention.forward (transpose + reshape)' },
  },
}
