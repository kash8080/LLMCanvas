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
    overview: 'Undoes Split Heads: … × H × T × d_head → … × T × (H · d_head), concatenating the heads back into one d_model vector per token.',
    pointsToRemember: ['Only a transpose + reshape: no parameters.', 'The output projection that follows mixes information across heads.'],
    formula: 'x.transpose(-3, -2).reshape(…, T, H · d_head)',
    cs336Ref: 'MultiHeadSelfAttention.py',
  },
}
