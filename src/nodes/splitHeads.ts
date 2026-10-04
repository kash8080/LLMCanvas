import { num, paramDim } from '../engine/resolve'
import type { NodeDef } from '../engine/types'
import { expectFloat, last, NO_PARAMS, NOTHING_SAVED } from './common'

export const splitHeads: NodeDef = {
  type: 'split_heads',
  label: 'Split Heads',
  category: 'attention',
  inputs: [{ id: 'in', label: 'x' }],
  outputs: [{ id: 'out', label: 'heads' }],
  params: [{ key: 'num_heads', label: 'num_heads', kind: 'int', default: { bind: 'num_heads' }, min: 1, help: 'Number of heads H; the last dim is cut into H pieces of d_head.' }],
  infer: ({ inputs: [x], p }) => {
    const bad = expectFloat(x, 'Split Heads input', 2)
    if (bad) return { outputs: [], errors: [bad] }
    const H = num(p, 'num_heads')
    const d = last(x)!
    if (d.size % H !== 0)
      return { outputs: [], errors: [`${d.label ?? 'last dim'} = ${d.size} is not divisible by num_heads = ${H}`] }
    const T = x.dims[x.dims.length - 2]
    const dHead = { size: d.size / H, ...(d.label === 'd_model' && p.num_heads.label ? { label: 'd_head' } : {}) }
    return { outputs: [{ dims: [...x.dims.slice(0, -2), paramDim(p, 'num_heads'), T, dHead], dtype: 'float' }], errors: [] }
  },
  paramCount: NO_PARAMS,
  savedForBackward: NOTHING_SAVED,
  docs: {
    overview: 'Reshapes … × T × d_model into … × H × T × d_head (reshape, then swap the T and H axes) so each head attends independently.',
    pointsToRemember: [
      'No parameters and no saved activations: it is only a view/reshape.',
      `d_model must be divisible by num_heads; d_head = d_model / H.`,
    ],
    formula: 'x.reshape(…, T, H, d_head).transpose(-3, -2)',
    cs336Ref: 'MultiHeadSelfAttention.py',
  },
}
