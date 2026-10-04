import { num } from '../engine/resolve'
import { formatDim } from '../engine/shape'
import type { NodeDef } from '../engine/types'
import { expectFloat, last, mismatch, NO_PARAMS, NOTHING_SAVED } from './common'

export const rope: NodeDef = {
  type: 'rope',
  label: 'RoPE',
  category: 'attention',
  inputs: [{ id: 'in', label: 'x' }],
  outputs: [{ id: 'out', label: 'out' }],
  params: [
    { key: 'head_dim', label: 'head_dim (d_k)', kind: 'int', default: { bind: 'd_head' }, min: 2, help: 'Size of each query/key vector; rotated in pairs, so it must be even.' },
    { key: 'theta', label: 'theta (Θ)', kind: 'float', default: { bind: 'rope_theta' }, min: 1, help: 'Base of the rotation frequencies: freq_i = Θ^(−2i/d_k).' },
    { key: 'max_seq_len', label: 'max_seq_len', kind: 'int', default: { bind: 'context_length' }, min: 1, help: 'Number of positions the cos/sin tables are precomputed for.' },
  ],
  infer: ({ inputs: [x], p }) => {
    const bad = expectFloat(x, 'RoPE input', 2)
    if (bad) return { outputs: [], errors: [bad] }
    const errors: string[] = []
    const dk = num(p, 'head_dim')
    if (!Number.isInteger(dk) || dk % 2 !== 0) errors.push(`RoPE needs an even head_dim, got ${dk}`)
    if (last(x)!.size !== dk) errors.push(mismatch('RoPE.head_dim', dk, last(x)!))
    const T = x.dims[x.dims.length - 2]
    const maxLen = num(p, 'max_seq_len')
    if (T.size > maxLen) errors.push(`sequence length ${formatDim(T)} > RoPE max_seq_len = ${maxLen} (context_length)`)
    return errors.length > 0 ? { outputs: [], errors } : { outputs: [structuredClone(x)], errors: [] }
  },
  // cos/sin tables (max_seq_len × head_dim/2 each) are non-persistent buffers, not parameters.
  paramCount: NO_PARAMS,
  savedForBackward: NOTHING_SAVED,
  docs: {
    overview: 'Rotary positional embedding: rotates each (even, odd) pair of a query/key vector by an angle that depends on the token position. Applied to Q and K only, never V.',
    pointsToRemember: [
      'No learned parameters; cos/sin tables are precomputed buffers of max_seq_len × d_k/2.',
      'Relative position falls out of q·k because rotations compose: R(m)ᵀR(n) = R(n−m).',
      'Sequence length must not exceed max_seq_len (= context_length).',
    ],
    formula: "x'_{2i} = x_{2i}·cos(mθ_i) − x_{2i+1}·sin(mθ_i),  x'_{2i+1} = x_{2i}·sin(mθ_i) + x_{2i+1}·cos(mθ_i)",
    cs336Ref: 'RotaryPositionalEmbedding.py',
  },
}
