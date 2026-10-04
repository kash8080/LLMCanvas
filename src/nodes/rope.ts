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
    overview:
      'Rotary positional embedding: tells attention where each token is by rotating every (even, odd) pair of a query/key vector by an angle proportional to the token’s position m. Since both q and k are rotated, their dot product only depends on how far apart the two tokens are. It runs per head, after Split Heads.',
    roles: {
      'RoPE q': 'RoPE q: rotates the queries of every head.',
      'RoPE k': 'RoPE k: rotates the keys of every head (same angles as the queries).',
    },
    formula: [
      'θᵢ = Θ^(−2i / d_k),   i = 0 … d_k/2 − 1',
      'x′₂ᵢ   = x₂ᵢ·cos(mθᵢ) − x₂ᵢ₊₁·sin(mθᵢ)',
      'x′₂ᵢ₊₁ = x₂ᵢ·sin(mθᵢ) + x₂ᵢ₊₁·cos(mθᵢ)',
      'm = token position',
    ],
    pointsToRemember: [
      'RoPE has no learnable params: the cos/sin tables (max_seq_len × d_k/2 each) are non-persistent buffers.',
      'Only Q and K are rotated, not V: position changes who attends to whom, not what gets passed along.',
      'Pairs are interleaved (dims 0&1, 2&3, …), not first half / second half.',
      'Relative position for free: R(m)ᵀR(n) = R(n − m), so q·k depends only on n − m.',
      'd_k must be even, and the sequence length must be ≤ max_seq_len (= context_length).',
      'Low pairs rotate fast, high pairs slowly; Θ = 10000 sets how slow the slowest one is.',
    ],
    paramHelp: {
      head_dim: 'd_k = d_head: size of each query/key vector. Rotated two dims at a time, so it must be even.',
      theta: 'Θ: base of the frequencies θᵢ = Θ^(−2i/d_k). Larger Θ → slower rotations → longer wavelengths.',
      max_seq_len: 'Positions the cos/sin tables are precomputed for (= context_length). Longer sequences are an error.',
    },
    cs336Ref: { file: 'RotaryPositionalEmbedding.py', symbol: 'RotaryPositionalEmbedding.forward' },
  },
}
