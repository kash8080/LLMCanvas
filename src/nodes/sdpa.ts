import { formatConcrete, formatDim } from '../engine/shape'
import type { NodeDef, Shape } from '../engine/types'
import { expectFloat, last, NO_PARAMS } from './common'

const seq = (s: Shape) => s.dims[s.dims.length - 2]
const lead = (s: Shape) => s.dims.slice(0, -2)

export const sdpa: NodeDef = {
  type: 'sdpa',
  label: 'Scaled Dot-Product Attention',
  category: 'attention',
  inputs: [
    { id: 'q', label: 'q' },
    { id: 'k', label: 'k' },
    { id: 'v', label: 'v' },
  ],
  outputs: [{ id: 'out', label: 'out' }],
  params: [{ key: 'causal', label: 'causal mask', kind: 'bool', default: { value: true }, help: 'Each position may only attend to itself and earlier positions.' }],
  infer: ({ inputs: [q, k, v], p }) => {
    const bad = expectFloat(q, 'q', 2) ?? expectFloat(k, 'k', 2) ?? expectFloat(v, 'v', 2)
    if (bad) return { outputs: [], errors: [bad] }
    const errors: string[] = []
    const leadQ = formatConcrete({ dims: lead(q), dtype: 'float' })
    if (leadQ !== formatConcrete({ dims: lead(k), dtype: 'float' }) || leadQ !== formatConcrete({ dims: lead(v), dtype: 'float' }))
      errors.push(`q, k, v batch/head dims differ: ${formatConcrete(q)}, ${formatConcrete(k)}, ${formatConcrete(v)}`)
    if (last(q)!.size !== last(k)!.size) errors.push(`q and k must have the same d_k: ${formatDim(last(q)!)} vs ${formatDim(last(k)!)}`)
    if (seq(k).size !== seq(v).size) errors.push(`k and v must have the same number of keys: ${formatDim(seq(k))} vs ${formatDim(seq(v))}`)
    if (p.causal.value === true && seq(q).size !== seq(k).size)
      errors.push(`causal mask needs as many queries as keys: ${formatDim(seq(q))} vs ${formatDim(seq(k))}`)
    if (errors.length > 0) return { outputs: [], errors }
    return { outputs: [{ dims: [...q.dims.slice(0, -1), last(v)!], dtype: 'float' }], errors: [] }
  },
  paramCount: NO_PARAMS,
  savedForBackward: ({ inputs: [q, k, v] }) => [
    { name: 'Q', which: 'input', port: 'q', shape: q },
    { name: 'K', which: 'input', port: 'k', shape: k },
    { name: 'V', which: 'input', port: 'v', shape: v },
    { name: 'attention probs', which: 'internal', shape: { dims: [...q.dims.slice(0, -1), seq(k)], dtype: 'float' } },
  ],
  docs: {
    overview: 'softmax(Q Kᵀ / √d_k + mask) V: every query position takes a weighted average of the value vectors, weighted by how well its query matches each key.',
    pointsToRemember: [
      'No parameters — all the weights live in the q/k/v/output projections.',
      'The attention-probability matrix is B × H × T × T: memory grows with T², which is why long contexts are expensive.',
      'Causal mask: position t only sees positions ≤ t (masked scores set to −∞ before softmax).',
    ],
    formula: 'Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V',
    cs336Ref: 'ScaledDotProductAttention.py',
  },
}
