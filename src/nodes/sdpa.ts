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
    overview:
      'For every query position, scores all key positions with a dot product, turns the scores into probabilities with a softmax, and returns the probability-weighted average of the value vectors. Dividing by √d_k keeps the scores from growing with the head size. The causal mask hides future tokens so the model can’t peek at the token it has to predict.',
    formula: ['Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + M) V', 'Mᵢⱼ = 0 if j ≤ i, −∞ otherwise   (causal mask)'],
    pointsToRemember: [
      'No parameters: all of attention’s weights live in the q/k/v/output projections.',
      'Attention probs are B × H × T × T — quadratic in T (32·16·256·256 ≈ 33.5M values per layer with the defaults).',
      'Causal mask prevents attending to future tokens: masked scores are set to −∞, so they get exactly 0 probability.',
      'The softmax subtracts the row max before exp(), so it never overflows.',
      'Without the 1/√d_k scale, dot products grow with d_k and the softmax saturates (tiny gradients).',
      'Backward needs Q, K, V and the attention probabilities.',
      'Generating with a KV cache: past K (after RoPE) and V never change, so each layer keeps them — 2 · B · H · T · d_head values, +4,096 B per new token per sequence here (fp32). Each step is then 1 query × T keys: probs B × H × 1 × T, O(T) work instead of O(T²).',
      'CS336’s Decoding.py has no KV cache: every step re-runs the whole (cropped) prefix through the model.',
    ],
    paramHelp: {
      causal: 'On: query i only sees keys j ≤ i (CS336 always uses it, via a torch.tril mask built in MultiHeadSelfAttention).',
    },
    cs336Ref: { file: 'ScaledDotProductAttention.py', symbol: 'scaled_dot_product_attention' },
  },
}
