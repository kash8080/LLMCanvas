import { formatConcrete } from '../engine/shape'
import type { NodeDef } from '../engine/types'
import { expectFloat, NO_PARAMS } from './common'

export const crossEntropy: NodeDef = {
  type: 'cross_entropy',
  label: 'Cross-Entropy',
  category: 'loss',
  inputs: [
    { id: 'logits', label: 'logits' },
    { id: 'targets', label: 'targets' },
  ],
  outputs: [{ id: 'out', label: 'loss' }],
  params: [],
  infer: ({ inputs: [x, t] }) => {
    const bad = expectFloat(x, 'logits')
    if (bad) return { outputs: [], errors: [bad] }
    if (t.dtype !== 'int64') return { outputs: [], errors: ['targets must be int64 token ids'] }
    const lead = x.dims.slice(0, -1)
    if (lead.length !== t.dims.length || lead.some((d, i) => d.size !== t.dims[i].size))
      return { outputs: [], errors: [`targets ${formatConcrete(t)} must match the logits without the vocab dim (${formatConcrete({ dims: lead, dtype: 'float' })})`] }
    return { outputs: [{ dims: [], dtype: 'float' }], errors: [] }
  },
  paramCount: NO_PARAMS,
  savedForBackward: ({ inputs: [x, t] }) => [
    { name: 'logits', which: 'input', port: 'logits', shape: x },
    { name: 'targets', which: 'input', port: 'targets', shape: t },
  ],
  docs: {
    overview:
      'Measures how surprised the model is by the true next token: the negative log-probability it gave to the target, averaged over all B·T positions. Lower is better, and this single number is what training minimises.',
    formula: ['ℓ = mean_{b,t} [ log Σᵥ exp(o_{b,t,v}) − o_{b,t,target} ]', 'stable form: shift o ← o − max(o) before exp'],
    pointsToRemember: [
      'Works on raw logits: log-softmax via log-sum-exp after subtracting the max logit — never softmax followed by log.',
      'Output is a scalar: the mean over all B·T positions.',
      'Sanity check: a uniform guess gives ℓ = ln(vocab_size) ≈ 9.21 for V = 10000.',
      'targets must be int64 and match the logits’ shape without the vocab dim (B × T).',
      'Perplexity = exp(ℓ).',
      'Backward needs the logits: the gradient is (softmax(o) − one_hot(target)) / (B·T).',
    ],
    cs336Ref: { file: 'CrossEntropy.py', symbol: 'cross_entropy' },
  },
}
