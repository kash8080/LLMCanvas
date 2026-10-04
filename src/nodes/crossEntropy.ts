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
    overview: 'Average negative log-probability of the correct next token: −log softmax(logits)[target], averaged over all B·T positions.',
    pointsToRemember: [
      'Computed stably as log-sum-exp after subtracting the max logit.',
      'Output is a single scalar (mean over batch and sequence).',
      'A uniform guess gives loss = ln(vocab_size) ≈ 9.21 for V = 10000.',
    ],
    formula: 'ℓ = mean_{b,t} [ logsumexp(o_{b,t}) − o_{b,t}[x_{b,t+1}] ]',
    cs336Ref: 'CrossEntropy.py',
  },
}
