import type { NodeDef } from '../engine/types'
import { expectFloat, NO_PARAMS, NOTHING_SAVED } from './common'

export const logits: NodeDef = {
  type: 'logits',
  label: 'Logits',
  category: 'io',
  inputs: [{ id: 'in', label: 'logits' }],
  outputs: [{ id: 'out', label: 'logits' }],
  params: [],
  infer: ({ inputs: [x] }) => {
    const bad = expectFloat(x, 'Logits input')
    return bad ? { outputs: [], errors: [bad] } : { outputs: [structuredClone(x)], errors: [] }
  },
  paramCount: NO_PARAMS,
  savedForBackward: NOTHING_SAVED,
  docs: {
    overview:
      'Marks the model’s output: one unnormalised score per vocabulary entry at every position (B × T × vocab_size). TransformerLM.forward returns exactly this tensor; a softmax over the last dim would give the next-token distribution. This part is a labelled pass-through.',
    formula: ['p(next = v | tokens ≤ t) = softmax(logits[b, t, :])ᵥ'],
    pointsToRemember: [
      'The model returns raw logits; the softmax is folded into the cross-entropy loss (more stable).',
      'B × T × V is one of the largest activations: 32·256·10000 ≈ 81.9M values with the defaults.',
      'Every position predicts its own next token; at generation time only the last position is used.',
      'No parameters: the scores come from lm_head.',
    ],
    cs336Ref: { file: 'TransformerLM.py', symbol: 'TransformerLM.forward (return value)' },
  },
}
