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
    overview: 'Marks the model output: one unnormalised score per vocabulary entry at every position (B × T × V). A pass-through.',
    pointsToRemember: [
      'TransformerLM.forward returns these logits; softmax turns them into next-token probabilities.',
      'With the CS336 defaults the logits are one of the largest activations (32·256·10000 floats).',
    ],
    cs336Ref: 'TransformerLM.py',
  },
}
