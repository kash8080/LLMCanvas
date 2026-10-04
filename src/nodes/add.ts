import { formatConcrete, sameSizes } from '../engine/shape'
import type { NodeDef } from '../engine/types'
import { expectFloat, NO_PARAMS, NOTHING_SAVED } from './common'

export const add: NodeDef = {
  type: 'add',
  label: 'Add (residual)',
  category: 'elementwise',
  inputs: [
    { id: 'a', label: 'a' },
    { id: 'b', label: 'b' },
  ],
  outputs: [{ id: 'out', label: 'out' }],
  params: [],
  infer: ({ inputs: [a, b] }) => {
    const bad = expectFloat(a, 'a') ?? expectFloat(b, 'b')
    if (bad) return { outputs: [], errors: [bad] }
    if (!sameSizes(a, b)) return { outputs: [], errors: [`Add inputs differ: a is ${formatConcrete(a)}, b is ${formatConcrete(b)}`] }
    return { outputs: [structuredClone(a)], errors: [] }
  },
  paramCount: NO_PARAMS,
  savedForBackward: NOTHING_SAVED,
  docs: {
    overview: 'Element-wise sum. As a residual connection it adds a sub-layer’s output back onto its input: x + f(x).',
    pointsToRemember: [
      'Both inputs must have the same shape.',
      'Nothing saved for backward: the gradient just copies to both inputs.',
      'Residuals give gradients a direct path through deep stacks.',
    ],
    formula: 'out = a + b',
    cs336Ref: 'TransformerBlock.py',
  },
}
