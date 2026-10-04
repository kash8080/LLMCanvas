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
    overview:
      'Element-wise sum of two same-shape tensors. As a residual connection it adds a sub-layer’s output back onto that sub-layer’s input (x + f(x)). Chained through every block, these Adds form the “residual stream” running from the embedding to ln_final.',
    roles: {
      'x + attn': 'x + attn: adds the attention output onto the block input.',
      'x + ffn': 'x + ffn: adds the FFN output onto the stream after attention.',
    },
    formula: ['out = a + b', 'residual: x ← x + Sublayer(RMSNorm(x))'],
    pointsToRemember: [
      'Both inputs must have the same shape.',
      'Pre-norm: the residual adds the un-normalised x; only the branch input goes through RMSNorm.',
      'The gradient passes through unchanged to both inputs, giving deep stacks a direct gradient path.',
      'No parameters and nothing saved for backward.',
    ],
    cs336Ref: { file: 'TransformerBlock.py', symbol: 'TransformerBlock.forward (x + …)' },
  },
}
