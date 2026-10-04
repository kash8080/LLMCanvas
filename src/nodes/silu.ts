import type { NodeDef } from '../engine/types'
import { expectFloat, NO_PARAMS } from './common'

export const silu: NodeDef = {
  type: 'silu',
  label: 'SiLU',
  category: 'ffn',
  inputs: [{ id: 'in', label: 'x' }],
  outputs: [{ id: 'out', label: 'out' }],
  params: [],
  infer: ({ inputs: [x] }) => {
    const bad = expectFloat(x, 'SiLU input')
    return bad ? { outputs: [], errors: [bad] } : { outputs: [structuredClone(x)], errors: [] }
  },
  paramCount: NO_PARAMS,
  savedForBackward: ({ inputs }) => [{ name: 'input x', which: 'input', port: 'in', shape: inputs[0] }],
  docs: {
    overview: 'SiLU (a.k.a. Swish): x · sigmoid(x). A smooth ReLU-like activation used inside SwiGLU.',
    pointsToRemember: ['Element-wise: shape unchanged, no parameters.', 'Backward needs the input x.'],
    formula: 'SiLU(x) = x · σ(x) = x / (1 + e^(−x))',
    cs336Ref: 'SwiGLU.py (silu)',
  },
}
