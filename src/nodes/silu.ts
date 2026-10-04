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
    overview:
      'The Sigmoid Linear Unit multiplies each value by its own sigmoid. It acts like ReLU for large positive inputs and fades smoothly to 0 for negative ones. In SwiGLU it is applied to the w1 branch, which then gates the w3 branch.',
    formula: ['SiLU(x) = x · σ(x) = x / (1 + e^(−x))'],
    pointsToRemember: [
      'Element-wise: shape unchanged, no parameters.',
      'Smooth and slightly non-monotonic (dips to ≈ −0.28 near x ≈ −1.28), unlike ReLU’s hard kink at 0.',
      'Backward needs the input x.',
      'Also known as Swish (with β = 1).',
    ],
    cs336Ref: { file: 'SwiGLU.py', symbol: 'SwiGLU.silu' },
  },
}
