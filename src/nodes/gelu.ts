import type { NodeDef } from '../engine/types'
import { expectFloat, NO_PARAMS } from './common'

/** GELU (Hendrycks & Gimpel, 2016): the activation of GPT-2 / BERT. Not in CS336, which uses SiLU. */
export const gelu: NodeDef = {
  type: 'gelu',
  label: 'GELU',
  category: 'ffn',
  inputs: [{ id: 'in', label: 'x' }],
  outputs: [{ id: 'out', label: 'out' }],
  params: [],
  infer: ({ inputs: [x] }) => {
    const bad = expectFloat(x, 'GELU input')
    return bad ? { outputs: [], errors: [bad] } : { outputs: [structuredClone(x)], errors: [] }
  },
  paramCount: NO_PARAMS,
  savedForBackward: ({ inputs }) => [{ name: 'input x', which: 'input', port: 'in', shape: inputs[0] }],
  docs: {
    overview:
      'The Gaussian Error Linear Unit weights each value by the probability that a standard normal variable is below it. Like SiLU it is a smooth version of ReLU: ≈ x for large positive x, ≈ 0 for large negative x. It is the FFN activation of GPT-2 and BERT; swap it in for SiLU in a non-gated FFN to compare.',
    formula: ['GELU(x) = x · Φ(x)     (Φ = standard normal CDF)', '        ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))', 'vs SiLU(x) = x · σ(x) ≈ x · Φ(1.702·x)'],
    pointsToRemember: [
      'Element-wise: shape unchanged, no parameters.',
      'Very close to SiLU in shape (both are x times a smooth 0→1 gate); dips to ≈ −0.17 near x ≈ −0.75.',
      'Backward needs the input x (the derivative is Φ(x) + x·φ(x)), so it keeps one B × T × d_ff tensor in an FFN.',
      'Used inside a gate it becomes GEGLU (GELU(W₁x) ⊙ W₃x), the gated cousin of SwiGLU.',
    ],
    cs336Note:
      'Not in the CS336 code (its FFNs use SiLU: SwiGLU.py, SiLU.py); variant for comparison. The handout cites the GELU paper (Hendrycks & Gimpel) with SiLU/Swish.',
  },
}
