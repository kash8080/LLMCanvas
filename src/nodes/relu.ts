import type { NodeDef } from '../engine/types'
import { expectFloat, NO_PARAMS } from './common'

/** ReLU: the activation of the original Transformer FFN. Not in CS336, which uses SiLU. */
export const relu: NodeDef = {
  type: 'relu',
  label: 'ReLU',
  category: 'ffn',
  inputs: [{ id: 'in', label: 'x' }],
  outputs: [{ id: 'out', label: 'out' }],
  params: [],
  infer: ({ inputs: [x] }) => {
    const bad = expectFloat(x, 'ReLU input')
    return bad ? { outputs: [], errors: [bad] } : { outputs: [structuredClone(x)], errors: [] }
  },
  paramCount: NO_PARAMS,
  // Like PyTorch: backward only needs to know where the output is > 0, and it reads that from the
  // output itself. The next Linear keeps the same tensor as its input, so it is stored once (shared).
  savedForBackward: ({ outputs }) => [{ name: 'output (where > 0)', which: 'output', port: 'out', shape: outputs[0] }],
  docs: {
    overview:
      'The Rectified Linear Unit keeps positive values and sets negative ones to 0. The original Transformer FFN was W₂·ReLU(W₁x) with d_ff = 4·d_model. Modern LLMs replaced it with smooth activations (GELU, SiLU) and gating (SwiGLU).',
    formula: ['ReLU(x) = max(0, x)', 'ReLU′(x) = 1 if x > 0 else 0'],
    pointsToRemember: [
      'Element-wise: shape unchanged, no parameters.',
      'Hard kink at 0 and exactly zero gradient for negative inputs (“dead” units can stop learning); SiLU / GELU are smooth there.',
      'Backward only needs the sign pattern, which it reads from the output: we count the saved output, and since the next Linear (w2) keeps that same tensor it is stored once. (Storing a bool mask instead would be 1 byte per value.)',
      'In a non-gated FFN with d_ff = 4·d_model this is the classic Transformer FFN (8·d² params per layer).',
    ],
    cs336Note:
      'Not in the CS336 code (its FFNs use SiLU: SwiGLU.py, SiLU.py); variant for comparison. The handout (§3.4.2) describes the original ReLU FFN.',
  },
}
