import { formatConcrete, sameSizes } from '../engine/shape'
import type { NodeDef } from '../engine/types'
import { expectFloat, NO_PARAMS } from './common'

export const multiply: NodeDef = {
  type: 'multiply',
  label: 'Multiply (⊙)',
  category: 'ffn',
  inputs: [
    { id: 'a', label: 'a' },
    { id: 'b', label: 'b' },
  ],
  outputs: [{ id: 'out', label: 'out' }],
  params: [],
  infer: ({ inputs: [a, b] }) => {
    const bad = expectFloat(a, 'a') ?? expectFloat(b, 'b')
    if (bad) return { outputs: [], errors: [bad] }
    if (!sameSizes(a, b)) return { outputs: [], errors: [`Multiply inputs differ: a is ${formatConcrete(a)}, b is ${formatConcrete(b)}`] }
    return { outputs: [structuredClone(a)], errors: [] }
  },
  paramCount: NO_PARAMS,
  savedForBackward: ({ inputs: [a, b] }) => [
    { name: 'a', which: 'input', port: 'a', shape: a },
    { name: 'b', which: 'input', port: 'b', shape: b },
  ],
  docs: {
    overview:
      'Element-wise (Hadamard) product of two tensors of the same shape. In SwiGLU it is the gate: SiLU(w1 x) decides, per hidden unit, how much of w3 x gets through.',
    roles: { gate: 'gate: SiLU(w1 x) ⊙ w3 x, the “GLU” part of SwiGLU.' },
    formula: ['out = a ⊙ b', 'SwiGLU gate: SiLU(W₁x) ⊙ W₃x'],
    pointsToRemember: [
      'Both inputs must have exactly the same shape (no broadcasting here).',
      'No parameters, but backward saves both inputs (∂out/∂a = b, ∂out/∂b = a).',
      'In SwiGLU both inputs are B × T × d_ff — the widest activations in the block.',
      'Gating is why SwiGLU beats a plain ReLU/SiLU FFN at a similar parameter count.',
    ],
    cs336Ref: { file: 'SwiGLU.py', symbol: 'SwiGLU.forward (a * b)' },
  },
}
