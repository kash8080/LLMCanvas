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
    overview: 'Element-wise product of two same-shape tensors. In SwiGLU it gates W3·x with SiLU(W1·x).',
    pointsToRemember: ['Both inputs must have the same shape.', 'Backward needs both inputs (d(a·b)/da = b and vice versa).'],
    formula: 'out = a ⊙ b',
    cs336Ref: 'SwiGLU.py',
  },
}
