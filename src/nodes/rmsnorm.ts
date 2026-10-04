import { num, paramDim } from '../engine/resolve'
import type { NodeDef } from '../engine/types'
import { expectFloat, last, mismatch, weights } from './common'

export const rmsnorm: NodeDef = {
  type: 'rmsnorm',
  label: 'RMSNorm',
  category: 'norm',
  inputs: [{ id: 'in', label: 'x' }],
  outputs: [{ id: 'out', label: 'out' }],
  params: [
    { key: 'd_model', label: 'd_model', kind: 'int', default: { bind: 'd_model' }, min: 1, help: 'Size of the last dim being normalised (length of the gain vector g).' },
    { key: 'eps', label: 'eps', kind: 'float', default: { value: 1e-5 }, min: 0, help: 'Added inside the square root for numerical stability.' },
  ],
  infer: ({ inputs: [x], p }) => {
    const bad = expectFloat(x, 'RMSNorm input')
    if (bad) return { outputs: [], errors: [bad] }
    const d = num(p, 'd_model')
    if (last(x)!.size !== d) return { outputs: [], errors: [mismatch('RMSNorm.d_model', d, last(x)!)] }
    return { outputs: [structuredClone(x)], errors: [] }
  },
  paramCount: (p) => weights({ name: 'weight (g)', dims: [paramDim(p, 'd_model')] }),
  savedForBackward: ({ inputs }) => [
    { name: 'input x', which: 'input', port: 'in', shape: inputs[0] },
    { name: 'rms', which: 'internal', shape: { dims: [...inputs[0].dims.slice(0, -1), { size: 1 }], dtype: 'float' } },
  ],
  docs: {
    overview: 'Rescales each d_model-vector to unit root-mean-square, then multiplies by a learned gain g. No mean subtraction and no bias (unlike LayerNorm).',
    pointsToRemember: [
      'Shape is unchanged; only d_model parameters.',
      'CS336 upcasts to float32 inside the norm to avoid overflow when squaring.',
      'Pre-norm: applied before attention and before the FFN in each block, plus once at the end (ln_final).',
    ],
    formula: 'RMSNorm(a)_i = a_i · g_i / sqrt(mean(a²) + eps)',
    cs336Ref: 'RMSNorm.py',
  },
}
