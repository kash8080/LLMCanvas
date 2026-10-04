import { num, paramDim } from '../engine/resolve'
import type { NodeDef } from '../engine/types'
import { expectFloat, last, mismatch, weights } from './common'

export const linear: NodeDef = {
  type: 'linear',
  label: 'Linear',
  category: 'linear',
  inputs: [{ id: 'in', label: 'x' }],
  outputs: [{ id: 'out', label: 'out' }],
  params: [
    { key: 'in_features', label: 'in_features', kind: 'int', default: { bind: 'd_model' }, min: 1, help: 'Must equal the last dim of the input.' },
    { key: 'out_features', label: 'out_features', kind: 'int', default: { bind: 'd_model' }, min: 1, help: 'Last dim of the output.' },
  ],
  infer: ({ inputs: [x], p }) => {
    const bad = expectFloat(x, 'Linear input')
    if (bad) return { outputs: [], errors: [bad] }
    const inF = num(p, 'in_features')
    if (last(x)!.size !== inF) return { outputs: [], errors: [mismatch('Linear.in_features', inF, last(x)!)] }
    return { outputs: [{ dims: [...x.dims.slice(0, -1), paramDim(p, 'out_features')], dtype: 'float' }], errors: [] }
  },
  paramCount: (p) => weights({ name: 'weight', dims: [paramDim(p, 'out_features'), paramDim(p, 'in_features')] }),
  savedForBackward: ({ inputs }) => [{ name: 'input x', which: 'input', port: 'in', shape: inputs[0] }],
  docs: {
    overview: 'A matrix multiply on the last dim: y = x Wᵀ. CS336 Linear has no bias.',
    pointsToRemember: [
      'weight is stored as out_features × in_features.',
      'Params = in_features · out_features.',
      'Backward needs the input x to compute the weight gradient.',
    ],
    formula: 'y = x · Wᵀ',
    cs336Ref: 'Linear.py',
  },
}
