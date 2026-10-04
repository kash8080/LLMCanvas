import { num } from '../engine/resolve'
import type { NodeDef } from '../engine/types'
import { expectFloat, NO_PARAMS } from './common'

export const softmax: NodeDef = {
  type: 'softmax',
  label: 'Softmax',
  category: 'attention',
  inputs: [{ id: 'in', label: 'x' }],
  outputs: [{ id: 'out', label: 'out' }],
  params: [{ key: 'dim', label: 'dim', kind: 'int', default: { value: -1 }, help: 'Dimension to normalise over (negative counts from the end).' }],
  infer: ({ inputs: [x], p }) => {
    const bad = expectFloat(x, 'Softmax input')
    if (bad) return { outputs: [], errors: [bad] }
    const dim = num(p, 'dim')
    const rank = x.dims.length
    if (dim < -rank || dim >= rank) return { outputs: [], errors: [`dim = ${dim} is out of range for a rank-${rank} tensor`] }
    return { outputs: [structuredClone(x)], errors: [] }
  },
  paramCount: NO_PARAMS,
  savedForBackward: ({ outputs }) => [{ name: 'output', which: 'output', port: 'out', shape: outputs[0] }],
  docs: {
    overview: 'Turns scores into probabilities along one dim: exp(x_i − max) / Σ exp(x_j − max). Used inside attention.',
    pointsToRemember: ['Subtracting the max first keeps exp() from overflowing.', 'Backward needs the output (softmax probabilities).'],
    formula: 'softmax(x)_i = exp(x_i) / Σ_j exp(x_j)',
    cs336Ref: 'Softmax.py',
  },
}
