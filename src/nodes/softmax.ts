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
    overview:
      'Turns a vector of arbitrary scores into probabilities: all positive and summing to 1 along one dimension. In the CS336 model it lives inside attention (normalising over the keys); the same idea appears inside cross-entropy.',
    formula: ['softmax(x)ᵢ = exp(xᵢ − max x) / Σⱼ exp(xⱼ − max x)'],
    pointsToRemember: [
      'Subtracting the max doesn’t change the result but keeps exp() from overflowing (numerically stable).',
      'Shape unchanged, no parameters.',
      'Backward needs only the output: the softmax Jacobian is built from the probabilities themselves.',
      'An input of −∞ becomes exactly 0 probability — that is how the causal mask works.',
      'Already included in Scaled Dot-Product Attention; use this part only when building attention by hand.',
    ],
    paramHelp: { dim: 'Dimension to normalise over (−1 = last). In attention that is the keys axis.' },
    cs336Ref: { file: 'Softmax.py', symbol: 'softmax' },
  },
}
