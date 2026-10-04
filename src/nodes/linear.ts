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
    overview:
      'A learned matrix multiply on the last dimension: every in_features vector becomes an out_features vector. Every projection in the model is one of these — q/k/v/output in attention, w1/w2/w3 in SwiGLU, and the final lm_head.',
    roles: {
      q_proj: 'q_proj (W_Q): turns each token into a query — what this token is looking for.',
      k_proj: 'k_proj (W_K): turns each token into a key — what this token offers to be matched on.',
      v_proj: 'v_proj (W_V): turns each token into a value — the content attention mixes together.',
      output_proj: 'output_proj (W_O): mixes the concatenated heads and writes the result back to the residual stream.',
      w1: 'w1: up-projection d_model → d_ff; its output goes through SiLU and acts as the gate.',
      w3: 'w3: second up-projection d_model → d_ff; it is multiplied by the gate.',
      w2: 'w2: down-projection d_ff → d_model, back to the residual stream.',
      lm_head: 'lm_head: maps the final hidden state to one score per vocabulary entry (d_model → vocab_size).',
    },
    formula: ['y = x Wᵀ', 'W ∈ ℝ^(out_features × in_features)'],
    pointsToRemember: [
      'No bias in CS336 Linear: params are exactly in_features · out_features.',
      'Weight is stored as out_features × in_features (PyTorch convention), hence the transpose in x Wᵀ.',
      'in_features must equal the input’s last dim; leading dims (B, T, …) pass through unchanged.',
      'Init: truncated normal with σ² = 2 / (in_features + out_features), clipped to [−3σ, 3σ].',
      'Backward saves the input x: it is needed for the weight gradient ∂L/∂W = (∂L/∂y)ᵀ x.',
    ],
    paramHelp: {
      in_features: 'Size of each input vector. Must equal the last dim of the incoming tensor.',
      out_features: 'Size of each output vector (the new last dim).',
    },
    cs336Ref: { file: 'Linear.py', symbol: 'Linear.forward' },
  },
}
