import { num, paramDim } from '../engine/resolve'
import type { NodeDef } from '../engine/types'
import { expectFloat, last, mismatch, weights } from './common'

/** LayerNorm (Ba et al., 2016): the original Transformer's norm. Not in CS336, which uses RMSNorm. */
export const layernorm: NodeDef = {
  type: 'layernorm',
  label: 'LayerNorm',
  category: 'norm',
  inputs: [{ id: 'in', label: 'x' }],
  outputs: [{ id: 'out', label: 'out' }],
  params: [
    { key: 'd_model', label: 'd_model', kind: 'int', default: { bind: 'd_model' }, min: 1, help: 'Size of the last dim being normalised (length of γ and β).' },
    { key: 'eps', label: 'eps', kind: 'float', default: { value: 1e-5 }, min: 0, help: 'Added to the variance inside the square root for numerical stability.' },
  ],
  infer: ({ inputs: [x], p }) => {
    const bad = expectFloat(x, 'LayerNorm input')
    if (bad) return { outputs: [], errors: [bad] }
    const d = num(p, 'd_model')
    if (last(x)!.size !== d) return { outputs: [], errors: [mismatch('LayerNorm.d_model', d, last(x)!)] }
    return { outputs: [structuredClone(x)], errors: [] }
  },
  paramCount: (p) => weights({ name: 'weight (γ)', dims: [paramDim(p, 'd_model')] }, { name: 'bias (β)', dims: [paramDim(p, 'd_model')] }),
  // Like PyTorch's native layer norm: the input plus one mean and one 1/σ (rstd) per token.
  savedForBackward: ({ inputs }) => {
    const perToken = { dims: [...inputs[0].dims.slice(0, -1), { size: 1 }], dtype: 'float' as const }
    return [
      { name: 'input x', which: 'input', port: 'in', shape: inputs[0] },
      { name: 'mean', which: 'internal', shape: perToken },
      { name: 'rstd', which: 'internal', shape: structuredClone(perToken) },
    ]
  },
  docs: {
    overview:
      'Normalises every d_model vector to mean 0 and variance 1, then scales it by a learned gain γ and shifts it by a learned bias β. It is the norm of the original Transformer and GPT-2. RMSNorm (used by CS336, LLaMA) drops the mean subtraction and the bias: it only divides by the root-mean-square and multiplies by a gain.',
    formula: [
      'LayerNorm(a)ᵢ = (aᵢ − μ) / √(σ² + ε) · γᵢ + βᵢ',
      'μ = (1/d_model) · Σⱼ aⱼ     σ² = (1/d_model) · Σⱼ (aⱼ − μ)²',
      'vs RMSNorm(a)ᵢ = aᵢ / √((1/d_model) · Σⱼ aⱼ² + ε) · gᵢ',
    ],
    pointsToRemember: [
      'Params = 2·d_model (gain γ initialised to 1, bias β initialised to 0) — twice RMSNorm’s d_model.',
      'RMSNorm skips the mean-centring and the bias: one reduction instead of two and a little cheaper, with about the same quality in practice.',
      'Shape is unchanged; the statistics are taken over the last dim only, separately for every token (no mixing across tokens or the batch).',
      'Backward keeps the input x plus a per-token mean and 1/σ (rstd) — tiny compared with x.',
      'Swapping it in for RMSNorm changes the Norms term of the formula from (2L + 1)·d to (2L + 1)·2d.',
    ],
    paramHelp: {
      d_model: 'Length of the last dim being normalised = length of γ and β (the only weights).',
      eps: 'ε added to the variance inside the square root so we never divide by ~0. PyTorch default 1e-5; not learned.',
    },
    cs336Note:
      'Not in the CS336 code (it uses RMSNorm.py); variant for comparison. The handout (§3.4.1) notes the original Transformer used LayerNorm and follows LLaMA in switching to RMSNorm.',
  },
}
