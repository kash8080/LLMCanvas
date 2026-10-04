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
    overview:
      'Rescales every d_model vector so its root-mean-square is 1, then multiplies it element-wise by a learned gain g. This keeps activations at a steady scale as they enter attention, the FFN and the LM head. Unlike LayerNorm there is no mean subtraction and no bias.',
    roles: {
      ln1: 'ln1: normalises the residual stream before attention.',
      ln2: 'ln2: normalises the residual stream before the SwiGLU FFN.',
      ln_final: 'ln_final: normalises the output of the last block before lm_head (needed because pre-norm leaves the stream un-normalised).',
    },
    formula: ['RMSNorm(a)ᵢ = aᵢ / RMS(a) · gᵢ', 'RMS(a) = √( (1/d_model) · Σⱼ aⱼ² + ε )'],
    pointsToRemember: [
      'Pre-norm: the norm is applied to the input of each sub-layer (ln1, ln2); the residual adds the un-normalised x.',
      'One extra RMSNorm (ln_final) sits after the last block, before lm_head.',
      'The input is upcast to float32 before squaring (avoids fp16/bf16 overflow), then cast back to its dtype.',
      'ε = 1e-5 keeps the division safe when a vector is close to 0.',
      'Only d_model params (the gain g, initialised to 1); the shape is unchanged.',
    ],
    paramHelp: {
      d_model: 'Length of the last dim being normalised = length of the gain vector g (the only weights).',
      eps: 'ε added inside the square root so we never divide by ~0. CS336 default 1e-5; not learned.',
    },
    cs336Ref: { file: 'RMSNorm.py', symbol: 'RMSNorm.forward' },
  },
}
