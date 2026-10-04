import { formatConcrete } from '../engine/shape'
import type { NodeDef } from '../engine/types'
import { NO_PARAMS, NOTHING_SAVED } from './common'

export const loss: NodeDef = {
  type: 'loss',
  label: 'Loss',
  category: 'loss',
  inputs: [{ id: 'in', label: 'loss' }],
  outputs: [],
  params: [],
  infer: ({ inputs: [x] }) =>
    x.dims.length === 0 ? { outputs: [], errors: [] } : { outputs: [], errors: [`Loss expects a scalar, got ${formatConcrete(x)}`] },
  paramCount: NO_PARAMS,
  savedForBackward: NOTHING_SAVED,
  docs: {
    overview:
      'The scalar training objective. Calling loss.backward() here sends gradients back through every part above it; the optimizer (AdamW in CS336) then uses them to update all the weights. Nothing flows out of it.',
    formula: ['θ ← AdamW(θ, ∂ℓ/∂θ)   for every weight tensor θ'],
    pointsToRemember: [
      'Must be a scalar (a rank-0 tensor).',
      'Every gradient has the same shape as its weight, so training needs at least weights + gradients in memory.',
      'AdamW keeps two extra tensors (m and v) per parameter: optimizer state = 2 × the weights.',
      'Gradient clipping and the cosine LR schedule act on these gradients but don’t change any shapes.',
    ],
    cs336Ref: { file: 'train.py', symbol: 'train (loss.backward → gradient_clipping → optimizer.step)' },
  },
}
