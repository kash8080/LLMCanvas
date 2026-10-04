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
    overview: 'The scalar training objective. loss.backward() starts here and flows gradients back through every part above.',
    pointsToRemember: ['Must be a scalar.', 'The optimizer (AdamW in CS336) then updates all parameters using those gradients.'],
    cs336Ref: 'train.py',
  },
}
