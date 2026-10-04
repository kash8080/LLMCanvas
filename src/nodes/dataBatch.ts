import { paramDim } from '../engine/resolve'
import type { NodeDef } from '../engine/types'
import { NO_PARAMS, NOTHING_SAVED } from './common'

export const dataBatch: NodeDef = {
  type: 'data_batch',
  label: 'Data Batch',
  category: 'io',
  inputs: [],
  outputs: [
    { id: 'input_ids', label: 'input_ids' },
    { id: 'targets', label: 'targets' },
  ],
  params: [
    { key: 'batch_size', label: 'batch_size', kind: 'int', default: { bind: 'batch_size' }, min: 1, help: 'Sequences per batch (B).' },
    { key: 'seq_len', label: 'seq_len', kind: 'int', default: { bind: 'context_length' }, min: 1, help: 'Tokens per sequence (T).' },
  ],
  infer: ({ p }) => {
    const ids = { dims: [paramDim(p, 'batch_size'), paramDim(p, 'seq_len')], dtype: 'int64' as const }
    return { outputs: [ids, structuredClone(ids)], errors: [] }
  },
  paramCount: NO_PARAMS,
  savedForBackward: NOTHING_SAVED,
  docs: {
    overview: 'A batch of B random windows of T tokens from the training data. targets are the same windows shifted one token to the right.',
    pointsToRemember: [
      'Both tensors are int64 token ids of shape B × T.',
      'targets[i, t] = input_ids[i, t + 1]: the model predicts the next token at every position.',
    ],
    cs336Ref: 'DataLoading.py',
  },
}
