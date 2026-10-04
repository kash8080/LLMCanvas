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
    overview:
      'Samples B random windows of T consecutive token ids from the tokenized training data. targets is the same window shifted one position to the right, so every position has a “next token” to predict. This is where the batch size B and sequence length T enter the model.',
    formula: ['s_b ~ Uniform(0, N − T)   (random start per row)', 'input_ids[b] = data[s_b : s_b + T]', 'targets[b]   = data[s_b + 1 : s_b + T + 1]'],
    pointsToRemember: [
      'Both outputs are int64 token ids of shape B × T (8 bytes each, whatever the model dtype).',
      'targets[b, t] = input_ids[b, t + 1]: one batch is B·T next-token prediction problems at once.',
      'Start positions are sampled at random (with replacement), not by walking through the data in order.',
      'seq_len must not exceed context_length: RoPE’s cos/sin tables only cover that many positions.',
      'B and T scale every activation downstream, but never the number of weights.',
    ],
    paramHelp: {
      batch_size: 'B: independent sequences processed together. Scales activation memory linearly; weights are unaffected.',
      seq_len: 'T: tokens per sequence. Follows context_length by default; attention memory grows with T².',
    },
    cs336Ref: { file: 'DataLoading.py', symbol: 'DataLoading.load' },
  },
}
