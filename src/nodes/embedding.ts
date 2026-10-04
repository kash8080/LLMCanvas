import { num, paramDim } from '../engine/resolve'
import { formatConcrete } from '../engine/shape'
import type { NodeDef } from '../engine/types'
import { weights } from './common'

export const embedding: NodeDef = {
  type: 'embedding',
  label: 'Embedding',
  category: 'embedding',
  inputs: [{ id: 'in', label: 'token_ids' }],
  outputs: [{ id: 'out', label: 'out' }],
  params: [
    { key: 'vocab_size', label: 'vocab_size', kind: 'int', default: { bind: 'vocab_size' }, min: 1, help: 'Rows of the embedding table (num_embeddings).' },
    { key: 'd_model', label: 'd_model', kind: 'int', default: { bind: 'd_model' }, min: 1, help: 'Length of each embedding vector (embedding_dim).' },
  ],
  infer: ({ inputs: [x], p }) => {
    if (x.dtype !== 'int64') return { outputs: [], errors: [`Embedding expects int64 token ids, got a float tensor (${formatConcrete(x)})`] }
    if (num(p, 'vocab_size') < 1) return { outputs: [], errors: ['vocab_size must be ≥ 1'] }
    return { outputs: [{ dims: [...x.dims, paramDim(p, 'd_model')], dtype: 'float' }], errors: [] }
  },
  paramCount: (p) => weights({ name: 'weight', dims: [paramDim(p, 'vocab_size'), paramDim(p, 'd_model')] }),
  // Backward only needs the token ids to scatter gradients into the rows that were used.
  savedForBackward: ({ inputs }) => [{ name: 'token_ids', which: 'input', port: 'in', shape: inputs[0] }],
  docs: {
    overview: 'Looks up one learned d_model-vector per token id: output = weight[token_ids]. It is indexing, not a matrix multiply.',
    pointsToRemember: [
      'weight has shape vocab_size × d_model.',
      'Output shape = input shape + (d_model,).',
      'CS336 does not tie the embedding to the LM head, so both are counted.',
    ],
    formula: 'out[b, t] = W[token_ids[b, t]]',
    cs336Ref: 'Embedding.py',
  },
}
