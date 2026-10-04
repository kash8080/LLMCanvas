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
    overview:
      'Turns each integer token id into a learned d_model-dimensional vector by picking that row of the weight table. This is where the residual stream starts: everything after it works on float vectors of size d_model.',
    roles: { token_embeddings: 'token_embeddings: the only input to Block 1 — CS336 adds no separate position embedding (position comes from RoPE).' },
    formula: ['out[b, t, :] = W[token_ids[b, t], :]', 'W ∈ ℝ^(vocab_size × d_model)'],
    pointsToRemember: [
      'Embedding is a lookup (weight[token_ids]), not a matmul: no one-hot vectors are ever built.',
      'Output shape = input shape + (d_model,): B × T → B × T × d_model.',
      'Init: truncated normal N(0, 1), clipped to [−3, 3].',
      'Backward only adds gradient to the rows that were looked up; unused rows get zero gradient.',
      'Not tied to lm_head in CS336, so the vocab_size · d_model weights are counted twice in the model.',
      'Token ids must lie in [0, vocab_size).',
    ],
    paramHelp: {
      vocab_size: 'num_embeddings: rows of the table, one per token id. Must match the tokenizer’s vocabulary.',
      d_model: 'embedding_dim: width of each vector = width of the residual stream.',
    },
    cs336Ref: { file: 'Embedding.py', symbol: 'Embedding.forward' },
  },
}
