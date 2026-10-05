// Global hyperparameters (CS336 train.py defaults; PLAN.md §3) and their symbols.
import type { BindKey, FloatDType, HyperparamKey, Hyperparams, TensorDType } from './types'

export const DEFAULT_HYPERPARAMS: Hyperparams = {
  vocab_size: 10000,
  context_length: 256,
  d_model: 512,
  num_heads: 16,
  d_ff: 1344,
  rope_theta: 10000,
  batch_size: 32,
  dtype: 'fp32',
  tie_embeddings: false,
}

export const FLOAT_DTYPES: FloatDType[] = ['fp32', 'bf16', 'fp16']

export interface HyperparamInfo {
  key: HyperparamKey
  /** Symbol used in symbolic shapes (B × T × d_model). */
  symbol: string
  help: string
  kind: 'int' | 'float'
}

/** Editable numeric hyperparams, in display order. */
export const HYPERPARAM_INFO: HyperparamInfo[] = [
  { key: 'vocab_size', symbol: 'V', kind: 'int', help: 'Number of token ids (rows of the embedding matrix).' },
  { key: 'context_length', symbol: 'T', kind: 'int', help: 'Maximum sequence length; the data batch uses it as seq len.' },
  { key: 'd_model', symbol: 'd_model', kind: 'int', help: 'Width of the residual stream.' },
  { key: 'num_heads', symbol: 'H', kind: 'int', help: 'Attention heads; d_model must be divisible by it.' },
  { key: 'd_ff', symbol: 'd_ff', kind: 'int', help: 'Hidden width of the SwiGLU feed-forward network.' },
  { key: 'rope_theta', symbol: 'θ', kind: 'float', help: 'RoPE base frequency Θ.' },
  { key: 'batch_size', symbol: 'B', kind: 'int', help: 'Sequences per training batch.' },
]

const SYMBOLS: Record<BindKey, string> = {
  vocab_size: 'V',
  context_length: 'T',
  d_model: 'd_model',
  num_heads: 'H',
  d_ff: 'd_ff',
  rope_theta: 'θ',
  batch_size: 'B',
  d_head: 'd_head',
}

export function bindSymbol(key: BindKey): string {
  return SYMBOLS[key]
}

/** Value of a bind target; d_head is derived (may be fractional if d_model % num_heads != 0). */
export function bindValue(key: BindKey, hp: Hyperparams): number {
  if (key === 'd_head') return hp.d_model / hp.num_heads
  return hp[key]
}

/** Problems with the global hyperparams themselves (shown in the hyperparams panel). */
export function validateHyperparams(hp: Hyperparams): string[] {
  const errors: string[] = []
  for (const { key, kind } of HYPERPARAM_INFO) {
    const v = hp[key]
    if (!Number.isFinite(v) || v <= 0 || (kind === 'int' && !Number.isInteger(v)))
      errors.push(`${key} must be a positive ${kind === 'int' ? 'integer' : 'number'}`)
  }
  if (Number.isInteger(hp.d_model) && Number.isInteger(hp.num_heads) && hp.num_heads > 0 && hp.d_model % hp.num_heads !== 0)
    errors.push(`d_model (${hp.d_model}) is not divisible by num_heads (${hp.num_heads})`)
  else if (hp.num_heads > 0 && (hp.d_model / hp.num_heads) % 2 !== 0)
    errors.push(`RoPE needs an even head dim, but d_model / num_heads = ${hp.d_model / hp.num_heads}`)
  return errors
}

/** Bytes per element: int64 = 8, fp32 = 4, bf16/fp16 = 2. */
export function bytesPerElement(dtype: TensorDType, hp: Pick<Hyperparams, 'dtype'>): number {
  if (dtype === 'int64') return 8
  return hp.dtype === 'fp32' ? 4 : 2
}

/** Display name of a tensor dtype ('float' → the global dtype). */
export function dtypeName(dtype: TensorDType, hp: Pick<Hyperparams, 'dtype'>): string {
  return dtype === 'int64' ? 'int64' : hp.dtype
}
