// Registry of model-part definitions (NodeDef), one file per part in this folder.
import { isMemKey, type MemoryCategory, type MemoryComponent, type MemoryHighlightKey } from '../engine/memory'
import type { HighlightKey } from '../engine/params'
import type { Category, NodeDef } from '../engine/types'
import { add } from './add'
import { crossEntropy } from './crossEntropy'
import { dataBatch } from './dataBatch'
import { embedding } from './embedding'
import { gelu } from './gelu'
import { PROXY_DEFS } from './groupProxy'
import { layernorm } from './layernorm'
import { linear } from './linear'
import { logits } from './logits'
import { loss } from './loss'
import { mergeHeads } from './mergeHeads'
import { multiply } from './multiply'
import { relu } from './relu'
import { rmsnorm } from './rmsnorm'
import { rope } from './rope'
import { sdpa } from './sdpa'
import { silu } from './silu'
import { softmax } from './softmax'
import { splitHeads } from './splitHeads'

/** In palette order. */
export const NODE_DEFS: NodeDef[] = [
  dataBatch,
  embedding,
  rmsnorm,
  layernorm,
  linear,
  splitHeads,
  rope,
  sdpa,
  softmax,
  mergeHeads,
  silu,
  gelu,
  relu,
  multiply,
  add,
  logits,
  crossEntropy,
  loss,
]

/** Every part type: the palette parts plus the group proxies (which only appear inside groups). */
export const nodeRegistry: Record<string, NodeDef> = Object.fromEntries([...NODE_DEFS, ...PROXY_DEFS].map((d) => [d.type, d]))

export function getNodeDef(type: string): NodeDef | undefined {
  return nodeRegistry[type]
}

export const CATEGORY_INFO: Record<Category, { label: string; color: string }> = {
  io: { label: 'Input / output', color: '#64748b' },
  embedding: { label: 'Embedding', color: '#3b82f6' },
  norm: { label: 'Normalization', color: '#14b8a6' },
  linear: { label: 'Linear', color: '#6366f1' },
  attention: { label: 'Attention', color: '#a855f7' },
  ffn: { label: 'Feed-forward', color: '#f97316' },
  elementwise: { label: 'Element-wise', color: '#0ea5e9' },
  loss: { label: 'Loss', color: '#ef4444' },
}

/** Palette category order. */
export const CATEGORY_ORDER: Category[] = ['io', 'embedding', 'norm', 'linear', 'attention', 'ffn', 'elementwise', 'loss']

/** Parameter-breakdown categories (engine/params.ts) and the canvas highlight for unconnected parts. */
export const PARAM_CATEGORY_INFO: Record<HighlightKey, { label: string; color: string; help: string }> = {
  embedding: { label: 'Embedding', color: '#3b82f6', help: 'Token embedding table: V · d_model' },
  attention: { label: 'Attention', color: '#a855f7', help: 'q/k/v/output projections inside attention groups: 4 · d_model² per layer' },
  ffn: { label: 'FFN', color: '#f97316', help: 'w1/w2/w3 inside SwiGLU groups: 3 · d_model · d_ff per layer (non-gated FFN: w1/w2, 2 · d_model · d_ff)' },
  norm: { label: 'Norms', color: '#14b8a6', help: 'RMSNorm gains: 2 · d_model per layer + ln_final (LayerNorm: 2 · d_model each)' },
  lm_head: { label: 'LM head', color: '#6366f1', help: 'The Linear feeding Logits: d_model · V (0 with weight tying: it reuses the embedding matrix)' },
  other: { label: 'Other', color: '#94a3b8', help: 'Weights outside the standard structure (e.g. an extra Linear)' },
  unconnected: { label: 'Unconnected', color: '#f59e0b', help: 'Parts that don’t feed Logits / Loss — not counted in the model total' },
}

/** Activation categories of the memory estimate (engine/memory.ts). */
export const MEMORY_CATEGORY_INFO: Record<MemoryCategory, { label: string; color: string; help: string }> = {
  attn_probs: { label: 'Attention probs (B·H·T·T)', color: '#e11d48', help: 'softmax(QKᵀ/√d_k) saved by each attention: B · H · T² per layer' },
  attention: { label: 'Attention other', color: '#a855f7', help: 'Q, K, V after RoPE, the merged heads (output_proj input): B · T · d each' },
  ffn: { label: 'FFN', color: '#f97316', help: 'SwiGLU tensors: w1 and w3 outputs, SiLU output, gate output — B · T · d_ff each (non-gated FFN: w1 output, activation output)' },
  norm: { label: 'Norms', color: '#14b8a6', help: 'Norm outputs (inputs of the projections) and their per-token statistics (rms; LayerNorm mean + rstd)' },
  embedding: { label: 'Embedding', color: '#3b82f6', help: 'Token ids (int64) and the embedding output (Block 1’s input)' },
  logits: { label: 'Logits / loss', color: '#6366f1', help: 'B · T · V logits saved by the cross-entropy, plus the targets' },
  residual: { label: 'Residual / other', color: '#94a3b8', help: 'Residual-stream tensors (Add outputs) and anything else' },
}

/** Memory components (stacked bar in the Memory tab). */
export const MEMORY_COMPONENT_INFO: Record<MemoryComponent, { label: string; color: string; help: string }> = {
  weights: { label: 'Weights', color: '#0ea5e9', help: 'Every parameter: P · bytes' },
  gradients: { label: 'Gradients', color: '#f59e0b', help: 'One gradient per parameter, same dtype: P · bytes' },
  optimizer: { label: 'Optimizer', color: '#8b5cf6', help: 'AdamW m and v per parameter: 2 · P · bytes' },
  activations: { label: 'Activations', color: '#ef4444', help: 'Tensors kept for backward (or live at the forward peak)' },
  buffers: { label: 'Buffers', color: '#64748b', help: 'RoPE cos/sin tables (non-persistent buffers)' },
}

/** Label + colour of anything the analysis panel can highlight. */
export function highlightInfo(key: HighlightKey | MemoryHighlightKey): { label: string; color: string; help: string } {
  return isMemKey(key) ? MEMORY_CATEGORY_INFO[key.slice(4) as MemoryCategory] : PARAM_CATEGORY_INFO[key]
}
