// Registry of model-part definitions (NodeDef), one file per part in this folder.
import type { HighlightKey } from '../engine/params'
import type { Category, NodeDef } from '../engine/types'
import { add } from './add'
import { crossEntropy } from './crossEntropy'
import { dataBatch } from './dataBatch'
import { embedding } from './embedding'
import { PROXY_DEFS } from './groupProxy'
import { linear } from './linear'
import { logits } from './logits'
import { loss } from './loss'
import { mergeHeads } from './mergeHeads'
import { multiply } from './multiply'
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
  linear,
  splitHeads,
  rope,
  sdpa,
  softmax,
  mergeHeads,
  silu,
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
  ffn: { label: 'Feed-forward (SwiGLU)', color: '#f97316' },
  elementwise: { label: 'Element-wise', color: '#0ea5e9' },
  loss: { label: 'Loss', color: '#ef4444' },
}

/** Palette category order. */
export const CATEGORY_ORDER: Category[] = ['io', 'embedding', 'norm', 'linear', 'attention', 'ffn', 'elementwise', 'loss']

/** Parameter-breakdown categories (engine/params.ts) and the canvas highlight for unconnected parts. */
export const PARAM_CATEGORY_INFO: Record<HighlightKey, { label: string; color: string; help: string }> = {
  embedding: { label: 'Embedding', color: '#3b82f6', help: 'Token embedding table: V · d_model' },
  attention: { label: 'Attention', color: '#a855f7', help: 'q/k/v/output projections inside attention groups: 4 · d_model² per layer' },
  ffn: { label: 'FFN', color: '#f97316', help: 'w1/w2/w3 inside SwiGLU groups: 3 · d_model · d_ff per layer' },
  norm: { label: 'Norms', color: '#14b8a6', help: 'RMSNorm gains: 2 · d_model per layer + ln_final' },
  lm_head: { label: 'LM head', color: '#6366f1', help: 'The Linear feeding Logits: d_model · V' },
  other: { label: 'Other', color: '#94a3b8', help: 'Weights outside the standard structure (e.g. an extra Linear)' },
  unconnected: { label: 'Unconnected', color: '#f59e0b', help: 'Parts that don’t feed Logits / Loss — not counted in the model total' },
}
