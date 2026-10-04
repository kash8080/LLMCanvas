// Registry of model-part definitions (NodeDef), one file per part in this folder.
import type { Category, NodeDef } from '../engine/types'
import { add } from './add'
import { crossEntropy } from './crossEntropy'
import { dataBatch } from './dataBatch'
import { embedding } from './embedding'
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

export const nodeRegistry: Record<string, NodeDef> = Object.fromEntries(NODE_DEFS.map((d) => [d.type, d]))

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
