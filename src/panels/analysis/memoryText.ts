// Display helpers for the memory estimate (shared by the Memory tab, the toolbar and the drawer).
import type { AppNode } from '../../canvas/types'
import type { MemoryMode, MemTensor } from '../../engine/memory'
import { nodeTitle } from '../../store/inference'

export const MODE_INFO: Record<MemoryMode, { label: string; short: string; explain: string }> = {
  forward: {
    label: 'Forward',
    short: 'Fwd',
    explain: 'Inference without gradients: weights + RoPE buffers + the largest set of tensors alive at once (one part’s inputs, outputs and temporaries).',
  },
  fwd_bwd: {
    label: 'Fwd+Bwd',
    short: 'Fwd+Bwd',
    explain: 'One forward + backward pass: weights + one gradient per weight + every activation saved for backward (each unique tensor once) + buffers.',
  },
  train: {
    label: 'Train',
    short: 'Train',
    explain: 'A training step with AdamW: weights + gradients + optimizer state (m and v per weight) + activations saved for backward + buffers.',
  },
}

/** Explanation of the generation (KV cache) view, shown instead of the Forward one. */
export const GENERATION_EXPLAIN =
  'Generation with a KV cache: weights + RoPE buffers + the K and V of every cached token in every attention layer + the tensors alive while ONE new token goes through the model (T = 1; attention reads all T_cache cached keys).'

export type NodeIndex = Map<string, AppNode>

export function titleOf(id: string, byId: NodeIndex): string {
  const n = byId.get(id)
  return n ? nodeTitle(n) : id
}

/** "Block 1 › ln1 output", "Block 2 › attention · attention probs", "lm_head output", "targets". */
export function tensorName(t: MemTensor, byId: NodeIndex): string {
  const prefix = t.row !== t.producer && byId.get(t.row)?.type === 'group' ? `${titleOf(t.row, byId)} › ` : ''
  if (t.internal) return `${prefix}${titleOf(t.producer, byId)} · ${t.label}`
  // A Data Batch output is named by its port (input_ids / targets).
  if (t.producer !== t.owner) return t.label
  const logits = t.category === 'logits' && t.label === 'out' ? ' (logits)' : ''
  return `${prefix}${titleOf(t.producer, byId)} ${t.label === 'out' ? 'output' : t.label}${logits}`
}

/** "q_proj, k_proj, v_proj" (a checkpointed block appears as "checkpoint of Block 2"). */
export function saversText(t: MemTensor, byId: NodeIndex, except?: string): string {
  return t.savedBy
    .filter((id) => id !== except)
    .map((id) => (byId.get(id)?.type === 'group' ? `checkpoint of ${titleOf(id, byId)}` : titleOf(id, byId)))
    .join(', ')
}

/** "saved by q_proj, k_proj, v_proj · stored once" / "kept as Block 2’s checkpointed input". */
export function savedText(t: MemTensor, byId: NodeIndex): string {
  if (t.role === 'block_input') {
    const blocks = t.savedBy.filter((id) => byId.get(id)?.type === 'group').map((id) => titleOf(id, byId))
    return `kept as the checkpointed input of ${blocks.join(', ')}`
  }
  return `saved by ${saversText(t, byId)}${t.savedBy.length > 1 ? ' · stored once' : ''}${t.role === 'recompute' ? ' · held while its block is recomputed in backward' : ''}`
}

export function pct(x: number, total: number): string {
  if (total <= 0) return '—'
  const v = (x / total) * 100
  if (v > 0 && v < 0.1) return '<0.1%'
  return `${v > 0 && v < 1 ? v.toFixed(1) : Math.round(v)}%`
}
