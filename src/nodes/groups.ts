// Composite parts (PLAN.md §2.3): Transformer Block, Multi-Head Self-Attention, SwiGLU FFN.
// A group has no shape logic of its own: its children are ordinary parts and the engine sees a
// flat graph (src/engine/groups.ts). This file holds what the UI needs to know about each kind;
// the child layout lives in src/canvas/groupTemplates.ts.
import type { NodeDocs, PortDef } from '../engine/types'

export const GROUP_TYPES = ['transformer_block', 'mha', 'swiglu'] as const
export type GroupType = (typeof GROUP_TYPES)[number]

export function isGroupType(v: unknown): v is GroupType {
  return typeof v === 'string' && (GROUP_TYPES as readonly string[]).includes(v)
}

export interface GroupDef {
  type: GroupType
  label: string
  /** Frame / card accent colour. */
  color: string
  /** Auto mode shows the group's insides from this level of detail on (see src/canvas/lod.ts). */
  expandAtLod: 1 | 2
  docs: NodeDocs
}

/** Every group has exactly one outer input and one outer output port. */
export const GROUP_PORTS: { inputs: PortDef[]; outputs: PortDef[] } = {
  inputs: [{ id: 'in', label: 'x' }],
  outputs: [{ id: 'out', label: 'out' }],
}

export const GROUP_DEFS: Record<GroupType, GroupDef> = {
  transformer_block: {
    type: 'transformer_block',
    label: 'Transformer Block',
    color: '#475569',
    expandAtLod: 1,
    docs: {
      overview:
        'One pre-norm Transformer layer: x + MHA(RMSNorm(x)), then x + SwiGLU(RMSNorm(x)). The model stacks num_layers of these.',
      formula: 'y = x + MHA(ln1(x));  out = y + SwiGLU(ln2(y))',
      pointsToRemember: [
        'Params per block = 4·d² (attention) + 3·d·d_ff (SwiGLU) + 2·d (two RMSNorms).',
        'Input and output are both B × T × d_model, so blocks chain freely.',
        'num_layers is the number of Transformer Blocks on the canvas.',
      ],
      cs336Ref: 'TransformerBlock.py',
    },
  },
  mha: {
    type: 'mha',
    label: 'Multi-Head Self-Attention',
    color: '#a855f7',
    expandAtLod: 2,
    docs: {
      overview:
        'Projects x to queries, keys and values, splits them into H heads, rotates q and k with RoPE, runs causal attention per head, merges the heads and projects back to d_model.',
      formula: 'MHA(x) = W_o · merge(softmax(QKᵀ/√d_head + mask) V)',
      pointsToRemember: ['Params = 4·d_model² (q, k, v and output projections, no biases).', 'RoPE and attention have no learned weights.'],
      cs336Ref: 'MultiHeadSelfAttention.py',
    },
  },
  swiglu: {
    type: 'swiglu',
    label: 'SwiGLU FFN',
    color: '#f97316',
    expandAtLod: 2,
    docs: {
      overview: 'Gated feed-forward network: w2(SiLU(w1 x) ⊙ w3 x). Widens to d_ff, gates, and projects back to d_model.',
      formula: 'FFN(x) = W₂ (SiLU(W₁x) ⊙ W₃x)',
      pointsToRemember: ['Params = 3·d_model·d_ff (w1, w2, w3, no biases).', 'CS336 uses d_ff ≈ 8/3·d_model (1344 for d_model = 512).'],
      cs336Ref: 'SwiGLU.py',
    },
  },
}

/** Palette order. */
export const GROUP_DEF_LIST: GroupDef[] = GROUP_TYPES.map((t) => GROUP_DEFS[t])
