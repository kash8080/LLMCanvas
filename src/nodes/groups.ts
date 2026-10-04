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
        'One pre-norm Transformer layer. First every token gathers information from earlier tokens (attention), then each token is transformed on its own (SwiGLU FFN). Each sub-layer reads a normalised copy of the residual stream and adds its result back. The model stacks num_layers of these blocks, each with its own weights.',
      formula: ['y   = x + MHA(RMSNorm₁(x))', 'out = y + SwiGLU(RMSNorm₂(y))'],
      paramFormula: '4·d_model² + 3·d_model·d_ff + 2·d_model',
      pointsToRemember: [
        'Pre-norm: norm is applied before attention and before the FFN; the residual adds the un-normalised input.',
        'Input and output are both B × T × d_model, so blocks chain freely; num_layers = Transformer Blocks on the canvas.',
        'Params per block = 4·d² (attention) + 3·d·d_ff (FFN) + 2·d (two RMSNorm gains) = 3,113,984 with the defaults.',
        'The FFN holds about two thirds of a block’s weights when d_ff ≈ 8/3·d_model.',
        'Attention is the only place tokens exchange information; everything else works per token.',
        'CS336 can checkpoint each block (checkpoint_blocks): keep only its input, recompute the rest during backward.',
      ],
      cs336Ref: { file: 'TransformerBlock.py', symbol: 'TransformerBlock.forward' },
    },
  },
  mha: {
    type: 'mha',
    label: 'Multi-Head Self-Attention',
    color: '#a855f7',
    expandAtLod: 2,
    docs: {
      overview:
        'Lets each token look at earlier tokens and pull in information from them. x is projected to queries, keys and values, split into H heads, Q and K are rotated by RoPE, each head runs causal scaled dot-product attention, and the heads are merged and mixed by output_proj.',
      formula: [
        'Q, K, V = x W_Qᵀ, x W_Kᵀ, x W_Vᵀ   → split into H heads',
        'headₕ = softmax(RoPE(Qₕ) RoPE(Kₕ)ᵀ / √d_head + mask) Vₕ',
        'MHA(x) = concat(head₁ … head_H) W_Oᵀ',
      ],
      paramFormula: '4·d_model²',
      pointsToRemember: [
        'Params = 4·d_model² (q, k, v and output projections, no biases), independent of the number of heads.',
        'd_model must be divisible by num_heads; d_head = d_model / H.',
        'RoPE has no learnable params; only Q and K are rotated, not V.',
        'Causal mask (torch.tril) prevents attending to future tokens.',
        'Attention probs are B × H × T × T — quadratic in T, usually the biggest activation in a block.',
      ],
      cs336Ref: { file: 'MultiHeadSelfAttention.py', symbol: 'MultiHeadSelfAttention.forward' },
    },
  },
  swiglu: {
    type: 'swiglu',
    label: 'SwiGLU FFN',
    color: '#f97316',
    expandAtLod: 2,
    docs: {
      overview:
        'The position-wise feed-forward network: it transforms each token on its own, with no mixing between positions. It widens to d_ff with two projections (w1, w3), gates one with SiLU of the other, and projects back to d_model with w2.',
      formula: ['FFN(x) = W₂ ( SiLU(W₁x) ⊙ W₃x )', 'W₁, W₃ ∈ ℝ^(d_ff × d_model)', 'W₂ ∈ ℝ^(d_model × d_ff)'],
      paramFormula: '3·d_model·d_ff',
      pointsToRemember: [
        'Params = 3·d_model·d_ff (w1, w2, w3, no biases).',
        'd_ff ≈ 8/3·d_model rounded to a multiple of 64 (512 → 1344): same weight count as a classic 2-matrix FFN with d_ff = 4·d_model.',
        'w1 and w3 both map d_model → d_ff; only the w1 branch goes through SiLU.',
        'Its hidden activations (B × T × d_ff) are the widest tensors in the block.',
        'No interaction between tokens here — mixing across positions only happens in attention.',
      ],
      cs336Ref: { file: 'SwiGLU.py', symbol: 'SwiGLU.forward' },
    },
  },
}

/** Palette order. */
export const GROUP_DEF_LIST: GroupDef[] = GROUP_TYPES.map((t) => GROUP_DEFS[t])
