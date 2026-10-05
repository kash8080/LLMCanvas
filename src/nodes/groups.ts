// Composite parts (PLAN.md §2.3): Transformer Block, Multi-Head Self-Attention, SwiGLU FFN, non-gated FFN.
// A group has no shape logic of its own: its children are ordinary parts and the engine sees a
// flat graph (src/engine/groups.ts). This file holds what the UI needs to know about each kind;
// the child layout lives in src/canvas/groupTemplates.ts.
import type { Hyperparams, NodeDocs, PortDef } from '../engine/types'

export const GROUP_TYPES = ['transformer_block', 'mha', 'swiglu', 'ffn'] as const
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
  /** Params of the standard template with every param bound (the drawer flags a group that differs). */
  standardParams: (hp: Hyperparams) => number
  docs: NodeDocs
}

/** Sub-layer groups that can be created inside a Transformer Block (quick-add in its empty area). */
export const SUBLAYER_GROUP_TYPES: GroupType[] = ['mha', 'swiglu', 'ffn']

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
    standardParams: ({ d_model: d, d_ff: F }) => 4 * d * d + 3 * d * F + 2 * d,
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
    standardParams: ({ d_model: d }) => 4 * d * d,
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
    standardParams: ({ d_model: d, d_ff: F }) => 3 * d * F,
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
  ffn: {
    type: 'ffn',
    label: 'FFN (non-gated)',
    color: '#fb923c',
    expandAtLod: 2,
    standardParams: ({ d_model: d, d_ff: F }) => 2 * d * F,
    docs: {
      overview:
        'The classic two-matrix feed-forward network: widen each token to d_ff with w1, apply an activation, project back with w2. There is no gate (no w3), so it has two weight matrices instead of SwiGLU’s three. This is the CS336 ablation baseline (FFN with SiLU, “SwiGLU vs. SiLU”); swap the activation part for GELU or ReLU to get GPT-2’s or the original Transformer’s FFN.',
      formula: ['FFN(x) = W₂ · act(W₁x)     act = SiLU (default), GELU or ReLU', 'W₁ ∈ ℝ^(d_ff × d_model)', 'W₂ ∈ ℝ^(d_model × d_ff)'],
      paramFormula: '2·d_model·d_ff',
      pointsToRemember: [
        'Params = 2·d_model·d_ff (w1, w2, no biases) = 1,376,256 with the default d_ff = 1344.',
        'With d_ff = 4·d_model (CS336’s choice for this ablation, 2048 here) it has 8·d² = 2,097,152 params — about the same as SwiGLU with d_ff ≈ 8/3·d_model (2,064,384).',
        'w1 and w2 follow the global d_ff (🔗); unlink w1.out_features and w2.in_features to give this FFN its own width (e.g. 2048).',
        'Saves less for backward than SwiGLU at the same d_ff: w1’s output (for the activation) and the activation’s output (for w2) — 2 vs 4 tensors of B × T × d_ff (just 1 with ReLU, which reuses its output).',
        'The activation is an ordinary part: delete it and drop in GELU or ReLU (same shape, no params).',
        'The params breakdown counts it as FFN; the standard formula assumes SwiGLU (3·d·d_ff), so it lists the difference.',
      ],
      cs336Ref: { file: 'SiLU.py', symbol: 'SiLU.forward' },
    },
  },
}

/** Palette order. */
export const GROUP_DEF_LIST: GroupDef[] = GROUP_TYPES.map((t) => GROUP_DEFS[t])
