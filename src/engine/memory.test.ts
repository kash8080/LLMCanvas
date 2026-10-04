import { describe, expect, it } from 'vitest'
import type { AppEdge, AppNode } from '../canvas/types'
import { cs336Document } from '../defaults/cs336Graph'
import { inferGraph, memoryInput } from '../store/inference'
import { DEFAULT_HYPERPARAMS } from './hyperparams'
import { estimateMemory, memoryFormulas, memoryInsights, type MemoryMode } from './memory'
import type { Hyperparams } from './types'

const graph = () => {
  const doc = cs336Document()
  return { nodes: doc.nodes as AppNode[], edges: doc.edges as AppEdge[] }
}
const input = (mode: MemoryMode, checkpointing = false, hp: Hyperparams = DEFAULT_HYPERPARAMS) => {
  const g = graph()
  return memoryInput(inferGraph(g.nodes, g.edges, hp), hp, mode, checkpointing)
}
const mem = (mode: MemoryMode, checkpointing = false, hp: Hyperparams = DEFAULT_HYPERPARAMS) => estimateMemory(input(mode, checkpointing, hp))

const P = 16_468_480
const BTd = 32 * 256 * 512 * 4 // one B·T·d fp32 tensor = 16,777,216
const BTF = 32 * 256 * 1344 * 4 // one B·T·d_ff fp32 tensor
const PROBS = 32 * 16 * 256 * 256 * 4 // 134,217,728 per layer
const LOGITS = 32 * 256 * 10000 * 4 // 327,680,000
const RMS = 32 * 256 * 4
const IDS = 32 * 256 * 8 // int64 token ids / targets
// Per block: block input, ln1 out, Q, K, V, merge out, add1 out, ln2 out (8 × B·T·d) + 2 rms + probs + 4 × B·T·d_ff
const PER_BLOCK = 8 * BTd + 2 * RMS + PROBS + 4 * BTF
// Outside blocks: token ids, ln_final input + rms, lm_head input, logits, targets
const OUTSIDE = IDS + BTd + RMS + BTd + LOGITS + IDS
const ROPE = 2 * 256 * 16 * 4 // cos + sin per attention group

describe('memory: weights, gradients, optimizer, buffers', () => {
  it('weights = P · 4 bytes in every mode', () => {
    for (const m of ['forward', 'fwd_bwd', 'train'] as const) expect(mem(m).byComponent.weights).toBe(65_873_920)
  })

  it('gradients = P · b in fwd_bwd and train only; optimizer = 2 · P · b in train only', () => {
    expect(mem('forward').byComponent.gradients).toBe(0)
    expect(mem('fwd_bwd').byComponent.gradients).toBe(65_873_920)
    expect(mem('fwd_bwd').byComponent.optimizer).toBe(0)
    expect(mem('train').byComponent.gradients).toBe(65_873_920)
    expect(mem('train').byComponent.optimizer).toBe(131_747_840)
  })

  it('RoPE buffers: one cos/sin pair per attention group (two RoPE parts share it)', () => {
    const r = mem('train')
    expect(r.buffers.items).toHaveLength(2)
    expect(r.buffers.items[0].ropeIds).toHaveLength(2)
    expect(r.buffers.total).toBe(2 * ROPE)
  })

  it('total = sum of components', () => {
    const r = mem('train')
    expect(r.total).toBe(4 * P * 4 + r.activations.total + 2 * ROPE)
  })
})

describe('memory: saved activations (fwd_bwd / train)', () => {
  const r = mem('fwd_bwd')
  const a = r.activations

  it('attention probs: 32·16·256·256·4 per layer', () => {
    const probs = a.tensors.filter((t) => t.category === 'attn_probs')
    expect(probs.map((t) => t.bytes)).toEqual([PROBS, PROBS])
    expect(a.byCategory.attn_probs).toBe(268_435_456)
    expect(probs[0].owner).toMatch(/\.sdpa$/)
  })

  it('logits saved by cross-entropy: 32·256·10000·4, attributed to lm_head', () => {
    const logits = a.tensors.find((t) => t.key === 'lm_head:out')!
    expect(logits.bytes).toBe(LOGITS)
    expect(logits.savedBy).toEqual(['xent'])
    expect(logits.category).toBe('logits')
    expect(a.tensors[0]).toBe(logits) // the biggest tensor
  })

  it('counts each unique tensor once: ln1 output feeds q/k/v but is stored once', () => {
    const ln1 = a.tensors.filter((t) => t.key === 'b1.ln1:out')
    expect(ln1).toHaveLength(1)
    expect(ln1[0].bytes).toBe(BTd)
    expect(ln1[0].savedBy.sort()).toEqual(['b1.k_proj', 'b1.q_proj', 'b1.v_proj'])
    expect(ln1[0].category).toBe('norm')
  })

  it('sums to the expected total', () => {
    expect(a.total).toBe(2 * PER_BLOCK + OUTSIDE)
    expect(r.total).toBe(2 * P * 4 + a.total + 2 * ROPE) // weights + gradients
  })

  it('same activations in train mode', () => {
    expect(mem('train').activations.total).toBe(a.total)
  })

  it('breaks activations down by layer and part', () => {
    expect(a.rows.map((x) => x.id)).toEqual(['embed', 'b1', 'b2', 'ln_final', 'lm_head', 'xent'])
    const b1 = a.rows.find((x) => x.id === 'b1')!
    // A row holds what its parts produce: b1 doesn't own its input (embed's output) but owns its
    // output (b1.add2, saved by Block 2's ln1), so it adds up to one block's worth.
    expect(b1.bytes).toBe(PER_BLOCK)
    expect(a.byPart['b1.sdpa']).toBe(PROBS)
    expect(r.groups.b1.keys).toContain('attn_probs')
    expect(r.groups['b1.attn'].activations).toBeGreaterThan(PROBS)
  })

  it('token ids are int64 (8 bytes) and attributed to the part that saves them', () => {
    const ids = a.tensors.find((t) => t.key === 'data:input_ids')!
    expect(ids.bytes).toBe(IDS)
    expect(ids.owner).toBe('embed')
    expect(ids.category).toBe('embedding')
    expect(a.tensors.find((t) => t.key === 'data:targets')!.category).toBe('logits')
  })
})

describe('memory: dtype, checkpointing, forward mode', () => {
  it('bf16 halves every float part (int64 stays 8 bytes)', () => {
    const fp32 = mem('train')
    const bf16 = mem('train', false, { ...DEFAULT_HYPERPARAMS, dtype: 'bf16' })
    expect(bf16.byComponent.weights).toBe(fp32.byComponent.weights / 2)
    expect(bf16.byComponent.optimizer).toBe(fp32.byComponent.optimizer / 2)
    expect(bf16.byComponent.buffers).toBe(fp32.byComponent.buffers / 2)
    expect(bf16.activations.byCategory.attn_probs).toBe(PROBS)
    expect(bf16.activations.total).toBe((fp32.activations.total - 2 * IDS) / 2 + 2 * IDS)
  })

  it('checkpointing keeps block inputs (B·T·d each) + one recomputed block', () => {
    const full = mem('train')
    const ck = mem('train', true)
    const a = ck.activations
    expect(a.checkpointedBlocks).toBe(2)
    expect(a.blockInputBytes).toBe(2 * BTd)
    expect(a.tensors.filter((t) => t.role === 'block_input').map((t) => t.bytes)).toEqual([BTd, BTd])
    // The recomputed block's saved tensors, minus its input (already kept as a block input).
    expect(a.recomputeBytes).toBe(PER_BLOCK - BTd)
    expect(a.total).toBe(OUTSIDE + 2 * BTd + PER_BLOCK - BTd)
    expect(a.total).toBeLessThan(full.activations.total)
    expect(a.byCategory.attn_probs).toBe(PROBS) // only one layer's probs at a time
  })

  it('checkpointing is ignored in forward mode', () => {
    expect(mem('forward', true).activations.total).toBe(mem('forward').activations.total)
  })

  it('forward mode: peak live tensors at one part, less than saved activations', () => {
    const f = mem('forward')
    expect(f.activations.peakPart).toBe('lm_head')
    expect(f.activations.total).toBe(BTd + LOGITS) // lm_head input + logits output
    expect(f.activations.total).toBeLessThan(mem('fwd_bwd').activations.total)
    // SDPA live set: Q, K, V, probs, out
    expect(f.activations.byPart['b1.sdpa']).toBe(4 * BTd + PROBS)
    expect(f.total).toBe(P * 4 + f.activations.total + 2 * ROPE)
  })

  it('attention probs grow with T²: context 1024 → 16× per layer', () => {
    const hp = { ...DEFAULT_HYPERPARAMS, context_length: 1024 }
    expect(mem('train', false, hp).activations.byCategory.attn_probs).toBe(16 * 2 * PROBS)
  })
})

describe('memory formulas and insights', () => {
  it('writes the component formulas with numbers', () => {
    const f = memoryFormulas(mem('train'))
    expect(f.optimizer).toContain('2 · 16,468,480 · 4')
    expect(f.weights).toBe('P · bytes = 16,468,480 · 4')
    expect(f.buffers).toContain('2 · 2 · 256 · 16 · 4')
  })

  it('insights mention probs share, logits, checkpointing savings and bf16', () => {
    const i = input('train')
    const lines = memoryInsights(i, estimateMemory(i))
    expect(lines.some((l) => l.startsWith('Attention probs are'))).toBe(true)
    expect(lines.some((l) => l.includes('B·T·V'))).toBe(true)
    expect(lines.some((l) => l.startsWith('Activation checkpointing would save'))).toBe(true)
    expect(lines.some((l) => l.startsWith('bf16 would'))).toBe(true)
    expect(lines.some((l) => l.includes('2 × weights'))).toBe(true)
  })
})
