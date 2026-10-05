// Phase 7c: LayerNorm / GELU / ReLU parts, the non-gated FFN group and weight tying.
import { describe, expect, it } from 'vitest'
import { edgeId, instantiateGroup } from '../canvas/groupTemplates'
import { createPartNode } from '../canvas/nodeFactory'
import type { AppEdge, AppNode, PartNode } from '../canvas/types'
import { cs336Document } from '../defaults/cs336Graph'
import { nodeRegistry } from '../nodes/registry'
import { parseDocument, toDocument } from '../store/persistence'
import { inferGraph, memoryInput } from '../store/inference'
import { DEFAULT_HYPERPARAMS } from './hyperparams'
import { estimateMemory } from './memory'
import { resolveParams } from './resolve'
import type { Hyperparams, Shape } from './types'

const hp = DEFAULT_HYPERPARAMS
const tiedHp: Hyperparams = { ...hp, tie_embeddings: true }
const BTd: Shape = {
  dims: [
    { size: 32, label: 'B' },
    { size: 256, label: 'T' },
    { size: 512, label: 'd_model' },
  ],
  dtype: 'float',
}
const BTF = 32 * 256 * 1344 * 4 // one B·T·d_ff fp32 tensor

function load() {
  const doc = cs336Document()
  return { nodes: doc.nodes as AppNode[], edges: doc.edges as AppEdge[] }
}
const infer = (g: { nodes: AppNode[]; edges: AppEdge[] }, h: Hyperparams = hp) => inferGraph(g.nodes, g.edges, h)
const mem = (g: { nodes: AppNode[]; edges: AppEdge[] }, h: Hyperparams = hp, mode: 'forward' | 'fwd_bwd' | 'train' = 'train') =>
  estimateMemory(memoryInput(infer(g, h), h, mode, false))

/** Run one part def on an input shape. */
function runPart(type: string, input: Shape) {
  const def = nodeRegistry[type]
  const p = resolveParams(def, createPartNode(type, { x: 0, y: 0 }).data.params, hp)
  const r = def.infer({ inputs: [input], p, hp })
  return { def, p, r, count: def.paramCount(p, hp), saved: def.savedForBackward({ inputs: [input], outputs: r.outputs, p, hp }) }
}

/** Default graph with Block 1's SwiGLU replaced by a non-gated FFN (activation `act`) inside the block. */
function withNonGatedFfn(act = 'silu') {
  const g = load()
  const drop = new Set(g.nodes.filter((n) => n.id === 'b1.ffn' || n.parentId === 'b1.ffn').map((n) => n.id))
  const nodes = g.nodes.filter((n) => !drop.has(n.id))
  const edges = g.edges.filter((e) => !drop.has(e.source) && !drop.has(e.target))
  const built = instantiateGroup('ffn', { x: 300, y: 1200 }, nodes, 'b1')
  const id = built.nodes[0].id
  if (act !== 'silu') {
    const actNode = built.nodes.find((n) => n.id === `${id}.act`) as PartNode
    actNode.data = { ...actNode.data, partType: act, params: createPartNode(act, { x: 0, y: 0 }).data.params, title: act }
  }
  edges.push(
    { id: edgeId('b1.ln2', 'out', id, 'in'), source: 'b1.ln2', sourceHandle: 'out', target: id, targetHandle: 'in' },
    { id: edgeId(id, 'out', 'b1.add2', 'b'), source: id, sourceHandle: 'out', target: 'b1.add2', targetHandle: 'b' },
  )
  return { nodes: [...nodes, ...built.nodes], edges: [...edges, ...built.edges], id }
}

describe('LayerNorm', () => {
  it('keeps the shape and has 2·d params (gain γ + bias β)', () => {
    const { r, count } = runPart('layernorm', BTd)
    expect(r.errors).toEqual([])
    expect(r.outputs[0]).toEqual(BTd)
    expect(count.total).toBe(2 * 512)
    expect(count.tensors.map((t) => t.name)).toEqual(['weight (γ)', 'bias (β)'])
    expect(count.tensors[0].dims).toEqual([{ size: 512, label: 'd_model' }])
  })

  it('saves its input plus a per-token mean and rstd', () => {
    const { saved } = runPart('layernorm', BTd)
    expect(saved.map((s) => [s.name, s.which])).toEqual([
      ['input x', 'input'],
      ['mean', 'internal'],
      ['rstd', 'internal'],
    ])
    expect(saved[1].shape.dims.map((d) => d.size)).toEqual([32, 256, 1])
  })

  it('reports a d_model mismatch like RMSNorm', () => {
    const { r } = runPart('layernorm', { dims: [...BTd.dims.slice(0, 2), { size: 768 }], dtype: 'float' })
    expect(r.errors[0]).toMatch(/LayerNorm\.d_model = 512 but input last dim is 768/)
  })

  it('swapped in for ln_final: Norms + 512 and a formula note', () => {
    const g = load()
    const ln = g.nodes.find((n) => n.id === 'ln_final') as PartNode
    ln.data = { ...ln.data, partType: 'layernorm', params: createPartNode('layernorm', { x: 0, y: 0 }).data.params }
    const r = infer(g).params
    expect(r.total).toBe(16_468_480 + 512)
    expect(r.parts.ln_final.category).toBe('norm')
    expect(r.formula.matches).toBe(false)
    expect(r.formula.notes.join(' ')).toMatch(/1 LayerNorm \(2·d each/)
  })
})

describe('GELU and ReLU', () => {
  it('are element-wise with no params', () => {
    for (const t of ['gelu', 'relu']) {
      const { r, count } = runPart(t, BTd)
      expect(r.outputs[0], t).toEqual(BTd)
      expect(count.total, t).toBe(0)
    }
    expect(runPart('gelu', { dims: [{ size: 32 }], dtype: 'int64' }).r.errors[0]).toMatch(/float tensor/)
  })

  it('GELU saves its input; ReLU saves its output', () => {
    expect(runPart('gelu', BTd).saved).toMatchObject([{ which: 'input', port: 'in' }])
    expect(runPart('relu', BTd).saved).toMatchObject([{ which: 'output', port: 'out' }])
  })
})

describe('FFN (non-gated) group', () => {
  it('has 2 · 512 · 1344 = 1,376,256 params with the defaults (d_ff bound to the global)', () => {
    const g = withNonGatedFfn()
    const res = infer(g)
    expect(res.groups[g.id].params).toBe(1_376_256)
    expect(res.groups[g.id].status).toBe('ok')
    const w1 = g.nodes.find((n) => n.id === `${g.id}.w1`) as PartNode
    expect(w1.data.params.out_features).toEqual({ bind: 'd_ff' })
  })

  it('swapping Block 1’s SwiGLU for it changes the totals, keeps FFN as the category and explains the formula gap', () => {
    const g = withNonGatedFfn()
    const r = infer(g).params
    expect(r.total).toBe(16_468_480 - 2_064_384 + 1_376_256)
    expect(r.byCategory.ffn).toBe(2_064_384 + 1_376_256)
    expect(r.parts[`${g.id}.w1`].category).toBe('ffn')
    expect(r.rows.find((x) => x.id === 'b1')!.params).toBe(3_113_984 - 2_064_384 + 1_376_256)
    expect(r.formula.matches).toBe(false)
    expect(r.formula.diffs).toEqual([{ category: 'ffn', actual: 3_440_640, expected: 2 * 2_064_384 }])
    expect(r.formula.notes[0]).toMatch(/1 non-gated FFN/)
  })

  it('own d_ff = 4·d_model (2048) gives 8·d² params', () => {
    const g = withNonGatedFfn()
    for (const [part, key] of [
      ['w1', 'out_features'],
      ['w2', 'in_features'],
    ]) {
      const n = g.nodes.find((x) => x.id === `${g.id}.${part}`) as PartNode
      n.data = { ...n.data, params: { ...n.data.params, [key]: { value: 2048 } } }
    }
    const res = infer(g)
    expect(res.groups[g.id].params).toBe(8 * 512 * 512)
    expect(res.groups[g.id].status).toBe('ok')
  })

  it('memory: weights shrink and it saves 2 B·T·d_ff tensors instead of SwiGLU’s 4', () => {
    const base = mem(load(), hp, 'fwd_bwd')
    const g = withNonGatedFfn()
    const m = mem(g, hp, 'fwd_bwd')
    expect(m.byComponent.weights).toBe((16_468_480 - 2_064_384 + 1_376_256) * 4)
    expect(base.activations.total - m.activations.total).toBe(2 * BTF)
    const ffnTensors = m.activations.tensors.filter((t) => t.row === 'b1' && t.category === 'ffn')
    expect(ffnTensors.map((t) => t.bytes)).toEqual([BTF, BTF])
  })

  it('with ReLU the activation output is stored once (shared by ReLU and w2), w1’s output not at all', () => {
    const g = withNonGatedFfn('relu')
    const m = mem(g, hp, 'fwd_bwd')
    const ffnTensors = m.activations.tensors.filter((t) => t.row === 'b1' && t.category === 'ffn')
    expect(ffnTensors).toHaveLength(1)
    expect(ffnTensors[0].savedBy.sort()).toEqual([`${g.id}.act`, `${g.id}.w2`].sort())
    expect(infer(g).params.total).toBe(16_468_480 - 2_064_384 + 1_376_256)
  })
})

describe('weight tying (tie_embeddings)', () => {
  it('is off by default (CS336)', () => {
    expect(hp.tie_embeddings).toBe(false)
    expect(infer(load()).params.tied).toEqual([])
  })

  it('on: lm_head counts 0, total = 16,468,480 − 5,120,000 = 11,348,480 and the formula drops + d·V', () => {
    const res = infer(load(), tiedHp)
    const r = res.params
    expect(r.total).toBe(11_348_480)
    expect(r.byCategory.lm_head).toBe(0)
    expect(r.byCategory.embedding).toBe(5_120_000)
    expect(r.tied).toEqual([{ id: 'lm_head', to: 'embed', params: 5_120_000 }])
    expect(r.parts.lm_head).toMatchObject({ params: 0, tiedTo: 'embed', category: 'lm_head' })
    expect(r.rows.map((x) => x.id)).toEqual(['embed', 'b1', 'b2', 'ln_final', 'lm_head'])
    expect(res.nodes.lm_head.paramCount).toMatchObject({ total: 0, tied: { to: 'embed', params: 5_120_000 } })
    expect(res.nodes.lm_head.status).toBe('ok')
    expect(r.formula).toMatchObject({ matches: true, tied: true, value: 11_348_480, symbolic: 'V·d + L·(4d² + 3d·d_ff + 2d) + d' })
    expect(r.formula.substituted).not.toMatch(/512·10000$/)
  })

  it('memory: weights, gradients and AdamW use the reduced P', () => {
    const m = mem(load(), tiedHp, 'train')
    expect(m.P).toBe(11_348_480)
    expect(m.byComponent.weights).toBe(45_393_920)
    expect(m.byComponent.gradients).toBe(45_393_920)
    expect(m.byComponent.optimizer).toBe(2 * 45_393_920)
    const untied = mem(load(), hp, 'train')
    expect(untied.total - m.total).toBe(4 * 5_120_000 * 4)
    expect(m.activations.total).toBe(untied.activations.total)
  })

  it('mismatch: an error on the LM head, nothing tied, the formula explains', () => {
    const g = load()
    const head = g.nodes.find((n) => n.id === 'lm_head') as PartNode
    head.data = { ...head.data, params: { ...head.data.params, out_features: { value: 12000 } } }
    const res = infer(g, tiedHp)
    expect(res.nodes.lm_head.status).toBe('error')
    expect(res.nodes.lm_head.errors[0]).toMatch(/Weight tying: .*12000 × 512.*10000 × 512/)
    expect(res.params.tied).toEqual([])
    expect(res.params.byCategory.lm_head).toBe(12000 * 512)
    expect(res.params.formula.notes.join(' ')).toMatch(/could not be tied/)
    // Off again: no error.
    expect(infer(g).nodes.lm_head.status).toBe('ok')
  })

  it('no Embedding feeding the LM head: an error', () => {
    const g = load()
    g.edges = g.edges.filter((e) => !(e.source === 'embed' && e.target === 'b1'))
    const res = infer(g, tiedHp)
    expect(res.nodes.lm_head.errors.join(' ')).toMatch(/no Embedding feeds this LM head/)
  })

  it('persists; old saves without the field load untied', () => {
    const g = load()
    const doc = toDocument(g.nodes, g.edges, tiedHp)
    expect(parseDocument(JSON.parse(JSON.stringify(doc))).hyperparams.tie_embeddings).toBe(true)
    const old = JSON.parse(JSON.stringify(doc))
    delete old.hyperparams.tie_embeddings
    expect(parseDocument(old).hyperparams.tie_embeddings).toBe(false)
  })
})
