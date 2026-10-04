import { describe, expect, it } from 'vitest'
import { cs336Document } from '../defaults/cs336Graph'
import { nodeRegistry } from '../nodes/registry'
import { toGraphModel } from '../store/inference'
import type { AppNode } from '../canvas/types'
import { flattenGroups } from './groups'
import { DEFAULT_HYPERPARAMS, validateHyperparams } from './hyperparams'
import { inferShapes, wouldCreateCycle } from './infer'
import { formatConcrete, formatSymbolic } from './shape'
import type { GraphModel, Hyperparams } from './types'

/** The default (grouped) graph, flattened onto the group proxies like the app does before inference. */
function defaultGraph(): GraphModel {
  const doc = cs336Document()
  return flattenGroups(toGraphModel(doc.nodes as AppNode[], doc.edges))
}

const run = (g: GraphModel, hp: Hyperparams = DEFAULT_HYPERPARAMS) => inferShapes(g, hp, nodeRegistry)
const out = (r: ReturnType<typeof run>, id: string, i = 0) => {
  const s = r.nodes[id].outputShapes[i]
  return s ? formatConcrete(s) : null
}

describe('inferShapes on the default CS336 graph', () => {
  const r = run(defaultGraph())

  it('has no errors and every part is ok', () => {
    const bad = Object.entries(r.nodes).filter(([, n]) => n.status !== 'ok')
    expect(bad.map(([id, n]) => `${id}: ${n.status} ${n.errors.join('; ')}`)).toEqual([])
  })

  it('produces the expected shapes at key points', () => {
    expect(out(r, 'data', 0)).toBe('32×256')
    expect(r.nodes.data.outputShapes[0]!.dtype).toBe('int64')
    expect(out(r, 'embed')).toBe('32×256×512')
    expect(formatSymbolic(r.nodes.embed.outputShapes[0]!)).toBe('B × T × d_model')
    expect(out(r, 'b1.split_q')).toBe('32×16×256×32')
    expect(formatSymbolic(r.nodes['b1.split_q'].outputShapes[0]!)).toBe('B × H × T × d_head')
    expect(out(r, 'b1.merge')).toBe('32×256×512')
    expect(formatSymbolic(r.nodes['b1.merge'].outputShapes[0]!)).toBe('B × T × d_model')
    expect(out(r, 'b1.w1')).toBe('32×256×1344')
    expect(out(r, 'b2.add2')).toBe('32×256×512')
    expect(out(r, 'logits')).toBe('32×256×10000')
    expect(formatSymbolic(r.nodes.logits.outputShapes[0]!)).toBe('B × T × V')
    expect(r.nodes.xent.outputShapes[0]).toEqual({ dims: [], dtype: 'float' })
    expect(r.nodes.loss.outputShapes).toEqual([])
  })

  it('labels every edge with a shape', () => {
    expect(Object.values(r.edges).every((s) => s !== null)).toBe(true)
  })

  it('SDPA saves the B×H×T×T attention probabilities for backward', () => {
    const n = r.nodes['b1.sdpa']
    const saved = nodeRegistry.sdpa.savedForBackward({
      inputs: n.inputShapes as never,
      outputs: n.outputShapes as never,
      p: n.resolved,
      hp: DEFAULT_HYPERPARAMS,
    })
    const probs = saved.find((s) => s.name === 'attention probs')!
    expect(formatConcrete(probs.shape)).toBe('32×16×256×256')
    expect(saved.filter((s) => s.which === 'input').map((s) => s.port)).toEqual(['q', 'k', 'v'])
  })

  it('total param count = 16,468,480 (2 blocks, no weight tying)', () => {
    const total = Object.values(r.nodes).reduce((acc, n) => acc + n.paramCount.total, 0)
    expect(total).toBe(16_468_480)
    expect(r.nodes.embed.paramCount.total).toBe(5_120_000)
    expect(r.nodes.lm_head.paramCount.total).toBe(5_120_000)
    expect(r.nodes['b1.w1'].paramCount.tensors[0].dims.map((d) => d.size)).toEqual([1344, 512])
  })
})

describe('inferShapes errors', () => {
  it('Linear in_features mismatch → error, downstream unknown (not red)', () => {
    const g = defaultGraph()
    const q = g.nodes.find((n) => n.id === 'b1.q_proj')!
    q.params = { ...q.params, in_features: { value: 768 } }
    const r = run(g)
    expect(r.nodes['b1.q_proj'].status).toBe('error')
    expect(r.nodes['b1.q_proj'].errors[0]).toBe('Linear.in_features = 768 but input last dim is 512 (d_model)')
    expect(r.nodes['b1.split_q'].status).toBe('unknown')
    expect(r.nodes['b1.sdpa'].status).toBe('unknown')
    expect(r.nodes.loss.status).toBe('unknown')
    expect(r.nodes['b1.k_proj'].status).toBe('ok')
    expect(r.edges['e:b1.q_proj.out->b1.split_q.in']).toBeNull()
  })

  it('d_model not divisible by num_heads → split heads error', () => {
    const hp = { ...DEFAULT_HYPERPARAMS, d_model: 500 }
    const r = run(defaultGraph(), hp)
    expect(r.nodes['b1.split_q'].status).toBe('error')
    expect(r.nodes['b1.split_q'].errors[0]).toMatch(/d_model = 500 is not divisible by num_heads = 16/)
    expect(r.nodes['b1.rope_q'].status).toBe('unknown')
    expect(validateHyperparams(hp)).toContain('d_model (500) is not divisible by num_heads (16)')
  })

  it('odd head dim → RoPE error', () => {
    const hp = { ...DEFAULT_HYPERPARAMS, d_model: 528, num_heads: 16 } // d_head = 33
    const r = run(defaultGraph(), hp)
    expect(r.nodes['b1.split_q'].status).toBe('ok')
    expect(r.nodes['b1.rope_q'].errors).toContain('RoPE needs an even head_dim, got 33')
  })

  it('seq_len > context_length → RoPE error', () => {
    const g = defaultGraph()
    const data = g.nodes.find((n) => n.id === 'data')!
    data.params = { ...data.params, seq_len: { value: 300 } }
    const r = run(g)
    expect(r.nodes['b1.rope_q'].errors[0]).toMatch(/sequence length 300 > RoPE max_seq_len = 256/)
  })

  it('unconnected input → error', () => {
    const g = defaultGraph()
    g.edges = g.edges.filter((e) => e.target !== 'xent' || e.targetHandle !== 'targets')
    const r = run(g)
    expect(r.nodes.xent.status).toBe('error')
    expect(r.nodes.xent.errors).toContain("Input 'targets' is not connected")
    expect(r.nodes.loss.status).toBe('unknown')
  })

  it('residual Add with different shapes → error', () => {
    const g = defaultGraph()
    const w2 = g.nodes.find((n) => n.id === 'b1.w2')!
    w2.params = { ...w2.params, out_features: { value: 768 } }
    const r = run(g)
    expect(r.nodes['b1.add2'].errors[0]).toBe('Add inputs differ: a is 32×256×512, b is 32×256×768')
  })

  it('detects cycles', () => {
    const g: GraphModel = {
      nodes: [
        { id: 'a', type: 'silu', params: {} },
        { id: 'b', type: 'silu', params: {} },
        { id: 'c', type: 'silu', params: {} },
      ],
      edges: [
        { id: 'ab', source: 'a', sourceHandle: 'out', target: 'b', targetHandle: 'in' },
        { id: 'ba', source: 'b', sourceHandle: 'out', target: 'a', targetHandle: 'in' },
        { id: 'bc', source: 'b', sourceHandle: 'out', target: 'c', targetHandle: 'in' },
      ],
    }
    const r = run(g)
    expect(r.nodes.a.status).toBe('error')
    expect(r.nodes.a.errors[0]).toMatch(/cycle/)
    expect(r.nodes.b.status).toBe('error')
    expect(r.nodes.c.status).toBe('unknown')
    expect(wouldCreateCycle([{ source: 'a', target: 'b' }], 'b', 'a')).toBe(true)
    expect(wouldCreateCycle([{ source: 'a', target: 'b' }], 'a', 'b')).toBe(false)
  })

  it('local param overrides are validated', () => {
    const g = defaultGraph()
    const ln = g.nodes.find((n) => n.id === 'ln_final')!
    ln.params = { ...ln.params, eps: { value: -1 } }
    expect(run(g).nodes.ln_final.errors).toContain('eps must be ≥ 0 (got -1)')
  })
})
