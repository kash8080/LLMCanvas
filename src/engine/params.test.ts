import { describe, expect, it } from 'vitest'
import { instantiateGroup } from '../canvas/groupTemplates'
import { createPartNode } from '../canvas/nodeFactory'
import type { AppEdge, AppNode } from '../canvas/types'
import { cs336Document } from '../defaults/cs336Graph'
import { inferGraph } from '../store/inference'
import { DEFAULT_HYPERPARAMS } from './hyperparams'
import { paramInsights, standardFormula } from './params'
import type { Hyperparams } from './types'

function load() {
  const doc = cs336Document()
  return { nodes: doc.nodes as AppNode[], edges: doc.edges as AppEdge[] }
}
const report = (g: { nodes: AppNode[]; edges: AppEdge[] }, hp: Hyperparams = DEFAULT_HYPERPARAMS) => inferGraph(g.nodes, g.edges, hp).params

/** Insert a fresh Transformer Block between Block 2 and ln_final. */
function withThirdBlock() {
  const g = load()
  const built = instantiateGroup('transformer_block', { x: 0, y: 6000 }, g.nodes)
  const id = built.nodes[0].id
  const edges = g.edges.filter((e) => !(e.source === 'b2' && e.target === 'ln_final'))
  edges.push(
    { id: 'x1', source: 'b2', sourceHandle: 'out', target: id, targetHandle: 'in' },
    { id: 'x2', source: id, sourceHandle: 'out', target: 'ln_final', targetHandle: 'in' },
  )
  return { nodes: [...g.nodes, ...built.nodes], edges: [...edges, ...built.edges], id }
}

describe('accountParams on the default CS336 graph', () => {
  const r = report(load())

  it('counts the connected model: 16,468,480', () => {
    expect(r.total).toBe(16_468_480)
    expect(r.unconnected).toBe(0)
    expect(r.hasOutput).toBe(true)
  })

  it('breaks params down by category', () => {
    expect(r.byCategory).toEqual({
      embedding: 5_120_000,
      attention: 2 * 1_048_576,
      ffn: 2 * 2_064_384,
      norm: 2 * 2 * 512 + 512,
      lm_head: 5_120_000,
      other: 0,
    })
    expect(r.parts.lm_head.category).toBe('lm_head')
    expect(r.parts['b1.q_proj'].category).toBe('attention')
    expect(r.parts['b2.w2'].category).toBe('ffn')
    expect(r.parts['b1.ln1'].category).toBe('norm')
  })

  it('lists rows in data-flow order: embedding, Block 1, Block 2, ln_final, lm_head', () => {
    expect(r.rows.map((x) => x.id)).toEqual(['embed', 'b1', 'b2', 'ln_final', 'lm_head'])
    expect(r.rows.filter((x) => x.isLayer).map((x) => x.params)).toEqual([3_113_984, 3_113_984])
    expect(r.rows[1].byCategory).toMatchObject({ attention: 1_048_576, ffn: 2_064_384, norm: 1024 })
  })

  it('matches the standard formula with the substituted numbers', () => {
    expect(r.formula.L).toBe(2)
    expect(r.formula.matches).toBe(true)
    expect(r.formula.value).toBe(16_468_480)
    expect(r.formula.substituted).toBe('10000·512 + 2·(4·512² + 3·512·1344 + 2·512) + 512 + 512·10000')
  })

  it('marks which highlight keys each group contains', () => {
    expect(r.groups['b1.attn'].keys).toEqual(['attention'])
    expect(r.groups.b1.keys.sort()).toEqual(['attention', 'ffn', 'norm'])
    expect(r.groups.b1.connected).toBe(3_113_984)
  })

  it('has a factual insight about embedding + LM head', () => {
    const [first, second] = paramInsights(r)
    expect(first).toMatch(/^Embedding \+ LM head = 62% /)
    expect(first).toMatch(/≥ 4 blocks/)
    expect(second).toMatch(/FFN has 1\.97× the weights of attention/)
  })
})

describe('accountParams when the canvas changes', () => {
  it('a wired-in third block → 19,582,464 and 3 layers', () => {
    const g = withThirdBlock()
    const r = report(g)
    expect(r.total).toBe(19_582_464)
    expect(r.rows.filter((x) => x.isLayer).map((x) => x.id)).toEqual(['b1', 'b2', g.id])
    expect(r.formula.L).toBe(3)
    expect(r.formula.matches).toBe(true)
  })

  it('a block that is not wired in is unconnected', () => {
    const g = load()
    const built = instantiateGroup('transformer_block', { x: 3000, y: 0 }, g.nodes)
    const r = report({ nodes: [...g.nodes, ...built.nodes], edges: [...g.edges, ...built.edges] })
    expect(r.total).toBe(16_468_480)
    expect(r.unconnected).toBe(3_113_984)
    expect(r.groups[built.nodes[0].id].keys).toEqual(['unconnected'])
    expect(r.formula.L).toBe(2)
  })

  it('a stray Linear doesn’t change the connected total', () => {
    const g = load()
    const stray = createPartNode('linear', { x: 2000, y: 0 }, 'stray')
    // Even when fed from the model, it's not on the path to Logits / Loss.
    const r = report({ nodes: [...g.nodes, stray], edges: [...g.edges, { id: 's', source: 'embed', sourceHandle: 'out', target: 'stray', targetHandle: 'in' }] })
    expect(r.total).toBe(16_468_480)
    expect(r.unconnected).toBe(512 * 512)
    expect(r.unconnectedIds).toEqual(['stray'])
    expect(r.parts.stray.connected).toBe(false)
    expect(r.formula.matches).toBe(true)
  })

  it('an extra connected Linear is "Other" and breaks the formula match', () => {
    const g = load()
    const extra = createPartNode('linear', { x: 0, y: 0 }, 'extra')
    const edges = g.edges.filter((e) => !(e.source === 'ln_final' && e.target === 'lm_head'))
    edges.push(
      { id: 'a', source: 'ln_final', sourceHandle: 'out', target: 'extra', targetHandle: 'in' },
      { id: 'b', source: 'extra', sourceHandle: 'out', target: 'lm_head', targetHandle: 'in' },
    )
    const r = report({ nodes: [...g.nodes, extra], edges })
    expect(r.total).toBe(16_468_480 + 512 * 512)
    expect(r.byCategory.other).toBe(512 * 512)
    expect(r.formula.matches).toBe(false)
    expect(r.formula.diffs).toEqual([{ category: 'other', actual: 512 * 512, expected: 0 }])
  })

  it('follows the hyperparams', () => {
    const r = report(load(), { ...DEFAULT_HYPERPARAMS, d_model: 768, d_ff: 2048, vocab_size: 32000 })
    expect(r.total).toBe(32000 * 768 * 2 + 2 * (4 * 768 ** 2 + 3 * 768 * 2048 + 2 * 768) + 768)
    expect(r.formula.matches).toBe(true)
  })

  it('without a Logits / Loss part every part counts', () => {
    const stray = createPartNode('linear', { x: 0, y: 0 }, 'stray')
    const r = report({ nodes: [stray], edges: [] })
    expect(r.hasOutput).toBe(false)
    expect(r.total).toBe(512 * 512)
  })
})

describe('standardFormula', () => {
  it('4 layers with the CS336 defaults = 22,696,448', () => {
    expect(standardFormula(DEFAULT_HYPERPARAMS, 4).value).toBe(22_696_448)
  })
})
