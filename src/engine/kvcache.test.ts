import { describe, expect, it } from 'vitest'
import { instantiateGroup } from '../canvas/groupTemplates'
import type { AppEdge, AppNode } from '../canvas/types'
import { cs336Document } from '../defaults/cs336Graph'
import { inferGraph, memoryInput } from '../store/inference'
import { DEFAULT_HYPERPARAMS } from './hyperparams'
import { estimateMemory, kvCacheOf, memoryFormulas, memoryInsights, type GenerationSettings, type MemoryMode } from './memory'
import type { Hyperparams } from './types'

type Graph = { nodes: AppNode[]; edges: AppEdge[] }
const load = (): Graph => {
  const doc = cs336Document()
  return { nodes: doc.nodes as AppNode[], edges: doc.edges as AppEdge[] }
}
/** Insert a fresh Transformer Block between Block 2 and ln_final. */
function withThirdBlock(): Graph {
  const g = load()
  const built = instantiateGroup('transformer_block', { x: 0, y: 6000 }, g.nodes)
  const id = built.nodes[0].id
  const edges = g.edges.filter((e) => !(e.source === 'b2' && e.target === 'ln_final'))
  edges.push(
    { id: 'x1', source: 'b2', sourceHandle: 'out', target: id, targetHandle: 'in' },
    { id: 'x2', source: id, sourceHandle: 'out', target: 'ln_final', targetHandle: 'in' },
  )
  return { nodes: [...g.nodes, ...built.nodes], edges: [...edges, ...built.edges] }
}

const input = (gen: GenerationSettings | null = {}, hp: Partial<Hyperparams> = {}, g: Graph = load(), mode: MemoryMode = 'forward') => {
  const h = { ...DEFAULT_HYPERPARAMS, ...hp }
  return memoryInput(inferGraph(g.nodes, g.edges, h), h, mode, false, gen)
}
const gen = (settings: GenerationSettings = {}, hp: Partial<Hyperparams> = {}, g?: Graph) => estimateMemory(input(settings, hp, g))

const KV = 2 * 2 * 32 * 256 * 512 * 4 // 67,108,864
const P = 16_468_480
const ROPE = 2 * 2 * 256 * 16 * 4

describe('KV cache (generation, forward mode)', () => {
  const r = gen()
  const g = r.generation!

  it('defaults: 2 · L · B · T · d_model · bytes = 2·2·32·256·512·4 = 67,108,864 B', () => {
    expect(KV).toBe(67_108_864)
    expect(g.kv.total).toBe(KV)
    expect(r.byComponent.kv_cache).toBe(KV)
    expect(g.length).toBe(256)
    expect(g.maxLength).toBe(256)
    expect(g.batch).toBe(32)
  })

  it('one K and one V cache per attention layer, B × H × T_cache × d_head each', () => {
    expect(g.kv.items.map((x) => x.sdpa)).toEqual(['b1.sdpa', 'b2.sdpa'])
    expect(g.kv.items.map((x) => x.row)).toEqual(['b1', 'b2'])
    expect(g.kv.items[0].groups).toEqual(['b1.attn', 'b1'])
    expect(g.kv.items[0].kShape.dims.map((d) => d.size)).toEqual([32, 16, 256, 32])
    expect(g.kv.items[0].vShape.dims.map((d) => d.size)).toEqual([32, 16, 256, 32])
    expect(g.kv.items[0].bytes).toBe(KV / 2)
    expect(kvCacheOf(r, 'b1.sdpa')).toBe(KV / 2)
    expect(kvCacheOf(r, 'b1.attn')).toBe(KV / 2)
    expect(kvCacheOf(r, 'b2')).toBe(KV / 2)
    expect(kvCacheOf(r, 'lm_head')).toBe(0)
  })

  it('per-token cost: 2 · L · d_model · bytes = 8,192 B per sequence', () => {
    expect(g.kv.perToken).toBe(8192)
    expect(g.kv.items[0].perToken).toBe(4096)
  })

  it('scales linearly with T_cache, B and the number of layers', () => {
    expect(gen({ length: 128 }).generation!.kv.total).toBe(KV / 2)
    expect(gen({ batch: 8 }).generation!.kv.total).toBe(KV / 4)
    expect(gen({}, { batch_size: 64 }).generation!.kv.total).toBe(KV * 2) // B_gen follows batch_size by default
    const three = gen({}, {}, withThirdBlock())
    expect(three.generation!.kv.total).toBe(KV * 1.5)
    expect(three.generation!.kv.perToken).toBe(3 * 4096)
  })

  it('T_cache defaults to context_length and is clamped to 1 … context_length', () => {
    expect(gen({}, { context_length: 512 }).generation!.kv.total).toBe(KV * 2)
    expect(gen({ length: 1000 }).generation!.length).toBe(256)
    expect(gen({ length: 0 }).generation!.length).toBe(1)
  })

  it('bf16 halves the cache; weight tying does not change it', () => {
    expect(gen({}, { dtype: 'bf16' }).generation!.kv.total).toBe(KV / 2)
    expect(gen({}, { tie_embeddings: true }).generation!.kv.total).toBe(KV)
  })

  it('total = weights + buffers + KV cache + one decode step; no gradients / optimizer', () => {
    expect(r.byComponent.weights).toBe(P * 4)
    expect(r.byComponent.buffers).toBe(ROPE)
    expect(r.byComponent.gradients).toBe(0)
    expect(r.byComponent.optimizer).toBe(0)
    expect(r.total).toBe(P * 4 + ROPE + KV + r.activations.total)
    expect(r.total).toBeGreaterThan(r.byComponent.weights)
  })

  it('decode step: T = 1, attention probs B × H × 1 × T_cache', () => {
    const a = r.activations
    // lm_head: input 32·1·512 + logits 32·1·10000 (fp32)
    expect(a.peakPart).toBe('lm_head')
    expect(a.total).toBe(32 * 512 * 4 + 32 * 10000 * 4)
    // SDPA: Q, K, V, out (32·16·1·32 each) + probs 32·16·1·256
    expect(a.byPart['b1.sdpa']).toBe(4 * 32 * 16 * 32 * 4 + 32 * 16 * 256 * 4)
    expect(gen({ length: 128 }).activations.byPart['b1.sdpa']).toBe(4 * 32 * 16 * 32 * 4 + 32 * 16 * 128 * 4)
    // Far below the plain forward pass over the full context.
    expect(a.total).toBeLessThan(estimateMemory(input(null)).activations.total)
  })

  it('only in forward mode; plain estimates have kv_cache = 0', () => {
    const train = estimateMemory(input({}, {}, load(), 'train'))
    expect(train.generation).toBeUndefined()
    expect(train.byComponent.kv_cache).toBe(0)
    expect(estimateMemory(input(null)).byComponent.kv_cache).toBe(0)
  })

  it('formula and insights', () => {
    expect(memoryFormulas(r).kv_cache).toBe('K + V: 2 · L · B · T_cache · d_model · bytes = 2 · 2 · 32 · 256 · 512 · 4')
    const lines = memoryInsights(input(), r)
    expect(lines[0]).toMatch(/^Each generated token adds 8,192 B per sequence/)
    expect(lines.some((l) => l.includes('O(T²)') && l.includes('Decoding.py'))).toBe(true)
    expect(lines.some((l) => l.includes('GQA / MQA') && l.includes('16×'))).toBe(true)
    // Defaults: 67.1 MB of cache vs 65.9 MB of weights — the crossover is B · T_cache = 65,873,920 / 8,192 ≈ 8,042 tokens.
    expect(lines.some((l) => l.includes('already bigger than the weights') && l.includes('8,042'))).toBe(true)
    const short = memoryInsights(input({ length: 64 }), gen({ length: 64 }))
    expect(short.some((l) => l.startsWith('At T_cache = context_length (256)'))).toBe(true)
    expect(short.some((l) => l.startsWith('The KV cache is 25% of the weights'))).toBe(true)
  })
})
