import { describe, expect, it } from 'vitest'
import { numLayers } from '../canvas/groupTemplates'
import type { AppNode } from '../canvas/types'
import { cs336Document } from '../defaults/cs336Graph'
import { flattenGroups } from '../engine/groups'
import { DEFAULT_HYPERPARAMS } from '../engine/hyperparams'
import { formatConcrete } from '../engine/shape'
import type { GraphModel } from '../engine/types'
import { inferGraph } from './inference'
import { useCanvasStore, withoutLoneProxies } from './useCanvasStore'

const total = (r: { nodes: Record<string, { paramCount: { total: number } }> }) =>
  Object.values(r.nodes).reduce((acc, n) => acc + n.paramCount.total, 0)

function defaultInference() {
  const doc = cs336Document()
  return inferGraph(doc.nodes as AppNode[], doc.edges, DEFAULT_HYPERPARAMS)
}

describe('flattenGroups', () => {
  it('redirects edges on a group’s outer ports onto its proxies', () => {
    const g: GraphModel = {
      nodes: [
        { id: 'a', type: 'silu', params: {} },
        { id: 'G.in', type: 'group_input', params: {}, parentId: 'G' },
        { id: 'x', type: 'silu', params: {}, parentId: 'G' },
        { id: 'G.out', type: 'group_output', params: {}, parentId: 'G' },
        { id: 'b', type: 'silu', params: {} },
      ],
      edges: [
        { id: 'e1', source: 'a', sourceHandle: 'out', target: 'G', targetHandle: 'in' },
        { id: 'e2', source: 'G.in', sourceHandle: 'out', target: 'x', targetHandle: 'in' },
        { id: 'e3', source: 'x', sourceHandle: 'out', target: 'G.out', targetHandle: 'in' },
        { id: 'e4', source: 'G', sourceHandle: 'out', target: 'b', targetHandle: 'in' },
      ],
    }
    const flat = flattenGroups(g).edges
    expect(flat[0]).toMatchObject({ id: 'e1', source: 'a', target: 'G.in', targetHandle: 'in' })
    expect(flat[1]).toBe(g.edges[1])
    expect(flat[3]).toMatchObject({ id: 'e4', source: 'G.out', sourceHandle: 'out', target: 'b' })
  })
})

describe('grouped default graph', () => {
  const r = defaultInference()

  it('infers with no errors and the same shapes / total as the flat graph (16,468,480)', () => {
    const bad = Object.entries(r.nodes).filter(([, n]) => n.status !== 'ok')
    expect(bad.map(([id, n]) => `${id}: ${n.status} ${n.errors.join('; ')}`)).toEqual([])
    expect(total(r)).toBe(16_468_480)
    expect(formatConcrete(r.nodes['b1.sdpa'].outputShapes[0]!)).toBe('32×16×256×32')
    expect(formatConcrete(r.nodes.logits.outputShapes[0]!)).toBe('32×256×10000')
    expect(Object.values(r.edges).every((s) => s !== null)).toBe(true)
    expect(formatConcrete(r.edges['e:b1.out->b2.in']!)).toBe('32×256×512')
  })

  it('summarises groups: params = sum of children, shapes at the outer ports', () => {
    for (const b of ['b1', 'b2']) {
      expect(r.groups[b].params).toBe(3_113_984)
      expect(r.groups[`${b}.attn`].params).toBe(1_048_576)
      expect(r.groups[`${b}.ffn`].params).toBe(2_064_384)
      expect(r.groups[b].status).toBe('ok')
    }
    expect(formatConcrete(r.groups.b1.inShape!)).toBe('32×256×512')
    expect(formatConcrete(r.groups.b1.outShape!)).toBe('32×256×512')
    expect(r.groups.b1.contains).toEqual(['ln1', 'attn', 'x + attn', 'ln2', 'ffn', 'x + ffn'])
  })

  it('reports errors from inside nested groups on every enclosing group', () => {
    const doc = cs336Document()
    const q = doc.nodes.find((n) => n.id === 'b1.q_proj')! as AppNode & { type: 'part' }
    q.data.params = { ...q.data.params, in_features: { value: 768 } }
    const r2 = inferGraph(doc.nodes as AppNode[], doc.edges, DEFAULT_HYPERPARAMS)
    expect(r2.groups['b1.attn'].status).toBe('error')
    expect(r2.groups.b1.status).toBe('error')
    expect(r2.groups.b1.errors[0]).toMatch(/^attn › q_proj: Linear.in_features = 768/)
    expect(r2.groups.b2.status).toBe('unknown')
  })

  it('num_layers = number of Transformer Block groups', () => {
    expect(numLayers(cs336Document().nodes as AppNode[])).toBe(2)
  })
})

describe('store: group templates, duplicate, delete', () => {
  it('dropping a Transformer Block template adds a 3rd layer (19,582,464 params)', () => {
    const store = useCanvasStore.getState()
    store.resetCanvas()
    store.addNode('group:transformer_block', { x: 2000, y: 0 })
    const s = useCanvasStore.getState()
    expect(numLayers(s.nodes)).toBe(3)
    expect(total(s.inference)).toBe(19_582_464)
    const block = s.nodes.find((n) => n.selected)!
    expect(block.type === 'group' && block.data.title).toBe('Block 3')
    expect(s.inference.groups[block.id].params).toBe(3_113_984)
    // Not wired in yet: its input proxy reports the missing input.
    expect(s.inference.groups[block.id].errors).toEqual(['Input not connected'])
    // Every child comes after its parent.
    const index = new Map(s.nodes.map((n, i) => [n.id, i]))
    expect(s.nodes.every((n) => !n.parentId || index.get(n.parentId)! < index.get(n.id)!)).toBe(true)
  })

  it('duplicating a block copies its children and internal edges', () => {
    const store = useCanvasStore.getState()
    store.resetCanvas()
    const before = useCanvasStore.getState()
    useCanvasStore.setState({ nodes: before.nodes.map((n) => (n.id === 'b2' ? { ...n, selected: true } : n)) })
    useCanvasStore.getState().duplicateSelection()
    const s = useCanvasStore.getState()
    const copy = s.nodes.find((n) => n.selected)!
    expect(copy.type === 'group' && copy.data.title).toBe('Block 3')
    const inside = (id: string): boolean => {
      const n = s.nodes.find((x) => x.id === id)
      return !!n?.parentId && (n.parentId === copy.id || inside(n.parentId))
    }
    const copiedChildren = s.nodes.filter((n) => inside(n.id))
    expect(copiedChildren.length).toBe(before.nodes.filter((n) => n.parentId?.startsWith('b2')).length)
    expect(s.edges.length - before.edges.length).toBe(before.edges.filter((e) => e.source.startsWith('b2.') && e.target.startsWith('b2.')).length)
    expect(numLayers(s.nodes)).toBe(3)
    expect(total(s.inference)).toBe(19_582_464)
  })

  it('a proxy is only deleted together with its group', () => {
    const doc = cs336Document()
    const nodes = doc.nodes as AppNode[]
    const proxy = nodes.find((n) => n.id === 'b1.in')!
    const ln1 = nodes.find((n) => n.id === 'b1.ln1')!
    const edge = doc.edges.find((e) => e.source === 'b1.in' && e.target === 'b1.ln1')!
    expect(withoutLoneProxies({ nodes: [proxy, ln1], edges: [edge] })).toEqual({ nodes: [ln1], edges: [edge] })
    const block = nodes.find((n) => n.id === 'b1')!
    expect(withoutLoneProxies({ nodes: [block, proxy], edges: [] }).nodes).toHaveLength(2)
  })
})
