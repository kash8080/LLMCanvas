import { describe, expect, it } from 'vitest'
import { cs336Document } from '../defaults/cs336Graph'
import { parseDocument, toDocument } from './persistence'
import type { AppNode } from '../canvas/types'

describe('persistence', () => {
  it('round-trips a document and strips UI-only fields', () => {
    const doc = cs336Document()
    const nodes = doc.nodes.map((n) => ({ ...n, selected: true, measured: { width: 1, height: 1 } })) as AppNode[]
    const out = toDocument(nodes, doc.edges, doc.hyperparams)
    expect(out.nodes[0]).not.toHaveProperty('selected')
    expect(out.nodes[0]).not.toHaveProperty('measured')
    expect(parseDocument(JSON.parse(JSON.stringify(out)))).toEqual(out)
  })

  it('rejects invalid files with a readable message', () => {
    expect(() => parseDocument({ foo: 1 })).toThrow('Not an LLM Canvas file')
    const bad = { ...cs336Document(), edges: [{ id: 'x', source: 'nope', target: 'embed' }] }
    expect(() => parseDocument(bad)).toThrow('missing node')
    const unknownPart = { ...cs336Document(), nodes: [{ id: 'p', type: 'part', position: { x: 0, y: 0 }, data: { partType: 'nope', params: {} } }], edges: [] }
    expect(() => parseDocument(unknownPart)).toThrow('unknown part type')
  })

  it('rejects older (v1 / flat v2) saves so the app falls back to the default graph', () => {
    expect(() => parseDocument({ app: 'llm-canvas', version: 1, nodes: [], edges: [] })).toThrow('Unsupported file version')
    expect(() => parseDocument({ app: 'llm-canvas', version: 2, nodes: [], edges: [] })).toThrow('Unsupported file version')
  })

  it('keeps group children (parentId / extent) and requires parents before children', () => {
    const doc = cs336Document()
    const out = parseDocument(JSON.parse(JSON.stringify(toDocument(doc.nodes as AppNode[], doc.edges, doc.hyperparams))))
    const ln1 = out.nodes.find((n) => n.id === 'b1.ln1')!
    expect(ln1.parentId).toBe('b1')
    expect(ln1.extent).toBe('parent')
    expect(out.nodes.find((n) => n.id === 'b1.q_proj')!.parentId).toBe('b1.attn')
    const swapped = { ...doc, nodes: [...doc.nodes].reverse() }
    expect(() => parseDocument(swapped)).toThrow('not a group listed before it')
  })

  it('fills missing/invalid hyperparams with defaults', () => {
    const doc = { ...cs336Document(), hyperparams: { d_model: 768, num_heads: -3, dtype: 'int4' } }
    const hp = parseDocument(doc).hyperparams
    expect(hp.d_model).toBe(768)
    expect(hp.num_heads).toBe(16)
    expect(hp.dtype).toBe('fp32')
  })
})
