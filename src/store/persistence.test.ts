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

  it('rejects Phase-1 (v1) saves so the app falls back to the default graph', () => {
    expect(() => parseDocument({ app: 'llm-canvas', version: 1, nodes: [], edges: [] })).toThrow('Unsupported file version')
  })

  it('fills missing/invalid hyperparams with defaults', () => {
    const doc = { ...cs336Document(), hyperparams: { d_model: 768, num_heads: -3, dtype: 'int4' } }
    const hp = parseDocument(doc).hyperparams
    expect(hp.d_model).toBe(768)
    expect(hp.num_heads).toBe(16)
    expect(hp.dtype).toBe('fp32')
  })
})
