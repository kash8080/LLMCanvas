import { describe, expect, it } from 'vitest'
import { demoDocument } from '../defaults/demoGraph'
import { parseDocument, toDocument } from './persistence'
import type { AppNode } from '../canvas/types'

describe('persistence', () => {
  it('round-trips a document and strips UI-only fields', () => {
    const doc = demoDocument()
    const nodes = doc.nodes.map((n) => ({ ...n, selected: true, measured: { width: 1, height: 1 } })) as AppNode[]
    const out = toDocument(nodes, doc.edges)
    expect(out.nodes[0]).not.toHaveProperty('selected')
    expect(out.nodes[0]).not.toHaveProperty('measured')
    expect(parseDocument(JSON.parse(JSON.stringify(out)))).toEqual(out)
  })

  it('rejects invalid files with a readable message', () => {
    expect(() => parseDocument({ foo: 1 })).toThrow('Not an LLM Canvas file')
    const bad = { ...demoDocument(), edges: [{ id: 'x', source: 'nope', target: 'part-a' }] }
    expect(() => parseDocument(bad)).toThrow('missing node')
  })
})
