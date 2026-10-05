import { describe, expect, it } from 'vitest'
import type { NodeDocs } from '../engine/types'
import { GROUP_DEF_LIST } from './groups'
import { nodeRegistry } from './registry'

const entries: [string, NodeDocs, string[]][] = [
  ...Object.values(nodeRegistry).map((d) => [`part ${d.type}`, d.docs, d.params.map((p) => p.key)] as [string, NodeDocs, string[]]),
  ...GROUP_DEF_LIST.map((g) => [`group ${g.type}`, g.docs, []] as [string, NodeDocs, string[]]),
]

describe('drawer docs', () => {
  it.each(entries)('%s has an overview and ≥ 3 points to remember', (_name, docs) => {
    expect(docs.overview.trim().length).toBeGreaterThan(20)
    expect(docs.pointsToRemember.length).toBeGreaterThanOrEqual(3)
    for (const p of docs.pointsToRemember) expect(p.trim()).not.toBe('')
  })

  it.each(entries)('%s only documents params it has', (_name, docs, paramKeys) => {
    for (const key of Object.keys(docs.paramHelp ?? {})) expect(paramKeys).toContain(key)
  })

  it('every palette part and group points at the CS336 code, or says why it is not there', () => {
    for (const [name, docs] of entries) {
      if (name.startsWith('part group_')) continue
      expect(docs.cs336Ref ?? docs.cs336Note, name).toBeDefined()
      if (!docs.cs336Ref) expect(docs.cs336Note, name).toMatch(/Not in the CS336 code.*variant for comparison/)
    }
  })

  it('the Phase 7c variants are registered and documented', () => {
    for (const t of ['layernorm', 'gelu', 'relu']) {
      expect(nodeRegistry[t], t).toBeDefined()
      expect(nodeRegistry[t].docs.cs336Note).toBeDefined()
    }
    expect(nodeRegistry.layernorm.docs.overview).toMatch(/RMSNorm/)
    expect(GROUP_DEF_LIST.find((g) => g.type === 'ffn')?.docs.cs336Ref?.file).toBe('SiLU.py')
  })
})
