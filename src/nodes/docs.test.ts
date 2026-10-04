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

  it('every palette part and group points at the CS336 code', () => {
    for (const [name, docs] of entries) if (!name.startsWith('part group_')) expect(docs.cs336Ref, name).toBeDefined()
  })
})
