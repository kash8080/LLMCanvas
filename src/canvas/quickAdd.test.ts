import { describe, expect, it } from 'vitest'
import { filterQuickAdd, quickAddItems, quickAddPosition } from './quickAdd'

const ids = (items: { item: string }[]) => items.map((i) => i.item)

describe('quick-add items', () => {
  it('from an output: only things with an input (no Data Batch), plus the three groups', () => {
    const items = ids(quickAddItems('source'))
    expect(items).not.toContain('part:data_batch')
    expect(items).toContain('part:linear')
    expect(items).toContain('part:loss')
    expect(items.filter((i) => i.startsWith('group:'))).toEqual(['group:transformer_block', 'group:mha', 'group:swiglu'])
    expect(items).not.toContain('sticky')
  })

  it('from an input: only things with an output (no Loss)', () => {
    const items = ids(quickAddItems('target'))
    expect(items).toContain('part:data_batch')
    expect(items).not.toContain('part:loss')
    expect(items).toContain('group:mha')
  })

  it('inside a group: no group templates; from the empty canvas: everything incl. annotations', () => {
    expect(ids(quickAddItems('source', true)).some((i) => i.startsWith('group:'))).toBe(false)
    const all = ids(quickAddItems(null))
    expect(all).toContain('part:data_batch')
    expect(all).toContain('part:loss')
    expect(all).toContain('sticky')
    expect(all).toContain('textbox')
  })

  it('never lists the group proxies', () => {
    expect(ids(quickAddItems(null)).some((i) => i.includes('group_input') || i.includes('group_output'))).toBe(false)
  })

  it('filters by words, label prefix first', () => {
    const items = quickAddItems('source')
    expect(ids(filterQuickAdd(items, 'lin'))[0]).toBe('part:linear')
    expect(ids(filterQuickAdd(items, 'RMS'))).toEqual(['part:rmsnorm'])
    expect(ids(filterQuickAdd(items, 'group swig'))).toEqual(['group:swiglu'])
    expect(filterQuickAdd(items, '  ')).toBe(items)
    expect(filterQuickAdd(items, 'zzz')).toEqual([])
  })

  it('places the connected port at the drop point', () => {
    expect(quickAddPosition('part:linear', { x: 100, y: 200 }, 'source')).toEqual({ x: 12, y: 200 })
    expect(quickAddPosition('part:linear', { x: 100, y: 200 }, 'target')).toEqual({ x: 12, y: 136 })
    expect(quickAddPosition('part:linear', { x: 100, y: 200 }, null)).toEqual({ x: 12, y: 168 })
  })
})
