import { describe, expect, it } from 'vitest'
import { emptyHistory, record, redo, undo } from './history'

describe('history', () => {
  it('undo / redo walk back and forth through recorded states', () => {
    let h = emptyHistory<number>()
    h = record(h, 0, { now: 0 }) // 0 → 1
    h = record(h, 1, { now: 10_000 }) // 1 → 2
    let cur = 2
    const u1 = undo(h, cur)!
    expect(u1.state).toBe(1)
    const u2 = undo(u1.history, u1.state)!
    expect(u2.state).toBe(0)
    expect(undo(u2.history, u2.state)).toBeNull()
    const r1 = redo(u2.history, u2.state)!
    expect(r1.state).toBe(1)
    const r2 = redo(r1.history, r1.state)!
    expect(r2.state).toBe(2)
    expect(redo(r2.history, r2.state)).toBeNull()
    cur = r2.state
    expect(cur).toBe(2)
  })

  it('a new edit after undo clears the redo stack', () => {
    let h = record(emptyHistory<string>(), 'a', { now: 0 })
    const u = undo(h, 'b')!
    expect(u.history.future).toEqual(['b'])
    h = record(u.history, 'a', { now: 5000 })
    expect(h.future).toEqual([])
    expect(h.past).toEqual(['a'])
  })

  it('coalesces same-key edits within the window (sliding), not other keys', () => {
    let h = emptyHistory<string>()
    h = record(h, 'v0', { key: 'title:x', now: 0 })
    h = record(h, 'v1', { key: 'title:x', now: 400 })
    h = record(h, 'v2', { key: 'title:x', now: 1300 }) // 900 ms after the previous one → still the same entry
    expect(h.past).toEqual(['v0'])
    h = record(h, 'v3', { key: 'title:x', now: 3000 }) // a pause → new entry
    expect(h.past).toEqual(['v0', 'v3'])
    h = record(h, 'v4', { key: 'title:y', now: 3100 }) // different field → new entry
    h = record(h, 'v5', { now: 3150 }) // no key → never coalesces
    h = record(h, 'v6', { now: 3160 })
    expect(h.past).toEqual(['v0', 'v3', 'v4', 'v5', 'v6'])
  })

  it('does not coalesce across an undo', () => {
    let h = record(emptyHistory<string>(), 'a', { key: 'k', now: 0 })
    h = undo(h, 'b')!.history
    h = record(h, 'a', { key: 'k', now: 10 })
    expect(h.past).toEqual(['a'])
    expect(h.lastKey).toBe('k')
  })

  it('caps the number of steps, dropping the oldest', () => {
    let h = emptyHistory<number>()
    for (let i = 0; i < 105; i++) h = record(h, i, { now: i * 10_000 })
    expect(h.past).toHaveLength(100)
    expect(h.past[0]).toBe(5)
    expect(h.past[99]).toBe(104)
    h = record(h, 999, { now: 2_000_000, limit: 3 })
    expect(h.past).toEqual([103, 104, 999])
  })
})
