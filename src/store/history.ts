// Undo / redo history (pure, no React): a stack of whole-graph snapshots.
//
// The store records the state *before* each undoable edit. Snapshots are cheap: nodes / edges are
// immutable arrays, so a snapshot just keeps references to the previous arrays.
//
// Coalescing: edits that pass the same `key` (e.g. `title:<id>` while typing a name, or
// `hp:d_model` while typing a number) within `COALESCE_MS` of the previous one extend the same
// entry instead of adding one per keystroke. The window slides: each coalesced edit restarts it.

export interface History<T> {
  past: T[]
  future: T[]
  /** Key of the most recent record (null after undo / redo or a record without a key). */
  lastKey: string | null
  /** Time of the most recent record or coalesced edit (ms). */
  lastAt: number
}

/** Max undo steps kept; the oldest are dropped. */
export const HISTORY_LIMIT = 100
/** Same-key edits closer together than this merge into one undo step. */
export const COALESCE_MS = 1000

export function emptyHistory<T>(): History<T> {
  return { past: [], future: [], lastKey: null, lastAt: 0 }
}

export interface RecordOptions {
  /** Edits with the same key in quick succession share one entry. */
  key?: string
  now?: number
  limit?: number
  windowMs?: number
}

/** Remember `before` (the state before an edit) as an undo step; clears the redo stack. */
export function record<T>(h: History<T>, before: T, opts: RecordOptions = {}): History<T> {
  const { key, now = Date.now(), limit = HISTORY_LIMIT, windowMs = COALESCE_MS } = opts
  if (key != null && key === h.lastKey && now - h.lastAt <= windowMs && h.past.length > 0) {
    return { ...h, future: [], lastAt: now }
  }
  const past = [...h.past, before]
  return { past: past.length > limit ? past.slice(past.length - limit) : past, future: [], lastKey: key ?? null, lastAt: now }
}

/** Step back: returns the state to restore, or null if there is nothing to undo. */
export function undo<T>(h: History<T>, current: T): { history: History<T>; state: T } | null {
  if (h.past.length === 0) return null
  const state = h.past[h.past.length - 1]
  return { state, history: { past: h.past.slice(0, -1), future: [...h.future, current], lastKey: null, lastAt: 0 } }
}

/** Step forward again after an undo, or null if there is nothing to redo. */
export function redo<T>(h: History<T>, current: T): { history: History<T>; state: T } | null {
  if (h.future.length === 0) return null
  const state = h.future[h.future.length - 1]
  return { state, history: { past: [...h.past, current], future: h.future.slice(0, -1), lastKey: null, lastAt: 0 } }
}
