// Canvas highlight driven by the analysis panel: click a parameter category (Parameters tab) or an
// activation category (Memory tab) → its parts glow, everything else dims. Also the optional memory
// "heat" tint. Selectors return a primitive so nodes only re-render when it changes.
import { isMemKey, type MemoryCategory } from '../engine/memory'
import { selectMemory, type CanvasState } from '../store/useCanvasStore'

export type HighlightState = 'match' | 'dim' | null

export function partHighlight(s: CanvasState, id: string): HighlightState {
  const key = s.highlight
  if (!key) return null
  if (isMemKey(key)) return selectMemory(s).partKeys[id]?.includes(key.slice(4) as MemoryCategory) ? 'match' : 'dim'
  const p = s.inference.params.parts[id]
  if (!p) return 'dim'
  const match = key === 'unconnected' ? !p.connected : p.connected && p.category === key
  return match ? 'match' : 'dim'
}

/** A group matches when any part inside it does. */
export function groupHighlight(s: CanvasState, id: string): HighlightState {
  const key = s.highlight
  if (!key) return null
  if (isMemKey(key)) return selectMemory(s).groups[id]?.keys.includes(key.slice(4) as MemoryCategory) ? 'match' : 'dim'
  return s.inference.params.groups[id]?.keys.includes(key) ? 'match' : 'dim'
}

/**
 * Memory heat: 0…1 share of the largest part's activation bytes (square-root scaled so mid-sized
 * tensors stay visible), or null when the heat view is off / the part holds nothing. Rounded so
 * the selector result is stable.
 */
export function partHeat(s: CanvasState, id: string): number | null {
  if (!s.memoryHeat) return null
  const byPart = selectMemory(s).activations.byPart
  const bytes = byPart[id] ?? 0
  if (bytes <= 0) return null
  const max = Math.max(...Object.values(byPart))
  return Math.round(Math.sqrt(bytes / max) * 20) / 20
}

/** Heat of a group (collapsed card): its activation bytes relative to the largest group's. */
export function groupHeat(s: CanvasState, id: string): number | null {
  if (!s.memoryHeat) return null
  const groups = selectMemory(s).groups
  const bytes = groups[id]?.activations ?? 0
  if (bytes <= 0) return null
  const max = Math.max(...Object.values(groups).map((g) => g.activations))
  return Math.round(Math.sqrt(bytes / max) * 20) / 20
}

/** Opaque background (white + red tint) for a heat value. */
export function heatColor(heat: number): string {
  const c = `rgba(239, 68, 68, ${(0.08 + heat * 0.55).toFixed(2)})`
  return `linear-gradient(${c}, ${c}), #fff`
}

/** Glow around a highlighted node, in the category's colour. `w` = ring width (px, or em on scaled cards). */
export function glow(color: string, w = '3px', blur = '18px'): string {
  return `0 0 0 ${w} ${color}, 0 0 ${blur} ${color}`
}
