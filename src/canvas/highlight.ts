// Canvas highlight driven by the analysis panel (click a parameter category → its parts glow,
// everything else dims). Selectors return a primitive so nodes only re-render when it changes.
import type { CanvasState } from '../store/useCanvasStore'

export type HighlightState = 'match' | 'dim' | null

export function partHighlight(s: CanvasState, id: string): HighlightState {
  if (!s.highlight) return null
  const p = s.inference.params.parts[id]
  if (!p) return 'dim'
  const match = s.highlight === 'unconnected' ? !p.connected : p.connected && p.category === s.highlight
  return match ? 'match' : 'dim'
}

/** A group matches when any weighted part inside it does. */
export function groupHighlight(s: CanvasState, id: string): HighlightState {
  if (!s.highlight) return null
  return s.inference.params.groups[id]?.keys.includes(s.highlight) ? 'match' : 'dim'
}

/** Glow around a highlighted node, in the category's colour. `w` = ring width (px, or em on scaled cards). */
export function glow(color: string, w = '3px', blur = '18px'): string {
  return `0 0 0 ${w} ${color}, 0 0 ${blur} ${color}`
}
