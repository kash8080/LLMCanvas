// Quick-add menu (drop a connection on empty canvas, or "Add part…" in the canvas context menu):
// which items can be created there, search, and where to put the new node. Pure, no React.
import type { XYPosition } from '@xyflow/react'
import { GROUP_DEF_LIST, SUBLAYER_GROUP_TYPES, type GroupType } from '../nodes/groups'
import { CATEGORY_INFO, NODE_DEFS } from '../nodes/registry'
import { paletteItemSize } from './groupTemplates'
import type { PaletteItemId } from './types'

/** Which end of a connection the user dragged from: an output ('source') or an input ('target'). */
export type DragFrom = 'source' | 'target'

export interface QuickAddItem {
  item: PaletteItemId
  label: string
  /** Category / kind shown greyed on the right. */
  group: string
  color: string
}

/**
 * Items that can be created at the drop point. Dragged from an output → things with an input;
 * from an input → things with an output (groups have both). `inside` = type of the group frame the new
 * item goes into (undefined = top level): inside a Transformer Block the sub-layer groups (attention,
 * SwiGLU, non-gated FFN) are offered too, so a block's FFN can be swapped; inside other groups only plain
 * parts. Annotations only when nothing gets connected (frames never inside a group).
 */
export function quickAddItems(from: DragFrom | null, inside?: GroupType): QuickAddItem[] {
  const insideGroup = inside !== undefined
  const parts = NODE_DEFS.filter((d) => (from === 'source' ? d.inputs.length > 0 : from === 'target' ? d.outputs.length > 0 : true)).map(
    (d): QuickAddItem => ({ item: `part:${d.type}`, label: d.label, group: CATEGORY_INFO[d.category].label, color: CATEGORY_INFO[d.category].color }),
  )
  const groups = GROUP_DEF_LIST.filter((g) => !insideGroup || (inside === 'transformer_block' && SUBLAYER_GROUP_TYPES.includes(g.type))).map((g): QuickAddItem => ({ item: `group:${g.type}`, label: g.label, group: 'Group', color: g.color }))
  const notes: QuickAddItem[] =
    from === null
      ? [
          { item: 'sticky', label: 'Sticky note', group: 'Annotation', color: '#fde047' },
          { item: 'textbox', label: 'Text box', group: 'Annotation', color: '#cbd5e1' },
          ...(insideGroup ? [] : [{ item: 'frame', label: 'Frame', group: 'Annotation', color: '#94a3b8' } satisfies QuickAddItem]),
        ]
      : []
  return [...parts, ...groups, ...notes]
}

/** Case-insensitive search over label / kind / id; every word must match. Label prefix matches come first. */
export function filterQuickAdd(items: QuickAddItem[], query: string): QuickAddItem[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (words.length === 0) return items
  const hits = items.filter((it) => {
    const hay = `${it.label} ${it.group} ${it.item}`.toLowerCase()
    return words.every((w) => hay.includes(w))
  })
  const first = words[0]
  return [...hits.filter((it) => it.label.toLowerCase().startsWith(first)), ...hits.filter((it) => !it.label.toLowerCase().startsWith(first))]
}

/**
 * Top-left position for a new node so that the port being connected sits at the drop point:
 * dragged from an output → the new node hangs below the point (its input on top);
 * from an input → it sits above (its output at the bottom); otherwise centred.
 */
export function quickAddPosition(item: PaletteItemId, point: XYPosition, from: DragFrom | null): XYPosition {
  const { width, height } = paletteItemSize(item)
  const y = from === 'source' ? point.y : from === 'target' ? point.y - height : point.y - height / 2
  return { x: point.x - width / 2, y }
}
