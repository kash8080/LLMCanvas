import type { XYPosition } from '@xyflow/react'
import { defaultParams } from '../engine/resolve'
import { isGroupType } from '../nodes/groups'
import { getNodeDef } from '../nodes/registry'
import type { AppNode, PaletteItemId, PartNode } from './types'

let counter = 0
export function newId(prefix: string): string {
  counter += 1
  return `${prefix}-${Date.now().toString(36)}-${counter.toString(36)}`
}

/** Fixed width of part nodes (keeps the default layout predictable). */
export const PART_WIDTH = 176
/** Size of the small in/out proxy pills inside groups. */
export const PROXY_WIDTH = 120
export const PROXY_HEIGHT = 24

/** Approximate size of each palette item (used to centre a dropped node on the cursor). */
export function defaultSize(item: PaletteItemId): { width: number; height: number } {
  if (item === 'sticky') return { width: 200, height: 160 }
  if (item === 'textbox') return { width: 240, height: 60 }
  return { width: PART_WIDTH, height: 64 }
}

export function isPaletteItemId(value: string): value is PaletteItemId {
  return (
    value === 'sticky' ||
    value === 'textbox' ||
    (value.startsWith('part:') && !!getNodeDef(value.slice(5))) ||
    (value.startsWith('group:') && isGroupType(value.slice(6)))
  )
}

export function createPartNode(partType: string, position: XYPosition, id = newId(partType), title?: string): PartNode {
  const def = getNodeDef(partType)
  if (!def) throw new Error(`Unknown part type "${partType}"`)
  return { id, type: 'part', position, data: { partType, params: defaultParams(def), ...(title ? { title } : {}) } }
}

/** A single node for an annotation or part palette item (groups: see instantiateGroup). */
export function createNode(item: Exclude<PaletteItemId, `group:${string}`>, position: XYPosition): AppNode {
  if (item === 'sticky') {
    return {
      id: newId('sticky'),
      type: 'sticky',
      position,
      ...defaultSize('sticky'),
      data: { text: '', bgColor: '#fef08a', textColor: '#1f2937', fontSize: 14 },
    }
  }
  if (item === 'textbox') {
    return {
      id: newId('textbox'),
      type: 'textbox',
      position,
      ...defaultSize('textbox'),
      data: { text: '', bgColor: 'transparent', textColor: '#111827', fontSize: 18 },
    }
  }
  return createPartNode(item.slice(5), position)
}

/**
 * The top-most top-level nodes (by y): fitting the view to these shows the start of a tall graph
 * at a readable zoom. (Children of groups have positions relative to their frame, so they're skipped.)
 */
/** Bounding box of `topNodes` from their declared position / size (no measuring needed). */
export function startBounds(nodes: AppNode[]): { x: number; y: number; width: number; height: number } | null {
  const ids = new Set(topNodes(nodes).map((n) => n.id))
  const boxes = nodes
    .filter((n) => ids.has(n.id))
    .map((n) => ({ x: n.position.x, y: n.position.y, w: n.width ?? n.measured?.width ?? PART_WIDTH, h: n.height ?? n.measured?.height ?? 64 }))
  if (boxes.length === 0) return null
  const x = Math.min(...boxes.map((b) => b.x))
  const y = Math.min(...boxes.map((b) => b.y))
  return { x, y, width: Math.max(...boxes.map((b) => b.x + b.w)) - x, height: Math.max(...boxes.map((b) => b.y + b.h)) - y }
}

export function topNodes(nodes: AppNode[], count = 4): { id: string }[] {
  return nodes
    .filter((n) => !n.parentId)
    .sort((a, b) => a.position.y - b.position.y)
    .slice(0, count)
    .map((n) => ({ id: n.id }))
}
