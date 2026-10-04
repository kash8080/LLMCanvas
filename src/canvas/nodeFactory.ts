import type { XYPosition } from '@xyflow/react'
import { defaultParams } from '../engine/resolve'
import { getNodeDef } from '../nodes/registry'
import type { AppNode, PaletteItemId, PartNode } from './types'

let counter = 0
export function newId(prefix: string): string {
  counter += 1
  return `${prefix}-${Date.now().toString(36)}-${counter.toString(36)}`
}

/** Fixed width of part nodes (keeps the default layout predictable). */
export const PART_WIDTH = 176

/** Approximate size of each palette item (used to centre a dropped node on the cursor). */
export function defaultSize(item: PaletteItemId): { width: number; height: number } {
  if (item === 'sticky') return { width: 200, height: 160 }
  if (item === 'textbox') return { width: 240, height: 60 }
  return { width: PART_WIDTH, height: 64 }
}

export function isPaletteItemId(value: string): value is PaletteItemId {
  return value === 'sticky' || value === 'textbox' || (value.startsWith('part:') && !!getNodeDef(value.slice(5)))
}

export function createPartNode(partType: string, position: XYPosition, id = newId(partType), title?: string): PartNode {
  const def = getNodeDef(partType)
  if (!def) throw new Error(`Unknown part type "${partType}"`)
  return { id, type: 'part', position, data: { partType, params: defaultParams(def), ...(title ? { title } : {}) } }
}

export function createNode(item: PaletteItemId, position: XYPosition): AppNode {
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

/** The top-most nodes (by y): fitting the view to these shows the start of a tall graph at a readable zoom. */
export function topNodes(nodes: AppNode[], count = 14): { id: string }[] {
  return [...nodes].sort((a, b) => a.position.y - b.position.y).slice(0, count).map((n) => ({ id: n.id }))
}
