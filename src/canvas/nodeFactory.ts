import type { XYPosition } from '@xyflow/react'
import type { AppNode, NodeKind } from './types'

let counter = 0
export function newId(prefix: string): string {
  counter += 1
  return `${prefix}-${Date.now().toString(36)}-${counter.toString(36)}`
}

/** Default size of each node kind (used to centre a dropped node on the cursor). */
export const DEFAULT_SIZE: Record<NodeKind, { width: number; height: number }> = {
  sticky: { width: 200, height: 160 },
  textbox: { width: 240, height: 60 },
  placeholder: { width: 180, height: 64 },
}

export function createNode(kind: NodeKind, position: XYPosition): AppNode {
  const id = newId(kind)
  switch (kind) {
    case 'sticky':
      return {
        id,
        type: 'sticky',
        position,
        ...DEFAULT_SIZE.sticky,
        data: { text: '', bgColor: '#fef08a', textColor: '#1f2937', fontSize: 14 },
      }
    case 'textbox':
      return {
        id,
        type: 'textbox',
        position,
        ...DEFAULT_SIZE.textbox,
        data: { text: '', bgColor: 'transparent', textColor: '#111827', fontSize: 18 },
      }
    case 'placeholder':
      return { id, type: 'placeholder', position, data: { label: 'Placeholder part' } }
  }
}
