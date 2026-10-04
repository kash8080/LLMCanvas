import type { NodeTypes } from '@xyflow/react'
import { StickyNoteNode, TextBoxNode } from './nodes/AnnotationNode'
import { PlaceholderNode } from './nodes/PlaceholderNode'

// Defined at module level so React Flow doesn't see a new object every render.
export const nodeTypes: NodeTypes = {
  sticky: StickyNoteNode,
  textbox: TextBoxNode,
  placeholder: PlaceholderNode,
}
