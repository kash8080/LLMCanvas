import type { EdgeTypes, NodeTypes } from '@xyflow/react'
import { StickyNoteNode, TextBoxNode } from './nodes/AnnotationNode'
import { PartNode } from './nodes/PartNode'
import { ShapeEdge } from './ShapeEdge'

// Defined at module level so React Flow doesn't see a new object every render.
export const nodeTypes: NodeTypes = {
  part: PartNode,
  sticky: StickyNoteNode,
  textbox: TextBoxNode,
}

/** Every edge (type unset = 'default') renders as a smooth-step edge with a shape label. */
export const edgeTypes: EdgeTypes = {
  default: ShapeEdge,
}
