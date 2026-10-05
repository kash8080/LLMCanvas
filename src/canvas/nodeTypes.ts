import type { EdgeTypes, NodeTypes } from '@xyflow/react'
import { StickyNoteNode, TextBoxNode } from './nodes/AnnotationNode'
import { FrameNode } from './nodes/FrameNode'
import { GroupNode } from './nodes/GroupNode'
import { PartNode } from './nodes/PartNode'
import { ShapeEdge } from './ShapeEdge'

// Defined at module level so React Flow doesn't see a new object every render.
export const nodeTypes: NodeTypes = {
  part: PartNode,
  group: GroupNode,
  sticky: StickyNoteNode,
  textbox: TextBoxNode,
  frame: FrameNode,
}

/** Every edge (type unset = 'default') renders as a smooth-step edge with a shape label. */
export const edgeTypes: EdgeTypes = {
  default: ShapeEdge,
}
