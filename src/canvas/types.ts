import type { Edge, Node } from '@xyflow/react'

/** All React Flow node kinds the canvas knows about (Phase 2+ adds model parts). */
export const NODE_KINDS = ['sticky', 'textbox', 'placeholder'] as const
export type NodeKind = (typeof NODE_KINDS)[number]

export function isNodeKind(value: unknown): value is NodeKind {
  return typeof value === 'string' && (NODE_KINDS as readonly string[]).includes(value)
}

/** Data for sticky notes and text boxes. */
export type AnnotationData = {
  text: string
  bgColor: string
  textColor: string
  fontSize: number
}

/** Data for the temporary Phase-1 "Placeholder part" (replaced by real parts in Phase 2). */
export type PlaceholderData = {
  label: string
}

export type StickyNode = Node<AnnotationData, 'sticky'>
export type TextBoxNode = Node<AnnotationData, 'textbox'>
export type PlaceholderNode = Node<PlaceholderData, 'placeholder'>
export type AppNode = StickyNode | TextBoxNode | PlaceholderNode
export type AppEdge = Edge

/** MIME type used when dragging a palette item onto the canvas. */
export const DND_MIME = 'application/x-llm-canvas-node'
