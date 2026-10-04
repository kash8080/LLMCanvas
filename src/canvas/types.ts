import type { Edge, Node } from '@xyflow/react'
import type { ParamValue } from '../engine/types'

/** All React Flow node kinds the canvas knows about. Every model part uses the generic 'part' kind. */
export const NODE_KINDS = ['part', 'sticky', 'textbox'] as const
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

/** Data for a model part: which NodeDef (src/nodes/registry.ts), its params, optional display title. */
export type PartData = {
  partType: string
  params: Record<string, ParamValue>
  /** e.g. "q_proj", "ln1"; falls back to the def label. */
  title?: string
}

export type PartNode = Node<PartData, 'part'>
export type StickyNode = Node<AnnotationData, 'sticky'>
export type TextBoxNode = Node<AnnotationData, 'textbox'>
export type AppNode = PartNode | StickyNode | TextBoxNode
export type AppEdge = Edge

/** MIME type used when dragging a palette item onto the canvas. */
export const DND_MIME = 'application/x-llm-canvas-node'

/** Palette drag payload: an annotation kind or `part:<partType>`. */
export type PaletteItemId = 'sticky' | 'textbox' | `part:${string}`
