import type { Edge, Node } from '@xyflow/react'
import type { ParamValue } from '../engine/types'
import type { GroupType } from '../nodes/groups'

/**
 * All React Flow node kinds the canvas knows about. Every model part uses the generic 'part' kind;
 * Transformer Block / MHA / SwiGLU frames use 'group' (children point at it via `parentId`).
 */
export const NODE_KINDS = ['part', 'group', 'sticky', 'textbox'] as const
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

/** Manual level-of-detail override for a group: follow the zoom, or stay open / closed. */
export type GroupMode = 'auto' | 'expanded' | 'collapsed'
export const GROUP_MODES: GroupMode[] = ['auto', 'expanded', 'collapsed']

/** Data for a group frame (src/nodes/groups.ts). Its size is fixed (node width/height). */
export type GroupData = {
  groupType: GroupType
  /** e.g. "Block 1", "attn"; falls back to the group label. */
  title?: string
  mode: GroupMode
}

export type PartNode = Node<PartData, 'part'>
export type GroupNode = Node<GroupData, 'group'>
export type StickyNode = Node<AnnotationData, 'sticky'>
export type TextBoxNode = Node<AnnotationData, 'textbox'>
export type AppNode = PartNode | GroupNode | StickyNode | TextBoxNode
export type AppEdge = Edge

/** MIME type used when dragging a palette item onto the canvas. */
export const DND_MIME = 'application/x-llm-canvas-node'

/** Palette drag payload: an annotation kind, `part:<partType>` or `group:<groupType>`. */
export type PaletteItemId = 'sticky' | 'textbox' | `part:${string}` | `group:${GroupType}`
