// User-made frames (PLAN.md §2.4, Phase 7b): Miro-style titled rectangles that label a region.
//
// A frame is NOT a React Flow parent. It never gets children (`parentId`), so connection rules,
// `flattenGroups`, inference, persistence order and deletion are untouched. Instead, "moving a frame
// moves what's inside it" is done here: when a frame is dragged (or nudged with the arrow keys), the
// top-level items whose box lies fully inside the frame — captured when the drag starts — get the
// same position delta in the same change batch (so the whole drag is still one undo step).
// Pure functions, no React.
import type { NodeChange } from '@xyflow/react'
import { defaultSize, PART_WIDTH } from './nodeFactory'
import type { AppNode } from './types'

export interface Box {
  x: number
  y: number
  width: number
  height: number
}

/** Space between the framed items and the frame's edge ("Frame selection"). */
export const FRAME_PADDING = 40
/** Frames sit below everything else (also when selected: React Flow adds 1000 to a selected node's z). */
export const FRAME_Z = -1_000_000
/** z step between frames: more than React Flow's selection boost (1000), so a selected outer frame stays below an inner one. */
export const FRAME_Z_STEP = 1001
/** Title size on screen at zoom ≥ 1 (px). */
export const FRAME_TITLE_PX = 13

const FALLBACK_HEIGHT: Partial<Record<AppNode['type'], number>> = { part: 64 }

/** A node's box in flow coordinates: declared size, else measured size, else the default size of its kind. */
export function nodeBox(n: AppNode): Box {
  const fallback = n.type === 'sticky' || n.type === 'textbox' || n.type === 'frame' ? defaultSize(n.type) : { width: PART_WIDTH, height: FALLBACK_HEIGHT[n.type] ?? 64 }
  return {
    x: n.position.x,
    y: n.position.y,
    width: n.width ?? n.measured?.width ?? fallback.width,
    height: n.height ?? n.measured?.height ?? fallback.height,
  }
}

/** Is `inner` fully inside `outer` (touching the edge counts as inside)? */
export function isInside(inner: Box, outer: Box): boolean {
  return inner.x >= outer.x && inner.y >= outer.y && inner.x + inner.width <= outer.x + outer.width && inner.y + inner.height <= outer.y + outer.height
}

/**
 * Ids of the top-level items fully inside frame `frameId` (other frames included, so nested frames
 * move along). Children of groups are left out: they move with their group.
 */
export function itemsInFrame(nodes: AppNode[], frameId: string): string[] {
  const frame = nodes.find((n) => n.id === frameId)
  if (frame?.type !== 'frame') return []
  const box = nodeBox(frame)
  return nodes.filter((n) => n.id !== frameId && !n.parentId && isInside(nodeBox(n), box)).map((n) => n.id)
}

/** Map ids to their top-level ancestors (a part inside Block 1 → Block 1), without duplicates, in node order. */
export function topLevelIds(nodes: AppNode[], ids: string[]): string[] {
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const top = new Set<string>()
  for (const id of ids) {
    let n = byId.get(id)
    while (n?.parentId) n = byId.get(n.parentId)
    if (n) top.add(n.id)
  }
  return nodes.filter((n) => top.has(n.id)).map((n) => n.id)
}

/** The frame box around these (top-level) items: their bounding box plus `padding` on every side. Null if none. */
export function frameBoxAround(nodes: AppNode[], ids: string[], padding = FRAME_PADDING): Box | null {
  const want = new Set(ids)
  const boxes = nodes.filter((n) => want.has(n.id)).map(nodeBox)
  if (boxes.length === 0) return null
  const x = Math.min(...boxes.map((b) => b.x)) - padding
  const y = Math.min(...boxes.map((b) => b.y)) - padding
  const right = Math.max(...boxes.map((b) => b.x + b.width)) + padding
  const bottom = Math.max(...boxes.map((b) => b.y + b.height)) + padding
  return { x, y, width: right - x, height: bottom - y }
}

/**
 * Font size (flow px) of a frame's title: constant on screen when zoomed out (so frames label regions
 * in an overview), but never wider than the frame itself (≈ 0.6 em per character).
 */
export function frameTitleFontSize(zoom: number, frameWidth: number, title: string): number {
  const onScreen = FRAME_TITLE_PX / Math.max(zoom, 0.01)
  const fitWidth = frameWidth / (Math.max(title.length, 4) * 0.6 + 2)
  return Math.max(FRAME_TITLE_PX, Math.min(onScreen, fitWidth))
}

/**
 * Extra position changes that make a frame's contents follow it. Call with every `onNodesChange` batch.
 *
 * - Only position changes that carry React Flow's `dragging` flag count: a drag (`true` while moving,
 *   `false` at the end) or an arrow-key nudge (`false`). NodeResizer's position changes (dragging the
 *   left / top edge) have no flag, so resizing never moves the contents.
 * - The contents are captured when a drag starts (`captured`, mutated: frame id → item ids) so items
 *   the frame passes over on the way aren't picked up; the capture is dropped when the drag ends.
 * - Items that move in this batch anyway (selected together with the frame) aren't moved twice.
 */
export function frameFollowChanges(changes: NodeChange<AppNode>[], nodes: AppNode[], captured: Map<string, string[]>): NodeChange<AppNode>[] {
  let byId: Map<string, AppNode> | null = null
  const moving = new Set<string>()
  for (const c of changes) if (c.type === 'position') moving.add(c.id)
  const extra: NodeChange<AppNode>[] = []
  for (const c of changes) {
    if (c.type !== 'position' || c.dragging === undefined || !c.position) continue
    byId ??= new Map(nodes.map((n) => [n.id, n]))
    const frame = byId.get(c.id)
    if (frame?.type !== 'frame') continue
    let ids = captured.get(frame.id)
    if (!ids) {
      ids = itemsInFrame(nodes, frame.id)
      if (c.dragging) captured.set(frame.id, ids)
    }
    if (!c.dragging) captured.delete(frame.id)
    const dx = c.position.x - frame.position.x
    const dy = c.position.y - frame.position.y
    if (dx === 0 && dy === 0) continue
    for (const id of ids) {
      const n = byId.get(id)
      if (!n || moving.has(id)) continue
      moving.add(id)
      extra.push({ type: 'position', id, position: { x: n.position.x + dx, y: n.position.y + dy } })
    }
  }
  return extra
}

/**
 * Display-only z-order for frames (Canvas): every frame sits below all other nodes and edges, and a
 * bigger frame below a smaller one (even while selected), so a frame inside another stays clickable. Unchanged nodes keep
 * their object identity.
 */
export function withFrameLayers(nodes: AppNode[]): AppNode[] {
  const frames = nodes.filter((n) => n.type === 'frame')
  if (frames.length === 0) return nodes
  const area = (n: AppNode) => {
    const b = nodeBox(n)
    return b.width * b.height
  }
  const z = new Map([...frames].sort((a, b) => area(b) - area(a)).map((f, i) => [f.id, FRAME_Z + i * FRAME_Z_STEP]))
  return nodes.map((n) => (z.has(n.id) && n.zIndex !== z.get(n.id) ? { ...n, zIndex: z.get(n.id) } : n))
}
