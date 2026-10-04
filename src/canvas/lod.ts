// Semantic zoom (PLAN.md §2.3, R5.2): which groups show their insides at the current zoom.
//
//   level 0  zoom <  LOD_ZOOM.blocks     → Transformer Blocks are single cards
//   level 1  zoom <  LOD_ZOOM.internals  → block internals visible, MHA / SwiGLU are cards
//   level 2  otherwise                   → everything expanded
//
// Components read the zoom through `useStore(lodSelector)`, which returns the bucketed level, so
// they only re-render when the level changes, not on every zoom step.
import type { ReactFlowState } from '@xyflow/react'
import { GROUP_DEFS } from '../nodes/groups'
import type { AppEdge, AppNode, GroupData } from './types'

export const LOD_ZOOM = { blocks: 0.25, internals: 0.6 }

export type Lod = 0 | 1 | 2

export function lodForZoom(zoom: number): Lod {
  return zoom < LOD_ZOOM.blocks ? 0 : zoom < LOD_ZOOM.internals ? 1 : 2
}

export const lodSelector = (s: ReactFlowState): Lod => lodForZoom(s.transform[2])

/** Does this group show its children (manual override first, then the zoom level)? */
export function isGroupExpanded(data: GroupData, lod: Lod): boolean {
  if (data.mode === 'expanded') return true
  if (data.mode === 'collapsed') return false
  return lod >= GROUP_DEFS[data.groupType].expandAtLod
}

/**
 * Hide every node inside a collapsed group (including nested ones). Parents come before children
 * in the nodes array, so one pass is enough. Unchanged nodes keep their object identity.
 */
export function applyLod(nodes: AppNode[], lod: Lod): AppNode[] {
  const collapsed = new Set(nodes.filter((n) => n.type === 'group' && !isGroupExpanded(n.data, lod)).map((n) => n.id))
  const hidden = new Set<string>()
  return nodes.map((n) => {
    const hide = !!n.parentId && (collapsed.has(n.parentId) || hidden.has(n.parentId))
    if (hide) hidden.add(n.id)
    return !!n.hidden === hide ? n : { ...n, hidden: hide }
  })
}

/** Hide edges that touch a hidden node (React Flow would otherwise keep drawing them). */
export function hideEdgesOfHiddenNodes(edges: AppEdge[], nodes: AppNode[]): AppEdge[] {
  const hidden = new Set(nodes.filter((n) => n.hidden).map((n) => n.id))
  return edges.map((e) => {
    const hide = hidden.has(e.source) || hidden.has(e.target)
    return !!e.hidden === hide ? e : { ...e, hidden: hide }
  })
}
