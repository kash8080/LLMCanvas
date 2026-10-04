import { useReactFlow } from '@xyflow/react'
import { isProxyType } from '../engine/groups'
import type { AppNode } from './types'

/**
 * Delete nodes / edges the same way the Delete key does (React Flow `deleteElements`): a group takes its
 * children along, a part takes its edges, and `onBeforeDelete` (Canvas) keeps lone group proxies and
 * shows a hint. Undo picks it up from the resulting change events (store/useCanvasStore.ts).
 */
export function useDeleteElements(): (nodeIds: string[], edgeIds?: string[]) => void {
  const { deleteElements } = useReactFlow()
  return (nodeIds, edgeIds = []) => void deleteElements({ nodes: nodeIds.map((id) => ({ id })), edges: edgeIds.map((id) => ({ id })) })
}

/** A group's in/out pill can't be deleted on its own (only with its group). */
export function isLoneProxy(node: AppNode): boolean {
  return node.type === 'part' && isProxyType(node.data.partType)
}

/** "Delete", "Delete group (12 parts)". */
export function deleteLabel(node: AppNode, nodes: AppNode[]): string {
  if (node.type !== 'group') return 'Delete'
  const inside = new Set([node.id])
  for (const n of nodes) if (n.parentId && inside.has(n.parentId)) inside.add(n.id)
  const parts = nodes.filter((n) => inside.has(n.id) && n.type === 'part' && !isProxyType(n.data.partType)).length
  return `Delete group (${parts} parts)`
}
