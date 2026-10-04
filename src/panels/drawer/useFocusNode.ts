import { useReactFlow } from '@xyflow/react'
import { PART_WIDTH } from '../../canvas/nodeFactory'
import type { AppEdge, AppNode } from '../../canvas/types'
import { useCanvasStore } from '../../store/useCanvasStore'

/**
 * Select a node and bring it into view. Parts are centred at a zoom where group internals are
 * shown (≥ 0.9, so a part inside a group in Auto mode becomes visible); groups are fitted.
 */
export function useFocusNode(): (id: string) => void {
  const { getInternalNode, setCenter, fitBounds, getZoom } = useReactFlow<AppNode, AppEdge>()
  const selectOnly = useCanvasStore((s) => s.selectOnly)
  return (id) => {
    selectOnly(id)
    const n = getInternalNode(id)
    if (!n) return
    const { x, y } = n.internals.positionAbsolute
    const width = n.width ?? n.measured.width ?? PART_WIDTH
    const height = n.height ?? n.measured.height ?? 60
    if (n.type === 'group') void fitBounds({ x, y, width, height }, { padding: 0.1, duration: 400 })
    else void setCenter(x + width / 2, y + height / 2, { zoom: Math.max(getZoom(), 0.9), duration: 400 })
  }
}
