import { Handle, Position } from '@xyflow/react'
import { useCanvasStore } from '../store/useCanvasStore'

/**
 * A node port. Inputs sit on top (hollow blue circle), outputs on the bottom
 * (filled green circle) so the graph reads top -> bottom (PLAN.md §2.4).
 * Dragging starts a connection; a click without a drag opens the shape popover.
 */
export function Port({
  nodeId,
  kind,
  id,
  label,
  leftPct = 50,
  size,
}: {
  nodeId: string
  kind: 'in' | 'out'
  id: string
  label: string
  leftPct?: number
  /** Diameter in px (default 12 from CSS); group ports are bigger so they can be grabbed zoomed out. */
  size?: number
}) {
  const setPortPopover = useCanvasStore((s) => s.setPortPopover)
  const isInput = kind === 'in'
  return (
    <Handle
      type={isInput ? 'target' : 'source'}
      position={isInput ? Position.Top : Position.Bottom}
      id={id}
      className={isInput ? 'port port-in' : 'port port-out'}
      title={`${isInput ? 'input' : 'output'}: ${label} — click for shape`}
      style={{ left: `${leftPct}%`, ...(size ? { width: size, height: size, borderWidth: Math.max(2, size / 6) } : {}) }}
      onClick={(e) => {
        e.stopPropagation()
        setPortPopover({ nodeId, portId: id, kind, x: e.clientX, y: e.clientY })
      }}
    />
  )
}

/** Evenly spread n ports across the node edge: 1 → 50%, 2 → 33/67%, 3 → 25/50/75%. */
export function portLeftPct(index: number, count: number): number {
  return ((index + 1) / (count + 1)) * 100
}
