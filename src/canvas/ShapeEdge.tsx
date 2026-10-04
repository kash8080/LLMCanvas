import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath, useStore, type EdgeProps } from '@xyflow/react'
import { X } from 'lucide-react'
import { formatConcrete } from '../engine/shape'
import { useCanvasStore } from '../store/useCanvasStore'
import { useDeleteElements } from './useDelete'

/**
 * Smooth-step edge with an optional concrete-shape label (toolbar → "Shapes on edges").
 * Selected: highlighted, with a small × under the label that deletes it.
 */
export function ShapeEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, style, selected }: EdgeProps) {
  const shape = useCanvasStore((s) => s.inference.edges[id])
  const showLabel = useCanvasStore((s) => s.showEdgeShapes)
  const [path, labelX, labelY] = getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, borderRadius: 10 })
  const label = showLabel && shape

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        style={{
          ...style,
          ...(shape ? {} : { strokeDasharray: '5 4', stroke: '#cbd5e1' }),
          ...(selected ? { stroke: '#6366f1', strokeWidth: 2.5 } : {}),
        }}
      />
      {(label || selected) && (
        <EdgeLabelRenderer>
          {label && (
            <div
              className={`pointer-events-none absolute rounded border bg-white/95 px-1 font-mono text-[10px] leading-4 ${
                selected ? 'border-indigo-300 text-indigo-700' : 'border-slate-200 text-slate-500'
              }`}
              style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
            >
              {formatConcrete(shape)}
            </div>
          )}
          {selected && <DeleteEdgeButton id={id} x={labelX} y={labelY} belowLabel={!!label} />}
        </EdgeLabelRenderer>
      )}
    </>
  )
}

/** The × on a selected edge. Counter-scaled so it stays ~20 px on screen when zoomed out. */
function DeleteEdgeButton({ id, x, y, belowLabel }: { id: string; x: number; y: number; belowLabel: boolean }) {
  const scale = useStore((s) => Math.max(1, 1 / s.transform[2]))
  const deleteElements = useDeleteElements()
  return (
    <button
      type="button"
      title="Delete connection (⌫)"
      onClick={() => deleteElements([], [id])}
      className="nodrag nopan pointer-events-auto absolute flex h-5 w-5 items-center justify-center rounded-full border border-red-200 bg-white text-red-500 shadow-sm hover:bg-red-50"
      style={{ transform: `translate(-50%, -50%) translate(${x}px, ${belowLabel ? y + 9 + 11 * scale : y}px) scale(${scale})` }}
    >
      <X size={12} />
    </button>
  )
}
