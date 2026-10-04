import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath, type EdgeProps } from '@xyflow/react'
import { formatConcrete } from '../engine/shape'
import { useCanvasStore } from '../store/useCanvasStore'

/** Smooth-step edge with an optional concrete-shape label (toolbar → "Shapes on edges"). */
export function ShapeEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, style, selected }: EdgeProps) {
  const shape = useCanvasStore((s) => s.inference.edges[id])
  const showLabel = useCanvasStore((s) => s.showEdgeShapes)
  const [path, labelX, labelY] = getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, borderRadius: 10 })

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
      {showLabel && shape && (
        <EdgeLabelRenderer>
          <div
            className="pointer-events-none absolute rounded border border-slate-200 bg-white/95 px-1 font-mono text-[10px] leading-4 text-slate-500"
            style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          >
            {formatConcrete(shape)}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}
