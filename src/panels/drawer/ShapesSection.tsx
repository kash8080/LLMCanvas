import { formatBytes } from '../../engine/format'
import { dtypeName } from '../../engine/hyperparams'
import { formatSymbolic, numel, shapeBytes } from '../../engine/shape'
import type { Shape } from '../../engine/types'
import { useCanvasStore } from '../../store/useCanvasStore'
import { Section, SubHeading } from './ui'

export interface PortShape {
  port: string
  shape: Shape | null
}

const UNKNOWN_IN = 'unknown — not connected, or something upstream has an error'

/**
 * Inputs and outputs: port name, symbolic + concrete shape, dtype, bytes.
 * `unknownOut` explains missing output shapes (e.g. because this part itself has an error).
 */
export function ShapesSection({ inputs, outputs, unknownOut = UNKNOWN_IN }: { inputs: PortShape[]; outputs: PortShape[]; unknownOut?: string }) {
  return (
    <Section id="shapes" title="Shapes">
      {inputs.length > 0 && (
        <>
          <SubHeading>In</SubHeading>
          <div className="space-y-1.5">
            {inputs.map((p) => (
              <ShapeRow key={p.port} {...p} unknown={UNKNOWN_IN} />
            ))}
          </div>
        </>
      )}
      {outputs.length > 0 && (
        <>
          <SubHeading>Out</SubHeading>
          <div className="space-y-1.5">
            {outputs.map((p) => (
              <ShapeRow key={p.port} {...p} unknown={unknownOut} />
            ))}
          </div>
        </>
      )}
    </Section>
  )
}

function ShapeRow({ port, shape, unknown }: PortShape & { unknown: string }) {
  const hp = useCanvasStore((s) => s.hyperparams)
  return (
    <div className="rounded-md border border-slate-100 bg-slate-50 px-2.5 py-1.5">
      <div className="flex items-baseline gap-2">
        <span className="min-w-0 truncate font-mono text-xs font-medium text-slate-700" title={port}>
          {port}
        </span>
        {shape && (
          <span className="ml-auto shrink-0 text-[11px] text-slate-400 tabular-nums" title={`${numel(shape).toLocaleString()} elements`}>
            {dtypeName(shape.dtype, hp)} · {formatBytes(shapeBytes(shape, hp))}
          </span>
        )}
      </div>
      {shape ? (
        <>
          <div className="font-mono text-[13px] break-words text-slate-800">{formatSymbolic(shape)}</div>
          <div className="font-mono text-[11px] break-words text-slate-400">{shape.dims.length === 0 ? 'one number' : shape.dims.map((d) => d.size).join(' × ')}</div>
        </>
      ) : (
        <div className="text-xs text-slate-400 italic">{unknown}</div>
      )}
    </div>
  )
}
