import type { NodeProps } from '@xyflow/react'
import { AlertTriangle } from 'lucide-react'
import { formatCount } from '../../engine/format'
import { formatConcrete } from '../../engine/shape'
import type { NodeResult, Shape } from '../../engine/types'
import { CATEGORY_INFO, getNodeDef } from '../../nodes/registry'
import { useCanvasStore } from '../../store/useCanvasStore'
import { PART_WIDTH } from '../nodeFactory'
import { Port, portLeftPct } from '../Port'
import type { PartNode as PartNodeType } from '../types'

/** One generic component for every model part in the registry. */
export function PartNode({ id, data, selected }: NodeProps<PartNodeType>) {
  const def = getNodeDef(data.partType)
  const result = useCanvasStore((s) => s.inference.nodes[id]) as NodeResult | undefined
  if (!def) return <div className="rounded border border-red-400 bg-white p-2 text-xs text-red-600">Unknown part “{data.partType}”</div>

  const color = CATEGORY_INFO[def.category].color
  const status = result?.status ?? 'unknown'
  const title = data.title || def.label
  const params = result?.paramCount.total ?? 0

  const frame =
    status === 'error'
      ? 'border-red-500 ring-2 ring-red-200'
      : status === 'unknown'
        ? 'border-dashed border-slate-300 bg-slate-50'
        : selected
          ? 'border-indigo-500 ring-2 ring-indigo-200'
          : 'border-slate-300 hover:shadow-md'

  return (
    <div
      className={`group relative rounded-lg border bg-white shadow-sm transition-shadow ${frame} ${selected && status !== 'ok' ? 'shadow-lg' : ''}`}
      style={{ width: PART_WIDTH }}
    >
      <div className="absolute inset-x-0 top-0 h-1.5 rounded-t-lg" style={{ background: color, opacity: status === 'unknown' ? 0.4 : 1 }} />

      {def.inputs.map((p, i) => (
        <Port key={p.id} nodeId={id} kind="in" id={p.id} label={p.label} leftPct={portLeftPct(i, def.inputs.length)} />
      ))}
      {def.inputs.length > 1 && <PortLabels labels={def.inputs.map((p) => p.label)} className="top-2" />}

      <div className={`px-2.5 pb-1.5 ${def.inputs.length > 1 ? 'pt-5' : 'pt-2.5'} ${status === 'unknown' ? 'opacity-60' : ''}`}>
        <div className="flex items-center gap-1.5">
          <span className="truncate text-[13px] font-semibold text-slate-800" title={title}>
            {title}
          </span>
          {status === 'error' && <AlertTriangle size={13} className="shrink-0 text-red-500" />}
          {params > 0 && (
            <span className="ml-auto shrink-0 rounded bg-slate-100 px-1 text-[10px] font-medium text-slate-500 tabular-nums" title={`${params.toLocaleString()} parameters`}>
              {formatCount(params)}
            </span>
          )}
        </div>
        {data.title && <div className="truncate text-[10px] leading-tight text-slate-400">{def.label}</div>}
        <ShapeLine status={status} result={result} />
      </div>

      {def.outputs.length > 1 && <PortLabels labels={def.outputs.map((p) => p.label)} className="bottom-1.5" />}
      {def.outputs.length > 1 && <div className="h-3" />}
      {def.outputs.map((p, i) => (
        <Port key={p.id} nodeId={id} kind="out" id={p.id} label={p.label} leftPct={portLeftPct(i, def.outputs.length)} />
      ))}

      {status === 'error' && result && (
        <div className="pointer-events-none absolute top-0 left-full z-10 ml-2 hidden w-64 rounded-md border border-red-200 bg-white p-2 text-[11px] leading-snug text-red-700 shadow-lg group-hover:block">
          {result.errors.map((e) => (
            <div key={e}>⚠ {e}</div>
          ))}
        </div>
      )}
    </div>
  )
}

function ShapeLine({ status, result }: { status: string; result?: NodeResult }) {
  if (status === 'error') return <div className="mt-0.5 truncate text-[11px] text-red-600">⚠ {result?.errors[0]}</div>
  if (status === 'unknown' || !result) return <div className="mt-0.5 text-[11px] text-slate-400 italic">shape unknown</div>
  // Show the output shape(s); a sink (Loss) shows its input.
  const shapes = (result.outputShapes.length > 0 ? result.outputShapes : result.inputShapes).filter((s): s is Shape => !!s)
  const text = [...new Set(shapes.map(formatConcrete))].join(' / ')
  return <div className="mt-0.5 truncate font-mono text-[11px] text-slate-500">{text || '—'}</div>
}

function PortLabels({ labels, className }: { labels: string[]; className: string }) {
  return (
    <>
      {labels.map((l, i) => (
        <span
          key={l}
          className={`pointer-events-none absolute -translate-x-1/2 text-[9px] leading-none text-slate-400 ${className}`}
          style={{ left: `${portLeftPct(i, labels.length)}%` }}
        >
          {l}
        </span>
      ))}
    </>
  )
}
