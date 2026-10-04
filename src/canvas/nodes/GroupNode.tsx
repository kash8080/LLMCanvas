import { useStore, type NodeProps } from '@xyflow/react'
import { AlertTriangle } from 'lucide-react'
import { formatCount } from '../../engine/format'
import { formatConcrete, formatSymbolic } from '../../engine/shape'
import type { Shape } from '../../engine/types'
import { GROUP_DEFS } from '../../nodes/groups'
import { highlightInfo } from '../../nodes/registry'
import { useCanvasStore } from '../../store/useCanvasStore'
import { glow, groupHeat, groupHighlight, heatColor } from '../highlight'
import { GROUP_HEADER, GROUP_LAYOUT } from '../groupTemplates'
import { isGroupExpanded, lodSelector } from '../lod'
import { Port } from '../Port'
import { GROUP_MODES, type GroupMode, type GroupNode as GroupNodeType } from '../types'

/**
 * A group frame (Transformer Block / MHA / SwiGLU). Same size whether expanded or collapsed:
 * expanded = a tinted frame with a header, children drawn on top; collapsed = a summary card
 * (children are hidden by `applyLod` in Canvas).
 */
export function GroupNode({ id, data, selected, width }: NodeProps<GroupNodeType>) {
  const def = GROUP_DEFS[data.groupType]
  const lod = useStore(lodSelector)
  const summary = useCanvasStore((s) => s.inference.groups[id])
  const highlight = useCanvasStore((s) => groupHighlight(s, id))
  const highlightKey = useCanvasStore((s) => s.highlight)
  const heat = useCanvasStore((s) => groupHeat(s, id))
  // Weights inside, none of them feeding Logits / Loss (e.g. a block not wired in yet).
  const uncounted = useCanvasStore((s) => {
    const g = s.inference.params.groups[id]
    return !!g && g.connected === 0 && g.unconnected > 0
  })
  const expanded = isGroupExpanded(data, lod)
  const portLeft = GROUP_LAYOUT[data.groupType].portX * 100
  const title = data.title || def.label
  const status = summary?.status ?? 'unknown'
  const errorTip = summary?.errors.map((e) => `⚠ ${e}`).join('\n')
  // Outer ports grow when zoomed out, so a collapsed card can still be wired up.
  const portSize = PORT_SIZE[lod]
  const params = summary?.params ?? 0
  const paramsTip = `${params.toLocaleString()} parameters${uncounted ? ' — not counted in the model total: this group doesn’t feed Logits / Loss' : ''}`

  return (
    <div className="relative h-full w-full">
      {expanded ? (
        <div
          className={`h-full w-full rounded-xl border-2 transition-opacity ${selected ? 'ring-4 ring-indigo-200' : ''} ${status === 'error' ? 'border-red-400' : ''} ${
            highlight === 'dim' ? 'opacity-30' : ''
          }`}
          style={{ borderColor: status === 'error' ? undefined : `${def.color}80`, background: `${def.color}0a` }}
        >
          <div className="flex items-center gap-2 rounded-t-[10px] px-3" style={{ height: GROUP_HEADER, background: `${def.color}1c` }}>
            <span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: def.color }} />
            <span className="truncate text-sm font-semibold text-slate-800">{title}</span>
            {data.title && <span className="truncate text-xs text-slate-500">{def.label}</span>}
            {status === 'error' && (
              <span title={errorTip} className="flex shrink-0 items-center gap-1 text-xs text-red-600">
                <AlertTriangle size={13} /> {summary!.errors.length}
              </span>
            )}
            <span
              className={`ml-auto shrink-0 rounded px-1.5 text-[11px] font-medium tabular-nums ${uncounted ? 'bg-amber-100 text-amber-700' : 'bg-white/70 text-slate-600'}`}
              title={paramsTip}
            >
              {formatCount(params)}
            </span>
            <ModeToggle id={id} mode={data.mode} className="text-[11px]" />
          </div>
        </div>
      ) : (
        <div
          className={`relative flex h-full w-full flex-col items-center justify-center gap-[0.35em] overflow-hidden rounded-xl border-2 bg-white px-[0.8em] text-center shadow-sm ${
            selected ? 'ring-4 ring-indigo-200' : ''
          } ${status === 'error' ? 'border-red-400' : status === 'unknown' ? 'border-dashed' : ''} ${highlight === 'dim' ? 'opacity-25' : ''}`}
          // Text scales with the frame: a collapsed block is read zoomed far out.
          style={{
            fontSize: (width ?? 400) / 16,
            borderColor: status === 'error' ? undefined : def.color,
            boxShadow: highlight === 'match' && highlightKey ? glow(highlightInfo(highlightKey).color, '0.5em', '2em') : undefined,
            background: heat !== null ? heatColor(heat) : undefined,
          }}
          title={errorTip}
        >
          <div className="absolute inset-x-0 top-0 h-[0.35em]" style={{ background: def.color }} />
          <ModeToggle id={id} mode={data.mode} className="absolute top-[0.8em] right-[0.6em] text-[0.45em]" />
          <div className="max-w-full truncate leading-tight font-bold text-slate-800" style={{ fontSize: '2em' }}>
            {title}
          </div>
          {data.title && <div className="text-slate-500">{def.label}</div>}
          <ShapeSummary inShape={summary?.inShape ?? null} outShape={summary?.outShape ?? null} />
          <div className={`font-semibold tabular-nums ${uncounted ? 'text-amber-600' : 'text-slate-700'}`} style={{ fontSize: '1.2em' }} title={paramsTip}>
            {formatCount(params)} <span className="font-normal text-slate-400">params{uncounted ? ' (not counted)' : ''}</span>
          </div>
          {summary && summary.contains.length > 0 && (
            <div className="max-w-[90%] leading-snug text-slate-400" style={{ fontSize: '0.6em' }}>
              contains: {summary.contains.join(' · ')}
            </div>
          )}
          {status === 'error' && (
            <div className="flex items-center gap-[0.3em] text-red-600" style={{ fontSize: '0.8em' }}>
              <AlertTriangle className="h-[1em] w-[1em]" />
              {summary!.errors.length} problem{summary!.errors.length > 1 ? 's' : ''} inside — {summary!.errors[0]}
            </div>
          )}
          <div className="text-slate-300" style={{ fontSize: '0.55em' }}>
            {data.mode === 'collapsed' ? 'Collapsed — switch to Auto or Open to see inside' : 'Zoom in to see inside'}
          </div>
        </div>
      )}
      {/* Ports last so they paint above the (positioned) card. */}
      <Port nodeId={id} kind="in" id="in" label="x" leftPct={portLeft} size={portSize} />
      <Port nodeId={id} kind="out" id="out" label="out" leftPct={portLeft} size={portSize} />
    </div>
  )
}

const PORT_SIZE = [56, 26, 16] as const

function ShapeSummary({ inShape, outShape }: { inShape: Shape | null; outShape: Shape | null }) {
  if (!inShape && !outShape) return <div className="text-slate-400 italic" style={{ fontSize: '0.8em' }}>shape unknown</div>
  const sym = (s: Shape | null) => (s ? formatSymbolic(s) : '?')
  const con = (s: Shape | null) => (s ? formatConcrete(s) : '?')
  return (
    <div className="font-mono leading-snug whitespace-nowrap" style={{ fontSize: '0.6em' }}>
      <div className="text-slate-600">
        {sym(inShape)} → {sym(outShape)}
      </div>
      <div className="text-slate-400">
        {con(inShape)} → {con(outShape)}
      </div>
    </div>
  )
}

const MODE_LABEL: Record<GroupMode, { label: string; help: string }> = {
  auto: { label: 'Auto', help: 'Open or close with the zoom level' },
  expanded: { label: 'Open', help: 'Always show the parts inside' },
  collapsed: { label: 'Closed', help: 'Always show as a single card' },
}

/** auto / always expanded / always collapsed. Sized in em so it scales with the surrounding text. */
export function ModeToggle({ id, mode, className = '' }: { id: string; mode: GroupMode; className?: string }) {
  const setGroupMode = useCanvasStore((s) => s.setGroupMode)
  return (
    <div className={`nodrag flex shrink-0 overflow-hidden rounded-[0.5em] border border-slate-300 bg-white ${className}`}>
      {GROUP_MODES.map((m) => (
        <button
          key={m}
          type="button"
          title={MODE_LABEL[m].help}
          onClick={(e) => {
            e.stopPropagation()
            setGroupMode(id, m)
          }}
          className={`px-[0.6em] py-[0.15em] leading-normal font-medium transition ${
            mode === m ? 'bg-indigo-600 text-white' : 'text-slate-500 hover:bg-slate-100'
          }`}
        >
          {MODE_LABEL[m].label}
        </button>
      ))}
    </div>
  )
}
