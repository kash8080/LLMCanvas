// "Size" section: parameter count (+ formula, weight tensors / per-child breakdown).
// Phase 6 adds the memory contribution as a sub-section: see <MemoryContribution> below.
import { formatCount } from '../../engine/format'
import type { ParamCountResult } from '../../engine/types'
import type { GroupSummary } from '../../store/inference'
import { useCanvasStore } from '../../store/useCanvasStore'
import { PARAM_CATEGORY_INFO } from '../../nodes/registry'
import { Formula } from './DocsSections'
import { Section, SubHeading } from './ui'
import { useFocusNode } from './useFocusNode'

/** Size of one part: its weight tensors and how the count is computed. */
export function PartSizeSection({ nodeId, title, count }: { nodeId: string; title?: string; count: ParamCountResult }) {
  const { total, tensors } = count
  const symbolic = tensors.map((t) => t.dims.map((d) => d.label ?? String(d.size)).join(' · ')).join(' + ')
  const concrete = tensors.map((t) => t.dims.map((d) => String(d.size)).join(' · ')).join(' + ')

  return (
    <Section id="size" title="Size" meta={`${formatCount(total)} params`}>
      <SubHeading>Parameters</SubHeading>
      {total === 0 ? (
        <p className="text-xs text-slate-500">No learned parameters.</p>
      ) : (
        <>
          <TotalLine total={total} />
          <ModelShare nodeId={nodeId} />
          <Formula lines={[`params = ${symbolic}`, `${symbolic === concrete ? '' : `= ${concrete} `}= ${total.toLocaleString()}`]} />
          <div className="mt-2 space-y-1">
            {tensors.map((t) => {
              const n = t.dims.reduce((a, d) => a * d.size, 1)
              return (
                <div key={t.name} className="rounded-md border border-slate-100 bg-slate-50 px-2.5 py-1.5 font-mono text-xs">
                  <div className="flex items-baseline gap-2">
                    <span className="min-w-0 break-all text-slate-700">
                      {title ? `${title}.` : ''}
                      {t.name}
                    </span>
                    <span className="ml-auto shrink-0 text-slate-500 tabular-nums">{n.toLocaleString()}</span>
                  </div>
                  <div className="text-slate-500">
                    {t.dims.map((d) => d.label ?? d.size).join(' × ')}
                    <span className="text-slate-400"> ({t.dims.map((d) => d.size).join(' × ')})</span>
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}
      <MemoryContribution nodeId={nodeId} />
    </Section>
  )
}

/** Size of a group: total, its formula, and the per-child breakdown (click a row to select that child). */
export function GroupSizeSection({ nodeId, summary, paramFormula, color }: { nodeId: string; summary?: GroupSummary; paramFormula?: string; color: string }) {
  const focus = useFocusNode()
  const total = summary?.params ?? 0
  const max = Math.max(1, ...(summary?.breakdown.map((b) => b.params) ?? []))

  return (
    <Section id="size" title="Size" meta={`${formatCount(total)} params`}>
      <SubHeading>Parameters (sum of the parts inside)</SubHeading>
      <TotalLine total={total} />
      <ModelShare nodeId={nodeId} />
      {paramFormula && <Formula lines={[`params = ${paramFormula}`, `= ${total.toLocaleString()}`]} />}
      <SubHeading>Breakdown</SubHeading>
      <div className="space-y-0.5">
        {summary?.breakdown.map((b) => (
          <button
            key={b.id}
            type="button"
            onClick={() => focus(b.id)}
            title="Select and show on the canvas"
            className="-mx-1.5 block w-[calc(100%+0.75rem)] rounded px-1.5 py-1 text-left text-xs hover:bg-slate-50"
          >
            <div className="flex items-baseline gap-2">
              <span className="min-w-0 truncate text-slate-700">{b.title}</span>
              <span className="ml-auto shrink-0 font-mono text-slate-500 tabular-nums">{b.params > 0 ? b.params.toLocaleString() : '—'}</span>
            </div>
            {b.params > 0 && (
              <div className="mt-0.5 h-1 rounded-full bg-slate-100">
                <div className="h-1 rounded-full" style={{ width: `${(b.params / max) * 100}%`, background: color }} />
              </div>
            )}
          </button>
        ))}
      </div>
      <MemoryContribution nodeId={nodeId} />
    </Section>
  )
}

function TotalLine({ total }: { total: number }) {
  return (
    <div className="mb-1.5 flex items-baseline gap-1.5">
      <span className="text-lg font-semibold text-slate-800 tabular-nums">{total.toLocaleString()}</span>
      <span className="text-xs text-slate-400">params{total >= 1000 ? ` (${formatCount(total)})` : ''}</span>
    </div>
  )
}

/** "6.4% of the model · Attention", or why this part / group isn't counted (engine/params.ts). */
function ModelShare({ nodeId }: { nodeId: string }) {
  const report = useCanvasStore((s) => s.inference.params)
  const part = report.parts[nodeId]
  const group = report.groups[nodeId]
  const counted = part ? (part.connected ? part.params : 0) : (group?.connected ?? 0)
  const uncounted = part ? (part.connected ? 0 : part.params) : (group?.unconnected ?? 0)
  if (counted === 0 && uncounted === 0) return null
  const share = report.total > 0 ? `${((counted / report.total) * 100).toFixed(1)}%` : '—'
  return (
    <div className="mb-2 space-y-0.5 text-xs">
      {counted > 0 && (
        <div className="text-slate-500">
          {share} of the model
          {part && (
            <>
              {' · '}
              <span className="inline-block h-2 w-2 rounded-sm align-middle" style={{ background: PARAM_CATEGORY_INFO[part.category].color }} />{' '}
              {PARAM_CATEGORY_INFO[part.category].label}
            </>
          )}
        </div>
      )}
      {uncounted > 0 && (
        <div className="rounded border border-amber-200 bg-amber-50 px-2 py-1 text-amber-800">
          {uncounted.toLocaleString()} params not counted in the model total: {part ? 'this part doesn’t' : 'these parts don’t'} feed Logits / Loss.
        </div>
      )}
    </div>
  )
}

/**
 * Phase 6 hook (R8.2): this part's / group's memory contribution — e.g. a `<SubHeading>Memory</SubHeading>`
 * followed by its saved-for-backward tensors and bytes for the current mode. Renders nothing until then.
 */
function MemoryContribution(_props: { nodeId: string }) {
  return null
}
