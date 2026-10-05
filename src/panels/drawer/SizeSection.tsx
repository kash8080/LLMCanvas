// "Size" section: parameter count (+ formula, weight tensors / per-child breakdown) and the
// memory contribution for the current mode (<MemoryContribution>, engine/memory.ts).
import type { ReactNode } from 'react'
import { formatBytes, formatCount } from '../../engine/format'
import type { MemoryReport, MemTensor } from '../../engine/memory'
import { formatConcrete } from '../../engine/shape'
import type { ParamCountResult } from '../../engine/types'
import { nodeTitle, type GroupSummary } from '../../store/inference'
import { selectMemory, useCanvasStore } from '../../store/useCanvasStore'
import { MEMORY_CATEGORY_INFO, PARAM_CATEGORY_INFO } from '../../nodes/registry'
import { MODE_INFO, pct, savedText, saversText, tensorName, titleOf, type NodeIndex } from '../analysis/memoryText'
import { Formula } from './DocsSections'
import { Section, SubHeading } from './ui'
import { useFocusNode } from './useFocusNode'

/** Size of one part: its weight tensors and how the count is computed. */
export function PartSizeSection({ nodeId, title, count }: { nodeId: string; title?: string; count: ParamCountResult }) {
  const { total, tensors, tied } = count
  const tiedTitle = useCanvasStore((s) => {
    const n = tied ? s.nodes.find((x) => x.id === tied.to) : undefined
    return n ? nodeTitle(n) : (tied?.to ?? '')
  })
  const own = tied ? tied.params : total
  const symbolic = tensors.map((t) => t.dims.map((d) => d.label ?? String(d.size)).join(' · ')).join(' + ')
  const concrete = tensors.map((t) => t.dims.map((d) => String(d.size)).join(' · ')).join(' + ')

  return (
    <Section id="size" title="Size" meta={tied ? '0 params · tied' : `${formatCount(total)} params`}>
      <SubHeading>Parameters</SubHeading>
      {own === 0 ? (
        <p className="text-xs text-slate-500">No learned parameters.</p>
      ) : (
        <>
          <TotalLine total={total} />
          {tied && (
            <div className="mb-2 rounded border border-indigo-200 bg-indigo-50 px-2 py-1 text-xs leading-snug text-indigo-800">
              Weight tying is on: this LM head reuses <b>{tiedTo(tiedTitle)}</b>’s matrix (vocab_size × d_model), so its {tied.params.toLocaleString()} weights are
              counted once, under the embedding — 0 here. Turn it off in Hyperparams (tie_embeddings).
            </div>
          )}
          <ModelShare nodeId={nodeId} />
          <Formula
            lines={
              tied
                ? [`shared weight = ${symbolic}`, `${symbolic === concrete ? '' : `= ${concrete} `}= ${tied.params.toLocaleString()} (counted under ${tiedTitle})`]
                : [`params = ${symbolic}`, `${symbolic === concrete ? '' : `= ${concrete} `}= ${total.toLocaleString()}`]
            }
          />
          <div className="mt-2 space-y-1">
            {tensors.map((t) => {
              const n = t.dims.reduce((a, d) => a * d.size, 1)
              return (
                <div key={t.name} className="rounded-md border border-slate-100 bg-slate-50 px-2.5 py-1.5 font-mono text-xs">
                  <div className="flex items-baseline gap-2">
                    <span className="min-w-0 break-all text-slate-700">
                      {title ? `${title}.` : ''}
                      {t.name}
                      {tied && <span className="font-sans text-indigo-600"> = {tiedTitle}.weight</span>}
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
export function GroupSizeSection({
  nodeId,
  summary,
  paramFormula,
  standard,
  color,
}: {
  nodeId: string
  summary?: GroupSummary
  paramFormula?: string
  /** Params of the standard template for the current hyperparams (GroupDef.standardParams). */
  standard?: number
  color: string
}) {
  const focus = useFocusNode()
  const total = summary?.params ?? 0
  const max = Math.max(1, ...(summary?.breakdown.map((b) => b.params) ?? []))

  return (
    <Section id="size" title="Size" meta={`${formatCount(total)} params`}>
      <SubHeading>Parameters (sum of the parts inside)</SubHeading>
      <TotalLine total={total} />
      <ModelShare nodeId={nodeId} />
      {paramFormula &&
        (standard === undefined || standard === total ? (
          <Formula lines={[`params = ${paramFormula}`, `= ${total.toLocaleString()}`]} />
        ) : (
          <>
            <Formula lines={[`standard: ${paramFormula}`, `= ${standard.toLocaleString()}`]} />
            <p className="mt-1 text-[11px] leading-snug text-amber-700">
              This group differs from the standard template (a swapped part, e.g. a non-gated FFN or LayerNorm, or an overridden param):
              {` ${total >= standard ? '+' : '−'}${Math.abs(total - standard).toLocaleString()}`}. The sum of its parts above is the real count.
            </p>
          </>
        ))}
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

const tiedTo = (title: string) => title || 'the embedding'

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
  // Weight tying: this Embedding's matrix is also the LM head's weight.
  const shared = report.tied.some((t) => t.to === nodeId)
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
      {shared && <div className="text-indigo-700">Also the LM head’s weight (weight tying): the matrix is counted once, here.</div>}
      {uncounted > 0 && (
        <div className="rounded border border-amber-200 bg-amber-50 px-2 py-1 text-amber-800">
          {uncounted.toLocaleString()} params not counted in the model total: {part ? 'this part doesn’t' : 'these parts don’t'} feed Logits / Loss.
        </div>
      )}
    </div>
  )
}

/**
 * Memory for the current mode (R8.2). Parts: the tensors they save for backward (or their live set at
 * the forward peak) and their weights' share; groups: the activations of the parts inside plus the
 * weights / gradients / optimizer state of their params.
 */
function MemoryContribution({ nodeId }: { nodeId: string }) {
  const report = useCanvasStore(selectMemory)
  const nodes = useCanvasStore((s) => s.nodes)
  const paramReport = useCanvasStore((s) => s.inference.params)
  const byId: NodeIndex = new Map(nodes.map((n) => [n.id, n]))
  const isGroup = byId.get(nodeId)?.type === 'group'
  const counted = paramReport.parts[nodeId]?.connected ? paramReport.parts[nodeId].params : (paramReport.groups[nodeId]?.connected ?? 0)
  const a = report.activations
  const acts = isGroup ? (report.groups[nodeId]?.activations ?? 0) : (a.byPart[nodeId] ?? 0)

  return (
    <>
      <SubHeading>
        Memory · {MODE_INFO[report.mode].label} · {report.b} B/value{report.checkpointing ? ' · checkpointing' : ''}
      </SubHeading>
      {counted > 0 && <WeightsLine params={counted} report={report} />}
      {report.mode === 'forward' ? (
        <p className="text-xs text-slate-500">
          {acts > 0 ? (
            <>
              {isGroup ? 'Largest live set inside' : 'Live while it runs (inputs + outputs + temporaries)'}:{' '}
              <span className="font-mono text-slate-700">{formatBytes(acts)}</span>.{' '}
            </>
          ) : (
            'Nothing live here (not in the model, or shape unknown). '
          )}
          {a.peakPart === nodeId ? (
            <span className="font-medium text-red-600">This is the forward peak.</span>
          ) : (
            a.peakPart && <>Forward peak: {titleOf(a.peakPart, byId)} ({formatBytes(a.total)}).</>
          )}
        </p>
      ) : isGroup ? (
        <p className="text-xs text-slate-500">
          Activations produced inside: <span className="font-mono text-slate-700">{formatBytes(acts)}</span> ({pct(acts, a.total)} of activations)
          {a.recomputeBlock === nodeId && ' — this block is the one recomputed during backward (checkpointing).'}
        </p>
      ) : (
        <PartSaved nodeId={nodeId} report={report} byId={byId} acts={acts} />
      )}
    </>
  )
}

/** "weights 1.0 MB · grads 1.0 MB · AdamW 2.1 MB" for these params in the current mode. */
function WeightsLine({ params, report }: { params: number; report: MemoryReport }) {
  const w = params * report.b
  const parts = [`weights ${formatBytes(w)}`]
  if (report.mode !== 'forward') parts.push(`grads ${formatBytes(w)}`)
  if (report.mode === 'train') parts.push(`AdamW m+v ${formatBytes(2 * w)}`)
  return <p className="mb-1 text-xs text-slate-500">{parts.join(' · ')}</p>
}

function PartSaved({ nodeId, report, byId, acts }: { nodeId: string; report: MemoryReport; byId: NodeIndex; acts: number }) {
  const tensors = report.activations.tensors
  const saves = tensors.filter((t) => t.savedBy.includes(nodeId))
  const keptForOthers = tensors.filter((t) => t.owner === nodeId && !t.savedBy.includes(nodeId))
  const nothing = saves.length === 0 && keptForOthers.length === 0
  return (
    <div className="space-y-1">
      {nothing && <p className="text-xs text-slate-500">Saves nothing for backward.</p>}
      {saves.length > 0 && <div className="text-[11px] text-slate-400">Saves for backward</div>}
      {saves.map((t) => (
        <TensorRow key={t.key} t={t} byId={byId}>
          {t.owner !== nodeId && <>counted under {titleOf(t.owner, byId)}, which produced it</>}
          {t.savedBy.length > 1 && <>{t.owner !== nodeId ? ' · ' : ''}counted once — shared with {saversText(t, byId, nodeId)}</>}
          {!t.counted && <span className="text-amber-700"> · not kept: recomputed during backward (checkpointing)</span>}
          {t.role === 'recompute' && ' · held while this block is recomputed'}
          {t.role === 'block_input' && ' · kept as the checkpointed block input'}
        </TensorRow>
      ))}
      {keptForOthers.length > 0 && <div className="pt-1 text-[11px] text-slate-400">Its output, kept for backward by others</div>}
      {keptForOthers.map((t) => (
        <TensorRow key={t.key} t={t} byId={byId}>
          {savedText(t, byId)}
          {!t.counted && <span className="text-amber-700"> · not kept: recomputed during backward (checkpointing)</span>}
        </TensorRow>
      ))}
      {acts > 0 && (
        <p className="text-xs text-slate-500">
          Counted under this part: <span className="font-mono text-slate-700">{formatBytes(acts)}</span> ({pct(acts, report.activations.total)} of activations)
        </p>
      )}
    </div>
  )
}

function TensorRow({ t, byId, children }: { t: MemTensor; byId: NodeIndex; children?: ReactNode }) {
  return (
    <div className={`rounded-md border border-slate-100 bg-slate-50 px-2.5 py-1.5 text-xs ${t.counted ? '' : 'opacity-60'}`}>
      <div className="flex items-baseline gap-2">
        <span className="mt-0.5 h-2 w-2 shrink-0 self-center rounded-sm" style={{ background: MEMORY_CATEGORY_INFO[t.category].color }} title={MEMORY_CATEGORY_INFO[t.category].label} />
        <span className="min-w-0 truncate text-slate-700">{tensorName(t, byId)}</span>
        <span className="ml-auto shrink-0 font-mono text-slate-600 tabular-nums">{formatBytes(t.bytes)}</span>
      </div>
      <div className="pl-4 font-mono text-[11px] text-slate-400">{formatConcrete(t.shape)}{t.shape.dtype === 'int64' ? ' int64' : ''}</div>
      {children && <div className="pl-4 text-[11px] leading-snug text-slate-500">{children}</div>}
    </div>
  )
}
