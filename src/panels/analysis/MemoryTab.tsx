// Analysis panel → Memory tab (R8): total for the current mode + dtype, components bar + formulas,
// activations by category (click = highlight on canvas), by layer and the biggest tensors
// (click = select + focus), and "where to optimise" insights. Everything comes from
// engine/memory.ts via `selectMemory` (memoised on graph + hyperparams + mode + checkpointing).
import { Flame, Info, Lightbulb } from 'lucide-react'
import { formatBytes } from '../../engine/format'
import {
  MEMORY_CATEGORIES,
  MEMORY_COMPONENTS,
  memKey,
  memoryFormulas,
  memoryInsights,
  type MemoryCategory,
  type MemoryReport,
} from '../../engine/memory'
import { formatConcrete } from '../../engine/shape'
import { MEMORY_CATEGORY_INFO, MEMORY_COMPONENT_INFO } from '../../nodes/registry'
import { memoryInput } from '../../store/inference'
import { selectMemory, useCanvasStore } from '../../store/useCanvasStore'
import { useFocusNode } from '../drawer/useFocusNode'
import { MODE_INFO, pct, savedText, tensorName, titleOf, type NodeIndex } from './memoryText'

const TOP_TENSORS = 8

export function MemoryTab() {
  const report = useCanvasStore(selectMemory)
  const nodes = useCanvasStore((s) => s.nodes)
  const byId: NodeIndex = new Map(nodes.map((n) => [n.id, n]))
  return (
    <div className="grid gap-x-6 gap-y-4 px-4 py-3 @3xl:grid-cols-2 @5xl:grid-cols-[minmax(0,5fr)_minmax(0,4fr)_minmax(0,4fr)]">
      <div className="min-w-0 space-y-3">
        <Total report={report} byId={byId} />
        <Components report={report} />
        <Categories report={report} />
        <Formulas report={report} byId={byId} />
      </div>
      <div className="min-w-0 space-y-3">
        <Insights report={report} />
        <Layers report={report} byId={byId} />
      </div>
      <div className="min-w-0 space-y-3">
        <TopTensors report={report} byId={byId} />
      </div>
    </div>
  )
}

function Heading({ children, note }: { children: string; note?: string }) {
  return (
    <div className="mb-1 flex items-baseline justify-between gap-2 text-[11px] font-medium tracking-wide text-slate-400 uppercase">
      <span>{children}</span>
      {note && <span className="truncate font-normal tracking-normal normal-case">{note}</span>}
    </div>
  )
}

function Total({ report, byId }: { report: MemoryReport; byId: NodeIndex }) {
  const dtype = useCanvasStore((s) => s.hyperparams.dtype)
  const heat = useCanvasStore((s) => s.memoryHeat)
  const setHeat = useCanvasStore((s) => s.setMemoryHeat)
  const skipped = report.activations.skipped.length
  return (
    <div>
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className="text-2xl font-semibold text-slate-800 tabular-nums">{formatBytes(report.total)}</span>
        <span className="text-sm text-slate-500">
          {MODE_INFO[report.mode].label} · {dtype} ({report.b} B/value){report.checkpointing ? ' · activation checkpointing' : ''}
        </span>
        <span className="text-slate-300" title="Educational estimate: no allocator overhead, no CUDA context, no temporary gradients of activations; autograd may keep a few more intermediates than listed here (e.g. inside a hand-written softmax).">
          <Info size={13} className="inline" />
        </span>
        <button
          type="button"
          onClick={() => setHeat(!heat)}
          title="Tint parts on the canvas by the activation memory they hold"
          className={`ml-auto flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs transition ${
            heat ? 'border-red-300 bg-red-50 text-red-700' : 'border-slate-200 text-slate-500 hover:bg-slate-50'
          }`}
        >
          <Flame size={12} /> Heat on canvas
        </button>
      </div>
      <p className="text-[11px] leading-snug text-slate-400">
        {MODE_INFO[report.mode].explain} {report.total.toLocaleString()} bytes.
        {skipped > 0 && <span className="text-amber-600"> {skipped} part{skipped === 1 ? '' : 's'} with unknown shapes not counted.</span>}
        {report.mode === 'forward' && report.activations.peakPart && <> Peak at {titleOf(report.activations.peakPart, byId)}.</>}
      </p>
    </div>
  )
}

function StackedBar({ items }: { items: { key: string; value: number; color: string; title: string; dimmed?: boolean; onClick?: () => void }[] }) {
  const total = items.reduce((a, x) => a + x.value, 0)
  return (
    <div className="flex h-5 w-full overflow-hidden rounded-md bg-slate-100">
      {total > 0 &&
        items
          .filter((x) => x.value > 0)
          .map((x) => (
            <button
              key={x.key}
              type="button"
              onClick={x.onClick}
              disabled={!x.onClick}
              title={x.title}
              className={`h-full border-r border-white/70 transition-opacity last:border-r-0 ${x.dimmed ? 'opacity-30' : x.onClick ? 'hover:opacity-80' : 'cursor-default'}`}
              style={{ width: `${(x.value / total) * 100}%`, background: x.color }}
            />
          ))}
    </div>
  )
}

function Chip({ color, label, value, share, title, active, dimmed, onClick }: { color: string; label: string; value: number; share: string; title: string; active?: boolean; dimmed?: boolean; onClick?: () => void }) {
  const Tag = onClick ? 'button' : 'span'
  return (
    <Tag
      {...(onClick ? { type: 'button' as const, onClick } : {})}
      title={title}
      className={`flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs transition ${active ? 'border-slate-400 bg-slate-100 shadow-sm' : 'border-slate-200'} ${
        onClick ? 'hover:bg-slate-50' : ''
      } ${dimmed ? 'opacity-50' : ''}`}
    >
      <span className="h-2.5 w-2.5 rounded-sm" style={{ background: color }} />
      <span className="font-medium text-slate-700">{label}</span>
      <span className="text-slate-500 tabular-nums">{formatBytes(value)}</span>
      <span className="text-slate-400 tabular-nums">{share}</span>
    </Tag>
  )
}

function Components({ report }: { report: MemoryReport }) {
  const comps = MEMORY_COMPONENTS.filter((c) => report.byComponent[c] > 0)
  return (
    <div>
      <Heading>By component</Heading>
      <StackedBar
        items={comps.map((c) => ({
          key: c,
          value: report.byComponent[c],
          color: MEMORY_COMPONENT_INFO[c].color,
          title: `${MEMORY_COMPONENT_INFO[c].label}: ${formatBytes(report.byComponent[c])} (${pct(report.byComponent[c], report.total)})`,
        }))}
      />
      <div className="mt-2 flex flex-wrap gap-1.5">
        {comps.map((c) => (
          <Chip
            key={c}
            color={MEMORY_COMPONENT_INFO[c].color}
            label={MEMORY_COMPONENT_INFO[c].label}
            value={report.byComponent[c]}
            share={pct(report.byComponent[c], report.total)}
            title={`${MEMORY_COMPONENT_INFO[c].help} — ${report.byComponent[c].toLocaleString()} bytes`}
          />
        ))}
      </div>
    </div>
  )
}

function Formulas({ report, byId }: { report: MemoryReport; byId: NodeIndex }) {
  const f = memoryFormulas(report)
  const tiedParams = useCanvasStore((s) => s.inference.params.tied.reduce((a, t) => a + t.params, 0))
  const peak = report.activations.peakPart
  const lines = MEMORY_COMPONENTS.map((c) => {
    let text = f[c]
    if (c === 'activations' && report.mode === 'forward' && peak) text = `peak live set at ${titleOf(peak, byId)} (inputs + outputs + temporaries)`
    return { c, text, value: report.byComponent[c] }
  })
  return (
    <div>
      <Heading>Formulas</Heading>
      <div className="space-y-0.5 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-[11.5px] leading-relaxed text-slate-700">
        {lines.map(({ c, text, value }) => (
          <div key={c} className="flex gap-2">
            <span className="w-24 shrink-0 text-slate-500">{c}</span>
            <span className={`min-w-0 flex-1 break-words ${value === 0 ? 'text-slate-400' : ''}`}>
              {text}
              {value > 0 && <span className="text-slate-500"> = {value.toLocaleString()} B</span>}
            </span>
          </div>
        ))}
      </div>
      <p className="mt-1 text-[11px] text-slate-400">
        P = {report.P.toLocaleString()} connected params · bytes = {report.b} per value (int64 token ids: 8)
        {tiedParams > 0 && (
          <span className="text-indigo-600">
            {' '}
            · weight tying: the shared embedding / LM head matrix is in P once (weights, its gradient and AdamW state: −{tiedParams.toLocaleString()} params)
          </span>
        )}
      </p>
    </div>
  )
}

function Categories({ report }: { report: MemoryReport }) {
  const highlight = useCanvasStore((s) => s.highlight)
  const setHighlight = useCanvasStore((s) => s.setHighlight)
  const a = report.activations
  const cats = MEMORY_CATEGORIES.filter((c) => a.byCategory[c] > 0)
  const total = cats.reduce((x, c) => x + a.byCategory[c], 0)
  const toggle = (c: MemoryCategory) => setHighlight(highlight === memKey(c) ? null : memKey(c))
  const dimmed = (c: MemoryCategory) => highlight !== null && highlight !== memKey(c)
  return (
    <div>
      <Heading note="click to highlight · Esc clears">
        {report.mode === 'forward' ? 'Activations at the peak' : report.checkpointing ? 'Activations kept (checkpointing)' : 'Activations saved for backward'}
      </Heading>
      <StackedBar
        items={cats.map((c) => ({
          key: c,
          value: a.byCategory[c],
          color: MEMORY_CATEGORY_INFO[c].color,
          title: `${MEMORY_CATEGORY_INFO[c].label}: ${formatBytes(a.byCategory[c])} (${pct(a.byCategory[c], total)})`,
          dimmed: dimmed(c),
          onClick: () => toggle(c),
        }))}
      />
      <div className="mt-2 flex flex-wrap gap-1.5">
        {cats.map((c) => (
          <Chip
            key={c}
            color={MEMORY_CATEGORY_INFO[c].color}
            label={MEMORY_CATEGORY_INFO[c].label}
            value={a.byCategory[c]}
            share={pct(a.byCategory[c], total)}
            title={MEMORY_CATEGORY_INFO[c].help}
            active={highlight === memKey(c)}
            dimmed={dimmed(c)}
            onClick={() => toggle(c)}
          />
        ))}
        {cats.length === 0 && <span className="text-xs text-slate-400">No activations (nothing connected with known shapes).</span>}
      </div>
    </div>
  )
}

function Insights({ report }: { report: MemoryReport }) {
  const inference = useCanvasStore((s) => s.inference)
  const hp = useCanvasStore((s) => s.hyperparams)
  // Re-runs the estimate for the "what if" lines (checkpointing, bf16) — cheap on graphs this size.
  const lines = memoryInsights(memoryInput(inference, hp, report.mode, report.checkpointing), report)
  if (lines.length === 0) return null
  return (
    <div>
      <Heading>Where to optimise</Heading>
      <div className="space-y-1">
        {lines.map((l) => (
          <p key={l} className="flex gap-1.5 text-xs leading-snug text-slate-600">
            <Lightbulb size={13} className="mt-px shrink-0 text-amber-500" />
            <span>{l}</span>
          </p>
        ))}
      </div>
    </div>
  )
}

function Layers({ report, byId }: { report: MemoryReport; byId: NodeIndex }) {
  const focus = useFocusNode()
  const a = report.activations
  const max = Math.max(1, ...a.rows.map((r) => r.bytes))
  const forward = report.mode === 'forward'
  return (
    <div>
      <Heading note={forward ? 'largest live set inside' : 'by producing part · click to show'}>Activations by layer</Heading>
      <div className="space-y-0.5">
        {a.rows.map((r) => {
          const cats = MEMORY_CATEGORIES.filter((c) => r.byCategory[c] > 0)
          const recompute = a.recomputeBlock === r.id
          return (
            <button key={r.id} type="button" onClick={() => focus(r.id)} title="Select and show on the canvas" className="block w-full rounded px-1.5 py-1 text-left text-xs hover:bg-slate-50">
              <div className="flex items-baseline gap-2">
                <span className="min-w-0 truncate font-medium text-slate-700">{titleOf(r.id, byId)}</span>
                {recompute && <span className="truncate text-[11px] text-slate-400">recomputed in backward</span>}
                <span className="ml-auto shrink-0 font-mono text-slate-600 tabular-nums">{formatBytes(r.bytes)}</span>
                {!forward && <span className="w-9 shrink-0 text-right text-[11px] text-slate-400 tabular-nums">{pct(r.bytes, a.total)}</span>}
              </div>
              <div className="mt-0.5 h-1.5 rounded-full bg-slate-100">
                <div className="flex h-1.5 overflow-hidden rounded-full" style={{ width: `${(r.bytes / max) * 100}%` }}>
                  {forward ? (
                    <div className="w-full" style={{ background: '#ef4444' }} />
                  ) : (
                    cats.map((c) => <div key={c} style={{ width: `${(r.byCategory[c] / r.bytes) * 100}%`, background: MEMORY_CATEGORY_INFO[c].color }} />)
                  )}
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function TopTensors({ report, byId }: { report: MemoryReport; byId: NodeIndex }) {
  const focus = useFocusNode()
  const counted = report.activations.tensors.filter((t) => t.counted)
  const top = counted.slice(0, TOP_TENSORS)
  return (
    <div>
      <Heading note={`top ${top.length} of ${counted.length} · click to show`}>{report.mode === 'forward' ? 'Tensors at the peak' : 'Biggest tensors'}</Heading>
      <div className="space-y-0.5">
        {top.map((t) => (
          <button key={t.key} type="button" onClick={() => focus(t.owner)} className="block w-full rounded px-1.5 py-1 text-left text-xs hover:bg-slate-50">
            <div className="flex items-baseline gap-2">
              <span className="mt-0.5 h-2 w-2 shrink-0 self-center rounded-sm" style={{ background: MEMORY_CATEGORY_INFO[t.category].color }} />
              <span className="min-w-0 truncate text-slate-700" title={tensorName(t, byId)}>
                {tensorName(t, byId)}
              </span>
              <span className="ml-auto shrink-0 font-mono text-slate-600 tabular-nums">{formatBytes(t.bytes)}</span>
            </div>
            <div className="truncate pl-4 text-[11px] text-slate-400">
              <span className="font-mono">{formatConcrete(t.shape)}</span>
              {report.mode !== 'forward' && (
                <>
                  {' '}
                  · {savedText(t, byId)}
                </>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
