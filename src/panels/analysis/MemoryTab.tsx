// Analysis panel → Memory tab (R8): total for the current mode + dtype, components bar + formulas,
// activations by category (click = highlight on canvas), by layer and the biggest tensors
// (click = select + focus), and "where to optimise" insights. Everything comes from
// engine/memory.ts via `selectMemory` (memoised on graph + hyperparams + mode + checkpointing + generation).
// Forward mode has a "Generation (KV cache)" switch: then the estimate is weights + buffers + KV cache + one
// decode step, with a KV-cache section (per layer, formula, per-token cost).
import { Flame, Info, Lightbulb } from 'lucide-react'
import { formatBytes } from '../../engine/format'
import {
  MEMORY_CATEGORIES,
  MEMORY_COMPONENTS,
  kvFormula,
  memKey,
  memoryFormulas,
  memoryInsights,
  type MemoryCategory,
  type MemoryReport,
} from '../../engine/memory'
import { formatConcrete } from '../../engine/shape'
import { MEMORY_CATEGORY_INFO, MEMORY_COMPONENT_INFO } from '../../nodes/registry'
import { memoryInput } from '../../store/inference'
import { selectGenerationSettings, selectMemory, useCanvasStore } from '../../store/useCanvasStore'
import { NumberField } from '../NumberField'
import { useFocusNode } from '../drawer/useFocusNode'
import { GENERATION_EXPLAIN, MODE_INFO, pct, savedText, tensorName, titleOf, type NodeIndex } from './memoryText'

const TOP_TENSORS = 8

export function MemoryTab() {
  const report = useCanvasStore(selectMemory)
  const nodes = useCanvasStore((s) => s.nodes)
  const byId: NodeIndex = new Map(nodes.map((n) => [n.id, n]))
  return (
    <div className="grid gap-x-6 gap-y-4 px-4 py-3 @3xl:grid-cols-2 @5xl:grid-cols-[minmax(0,5fr)_minmax(0,4fr)_minmax(0,4fr)]">
      <div className="min-w-0 space-y-3">
        <Total report={report} byId={byId} />
        <GenerationControls report={report} />
        <Components report={report} />
        <Categories report={report} />
        <Formulas report={report} byId={byId} />
      </div>
      <div className="min-w-0 space-y-3">
        <KvCache report={report} byId={byId} />
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
          {report.generation ? 'Generation (KV cache)' : MODE_INFO[report.mode].label} · {dtype} ({report.b} B/value)
          {report.checkpointing ? ' · activation checkpointing' : ''}
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
        {report.generation ? GENERATION_EXPLAIN : MODE_INFO[report.mode].explain} {report.total.toLocaleString()} bytes.
        {skipped > 0 && <span className="text-amber-600"> {skipped} part{skipped === 1 ? '' : 's'} with unknown shapes not counted.</span>}
        {report.mode === 'forward' && report.activations.peakPart && (
          <>
            {' '}
            {report.generation ? 'Decode-step peak' : 'Peak'} at {titleOf(report.activations.peakPart, byId)}.
          </>
        )}
      </p>
    </div>
  )
}

/**
 * "Generation (KV cache)" switch + T_cache / batch inputs (Forward mode only). UI state: not saved, not undoable.
 * Empty settings follow context_length / batch_size; the engine clamps T_cache to 1 … context_length.
 */
function GenerationControls({ report }: { report: MemoryReport }) {
  const generation = useCanvasStore((s) => s.generation)
  const active = useCanvasStore((s) => selectGenerationSettings(s) !== null)
  const forward = report.mode === 'forward'
  const ctx = useCanvasStore((s) => s.hyperparams.context_length)
  const B = useCanvasStore((s) => s.hyperparams.batch_size)
  const setGeneration = useCanvasStore((s) => s.setGeneration)
  const length = report.generation?.length ?? Math.min(generation.length ?? ctx, ctx)
  const batch = report.generation?.batch ?? generation.batch ?? B
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-emerald-200 bg-emerald-50/50 px-2.5 py-1.5 text-xs text-slate-600">
      <label
        className={`flex items-center gap-1.5 font-medium ${forward ? 'text-emerald-800' : 'text-slate-400'}`}
        title={
          forward
            ? 'Estimate autoregressive generation: one new token per step, with the K and V of earlier tokens cached in every attention layer'
            : 'Switch the mode to Forward to estimate generation with a KV cache'
        }
      >
        <input type="checkbox" checked={active} disabled={!forward} onChange={(e) => setGeneration({ on: e.target.checked })} className="accent-emerald-600" />
        Generation (KV cache)
      </label>
      {!forward && <span className="text-slate-400">Forward mode only</span>}
      {active && (
        <>
          <span className="flex items-center gap-1" title={`Tokens held in the cache (prompt + generated so far) = the sequence length at the last step. At most context_length (${ctx}): RoPE’s tables end there.`}>
            T_cache
            <NumberField value={length} integer min={1} max={ctx} onCommit={(n) => setGeneration({ length: n })} className="w-16" />
            <span className="text-slate-400">/ {ctx}</span>
            {generation.length !== null && (
              <button type="button" onClick={() => setGeneration({ length: null })} className="text-emerald-700 hover:underline" title="Follow context_length again">
                = ctx
              </button>
            )}
          </span>
          <span className="flex items-center gap-1" title="Sequences generated together (B). Defaults to batch_size.">
            B
            <NumberField value={batch} integer min={1} onCommit={(n) => setGeneration({ batch: n })} className="w-14" />
            {generation.batch !== null && (
              <button type="button" onClick={() => setGeneration({ batch: null })} className="text-emerald-700 hover:underline" title="Follow batch_size again">
                = batch_size
              </button>
            )}
          </span>
        </>
      )}
    </div>
  )
}

/** KV cache: total, per-token cost, formula and one row per attention layer (click = focus its SDPA part). */
function KvCache({ report, byId }: { report: MemoryReport; byId: NodeIndex }) {
  const focus = useFocusNode()
  const gen = report.generation
  if (!gen) return null
  const { kv } = gen
  const max = Math.max(1, ...kv.items.map((x) => x.bytes))
  return (
    <div>
      <Heading note={`T_cache ${gen.length} · B ${gen.batch} · click to show`}>KV cache</Heading>
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className="text-lg font-semibold text-emerald-700 tabular-nums">{formatBytes(kv.total)}</span>
        <span className="text-xs text-slate-500">
          {kv.items.length} layer{kv.items.length === 1 ? '' : 's'} · +{kv.perToken.toLocaleString()} B per token per sequence ({pct(kv.total, report.total)} of the total)
        </span>
      </div>
      <div className="mt-1 rounded-md border border-slate-200 bg-slate-50 px-3 py-1.5 font-mono text-[11.5px] leading-relaxed break-words text-slate-700">
        {kvFormula(gen, report.b)}
        {kv.total > 0 && <span className="text-slate-500"> = {kv.total.toLocaleString()} B</span>}
      </div>
      <div className="mt-1 space-y-0.5">
        {kv.items.map((x) => (
          <button key={x.sdpa} type="button" onClick={() => focus(x.sdpa)} title="Select its attention (SDPA) part" className="block w-full rounded px-1.5 py-1 text-left text-xs hover:bg-slate-50">
            <div className="flex items-baseline gap-2">
              <span className="min-w-0 truncate font-medium text-slate-700">{titleOf(x.row, byId)}</span>
              <span className="min-w-0 truncate font-mono text-[11px] text-slate-400">K, V: {formatConcrete(x.kShape)} each</span>
              <span className="ml-auto shrink-0 font-mono text-slate-600 tabular-nums">{formatBytes(x.bytes)}</span>
            </div>
            <div className="mt-0.5 h-1.5 rounded-full bg-slate-100">
              <div className="h-1.5 rounded-full" style={{ width: `${(x.bytes / max) * 100}%`, background: MEMORY_COMPONENT_INFO.kv_cache.color }} />
            </div>
          </button>
        ))}
        {kv.skipped.length > 0 && <p className="text-[11px] text-amber-600">{kv.skipped.length} attention part(s) with unknown shapes not counted.</p>}
      </div>
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
  const lines = MEMORY_COMPONENTS.filter((c) => c !== 'kv_cache' || report.generation).map((c) => {
    let text = f[c]
    if (c === 'activations' && report.mode === 'forward' && peak)
      text = report.generation
        ? `one new token (T = 1; attention probs B·H·1·T_cache): peak live set at ${titleOf(peak, byId)}`
        : `peak live set at ${titleOf(peak, byId)} (inputs + outputs + temporaries)`
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
        {report.generation ? 'Activations at the decode-step peak' : report.mode === 'forward' ? 'Activations at the peak' : report.checkpointing ? 'Activations kept (checkpointing)' : 'Activations saved for backward'}
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
  const generation = useCanvasStore(selectGenerationSettings)
  const lines = memoryInsights(memoryInput(inference, hp, report.mode, report.checkpointing, generation), report)
  if (lines.length === 0) return null
  return (
    <div>
      <Heading>{report.generation ? 'KV cache: what to remember' : 'Where to optimise'}</Heading>
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
      <Heading note={forward ? (report.generation ? 'largest live set inside · one token' : 'largest live set inside') : 'by producing part · click to show'}>Activations by layer</Heading>
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
      <Heading note={`top ${top.length} of ${counted.length} · click to show`}>
        {report.generation ? 'Tensors at the decode-step peak' : report.mode === 'forward' ? 'Tensors at the peak' : 'Biggest tensors'}
      </Heading>
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
