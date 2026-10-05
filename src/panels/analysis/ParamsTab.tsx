// Analysis panel → Parameters tab (R7): total + formula, category bar (click = highlight on canvas),
// per-layer breakdown (click = select + focus), insights. Everything comes from
// `inference.params` (src/engine/params.ts), derived in the store on every graph change.
import { Check, Info, Lightbulb } from 'lucide-react'
import { formatCount } from '../../engine/format'
import { PARAM_CATEGORIES, paramInsights, type HighlightKey, type ParamCategory, type ParamReport } from '../../engine/params'
import { PARAM_CATEGORY_INFO } from '../../nodes/registry'
import { nodeTitle } from '../../store/inference'
import { useCanvasStore } from '../../store/useCanvasStore'
import { useFocusNode } from '../drawer/useFocusNode'

/** The model-total rule, shown wherever the total appears. */
export const CONNECTED_RULE = 'Counted: parts that feed Logits / Loss. Parts that don’t (e.g. a stray Linear or a block that isn’t wired in) are listed as unconnected.'

const pct = (x: number, total: number) => {
  if (total <= 0) return '—'
  const v = (x / total) * 100
  return `${v > 0 && v < 1 ? v.toFixed(2) : v.toFixed(1)}%`
}

export function ParamsTab() {
  const report = useCanvasStore((s) => s.inference.params)
  return (
    <div className="grid gap-x-8 gap-y-4 px-4 py-3 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <div className="min-w-0 space-y-3">
        <Total report={report} />
        <FormulaBlock report={report} />
        <CategoryBreakdown report={report} />
      </div>
      <div className="min-w-0 space-y-3">
        <LayerList report={report} />
        <Insights report={report} />
      </div>
    </div>
  )
}

function Total({ report }: { report: ParamReport }) {
  const setHighlight = useCanvasStore((s) => s.setHighlight)
  const n = report.unconnectedIds.length
  return (
    <div>
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className="text-2xl font-semibold text-slate-800 tabular-nums">{report.total.toLocaleString()}</span>
        <span className="text-sm text-slate-500">parameters ({formatCount(report.total)})</span>
        <span className="text-slate-300" title={CONNECTED_RULE}>
          <Info size={13} className="inline" />
        </span>
      </div>
      <div className="text-[11px] text-slate-400">
        {report.hasOutput ? 'Connected model: parts that feed Logits / Loss.' : 'No Logits / Loss part on the canvas, so every part is counted.'}
        {report.unconnected > 0 && (
          <>
            {' '}
            <button type="button" onClick={() => setHighlight('unconnected')} className="text-amber-600 underline decoration-dotted hover:text-amber-700" title="Highlight them on the canvas">
              +{report.unconnected.toLocaleString()} ({formatCount(report.unconnected)}) in {n} unconnected part{n === 1 ? '' : 's'}, not counted
            </button>
          </>
        )}
      </div>
    </div>
  )
}

/** Hanging indent: wrapped formula lines continue under the text after "params = ". */
const HANG = 'break-words pl-[9ch] -indent-[9ch]'

function FormulaBlock({ report }: { report: ParamReport }) {
  const f = report.formula
  const diff = report.total - f.value
  return (
    <div>
      <div className="mb-1 text-[11px] font-medium tracking-wide text-slate-400 uppercase">{f.matches ? 'Formula' : 'Standard CS336 formula'}</div>
      <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-[12px] leading-relaxed whitespace-pre-wrap text-slate-700">
        <div className={HANG}>params = {f.symbolic}</div>
        <div className={HANG}>{'       '}= {f.substituted}</div>
        <div className="flex flex-wrap items-center gap-x-2">
          <span>{'       '}= {f.value.toLocaleString()}</span>
          {f.matches ? (
            <span className="flex items-center gap-1 font-sans text-[11px] text-emerald-600">
              <Check size={12} /> matches the canvas
            </span>
          ) : (
            <span className="font-sans text-[11px] text-amber-700">
              ≠ canvas {report.total.toLocaleString()} ({diff >= 0 ? '+' : '−'}
              {Math.abs(diff).toLocaleString()})
            </span>
          )}
        </div>
      </div>
      <p
        className="mt-1 text-[11px] leading-snug text-slate-400"
        title={`Terms: embedding V·d + L × (attention 4d² + SwiGLU 3d·d_ff + two RMSNorms 2d) + ln_final d${f.tied ? ' (lm_head tied: reuses the embedding, no d·V term)' : ' + lm_head d·V'}`}
      >
        V = vocab_size · d = d_model · L = {f.L} connected Transformer Block{f.L === 1 ? '' : 's'} · no biases ·{' '}
        {f.tied ? (
          <span className="text-indigo-600">weight tying on: lm_head reuses the embedding, so the + d·V term is dropped</span>
        ) : (
          'no weight tying'
        )}
      </p>
      {!f.matches && (
        <div className="mt-1 rounded-md border border-amber-200 bg-amber-50 px-2.5 py-1.5 text-[11px] leading-snug text-amber-800">
          The canvas differs from the standard structure (extra parts or overridden params), so the per-part sum above is the real count:
          <ul className="mt-0.5 list-disc pl-4">
            {f.diffs.map((d) => (
              <li key={d.category}>
                {PARAM_CATEGORY_INFO[d.category].label}: {d.actual.toLocaleString()} on the canvas vs {d.expected.toLocaleString()} in the formula
              </li>
            ))}
          </ul>
          {f.notes.length > 0 && (
            <ul className="mt-1 space-y-0.5 border-t border-amber-200 pt-1">
              {f.notes.map((n) => (
                <li key={n}>Why: {n}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

function CategoryBreakdown({ report }: { report: ParamReport }) {
  const highlight = useCanvasStore((s) => s.highlight)
  const setHighlight = useCanvasStore((s) => s.setHighlight)
  const toggle = (k: HighlightKey) => setHighlight(highlight === k ? null : k)
  const cats = PARAM_CATEGORIES.filter((c) => report.byCategory[c] > 0)
  const dimmed = (k: HighlightKey) => highlight !== null && highlight !== k

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[11px] font-medium tracking-wide text-slate-400 uppercase">
        <span>Where the parameters live</span>
        <span className="font-normal tracking-normal normal-case">click to highlight on the canvas · Esc clears</span>
      </div>
      <div className="flex h-5 w-full overflow-hidden rounded-md bg-slate-100">
        {cats.map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => toggle(c)}
            title={`${PARAM_CATEGORY_INFO[c].label}: ${report.byCategory[c].toLocaleString()} (${pct(report.byCategory[c], report.total)})`}
            className={`h-full border-r border-white/70 transition-opacity last:border-r-0 ${dimmed(c) ? 'opacity-30' : 'hover:opacity-80'}`}
            style={{ width: `${(report.byCategory[c] / report.total) * 100}%`, background: PARAM_CATEGORY_INFO[c].color }}
          />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {cats.map((c) => (
          <LegendChip key={c} k={c} value={report.byCategory[c]} share={pct(report.byCategory[c], report.total)} active={highlight === c} dimmed={dimmed(c)} onClick={() => toggle(c)} />
        ))}
        {report.tied.length > 0 && (
          <span
            className="flex items-center gap-1.5 rounded-full border border-dashed border-slate-200 px-2 py-0.5 text-xs"
            title={`Weight tying: the LM head reuses the embedding matrix, so it adds 0 (would be ${report.tied.reduce((a, t) => a + t.params, 0).toLocaleString()} untied)`}
          >
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: PARAM_CATEGORY_INFO.lm_head.color }} />
            <span className="font-medium text-slate-700">{PARAM_CATEGORY_INFO.lm_head.label}</span>
            <span className="text-indigo-600">tied · 0</span>
          </span>
        )}
        {report.unconnected > 0 && (
          <LegendChip k="unconnected" value={report.unconnected} share="not counted" active={highlight === 'unconnected'} dimmed={dimmed('unconnected')} onClick={() => toggle('unconnected')} dashed />
        )}
      </div>
    </div>
  )
}

function LegendChip(props: { k: HighlightKey; value: number; share: string; active: boolean; dimmed: boolean; dashed?: boolean; onClick: () => void }) {
  const info = PARAM_CATEGORY_INFO[props.k]
  return (
    <button
      type="button"
      onClick={props.onClick}
      title={`${info.help} — ${props.value.toLocaleString()} params`}
      className={`flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs transition ${props.dashed ? 'border-dashed' : ''} ${
        props.active ? 'border-slate-400 bg-slate-100 shadow-sm' : 'border-slate-200 hover:bg-slate-50'
      } ${props.dimmed ? 'opacity-50' : ''}`}
    >
      <span className="h-2.5 w-2.5 rounded-sm" style={{ background: info.color }} />
      <span className="font-medium text-slate-700">{info.label}</span>
      <span className="text-slate-500 tabular-nums">{formatCount(props.value)}</span>
      <span className="text-slate-400 tabular-nums">{props.share}</span>
    </button>
  )
}

function Insights({ report }: { report: ParamReport }) {
  const lines = paramInsights(report)
  if (lines.length === 0) return null
  return (
    <div className="space-y-0.5">
      {lines.map((l) => (
        <p key={l} className="flex gap-1.5 text-xs leading-snug text-slate-500">
          <Lightbulb size={13} className="mt-px shrink-0 text-amber-500" />
          <span>{l}</span>
        </p>
      ))}
    </div>
  )
}

function LayerList({ report }: { report: ParamReport }) {
  const nodes = useCanvasStore((s) => s.nodes)
  const setHighlight = useCanvasStore((s) => s.setHighlight)
  const focus = useFocusNode()
  const byId = new Map(nodes.map((n) => [n.id, n]))
  const max = Math.max(1, ...report.rows.map((r) => r.params))
  const L = report.formula.L

  return (
    <div className="min-w-0">
      <div className="mb-1 flex items-baseline justify-between text-[11px] font-medium tracking-wide text-slate-400 uppercase">
        <span>By layer</span>
        <span className="font-normal tracking-normal normal-case">
          {L} layer{L === 1 ? '' : 's'} · click a row to show it
        </span>
      </div>
      <div className="space-y-0.5">
        {report.rows.map((r) => {
          const node = byId.get(r.id)
          const title = node ? nodeTitle(node) : r.id
          const cats = PARAM_CATEGORIES.filter((c) => r.byCategory[c] > 0)
          const tied = report.tied.find((t) => t.id === r.id)
          const tiedTo = tied && byId.get(tied.to) ? nodeTitle(byId.get(tied.to)!) : tied?.to
          const sub = r.isLayer ? 'Transformer Block' : tied ? `LM head · tied to ${tiedTo}` : cats.map((c) => PARAM_CATEGORY_INFO[c].label).join(' + ')
          return (
            <button
              key={r.id}
              type="button"
              onClick={() => focus(r.id)}
              title="Select and show on the canvas"
              className="block w-full rounded px-1.5 py-1 text-left text-xs hover:bg-slate-50"
            >
              <div className="flex items-baseline gap-2">
                <span className="min-w-0 truncate font-medium text-slate-700">{title}</span>
                <span className="truncate text-[11px] text-slate-400">{sub}</span>
                <span className="ml-auto shrink-0 font-mono text-slate-600 tabular-nums">{r.params.toLocaleString()}</span>
                <span className="w-11 shrink-0 text-right text-[11px] text-slate-400 tabular-nums">{pct(r.params, report.total)}</span>
              </div>
              <MiniBar byCategory={r.byCategory} width={r.params / max} />
            </button>
          )
        })}
        {report.unconnected > 0 && (
          <button
            type="button"
            onClick={() => setHighlight('unconnected')}
            title="Highlight the unconnected parts on the canvas"
            className="block w-full rounded border border-dashed border-amber-200 px-1.5 py-1 text-left text-xs hover:bg-amber-50"
          >
            <div className="flex items-baseline gap-2">
              <span className="font-medium text-amber-700">Unconnected</span>
              <span className="truncate text-[11px] text-slate-400">
                {report.unconnectedIds.map((id) => (byId.get(id) ? nodeTitle(byId.get(id)!) : id)).join(', ')}
              </span>
              <span className="ml-auto shrink-0 font-mono text-amber-700 tabular-nums">{report.unconnected.toLocaleString()}</span>
              <span className="w-11 shrink-0 text-right text-[11px] text-slate-400">not counted</span>
            </div>
          </button>
        )}
        {report.rows.length === 0 && <p className="text-xs text-slate-400">No parts with weights in the model yet.</p>}
      </div>
    </div>
  )
}

/** A thin stacked bar (by category), scaled to the largest row. */
function MiniBar({ byCategory, width }: { byCategory: Record<ParamCategory, number>; width: number }) {
  const total = PARAM_CATEGORIES.reduce((a, c) => a + byCategory[c], 0)
  return (
    <div className="mt-0.5 h-1.5 rounded-full bg-slate-100">
      <div className="flex h-1.5 overflow-hidden rounded-full" style={{ width: `${width * 100}%` }}>
        {PARAM_CATEGORIES.filter((c) => byCategory[c] > 0).map((c) => (
          <div key={c} style={{ width: `${(byCategory[c] / total) * 100}%`, background: PARAM_CATEGORY_INFO[c].color }} />
        ))}
      </div>
    </div>
  )
}
