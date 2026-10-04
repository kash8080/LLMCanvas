import { ChevronDown, SlidersHorizontal } from 'lucide-react'
import { numLayers } from '../canvas/groupTemplates'
import { FLOAT_DTYPES, HYPERPARAM_INFO, validateHyperparams } from '../engine/hyperparams'
import type { FloatDType } from '../engine/types'
import { useCanvasStore } from '../store/useCanvasStore'
import { NumberField } from './NumberField'

/** Toolbar "Hyperparams ▾" button + popover with the global hyperparameters (R6). */
export function HyperparamsMenu() {
  const open = useCanvasStore((s) => s.hyperparamsOpen)
  const setOpen = useCanvasStore((s) => s.setHyperparamsOpen)
  const hp = useCanvasStore((s) => s.hyperparams)
  const setHyperparam = useCanvasStore((s) => s.setHyperparam)
  const layers = useCanvasStore((s) => numLayers(s.nodes))
  const problems = validateHyperparams(hp)
  const dHead = hp.d_model / hp.num_heads

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm transition ${
          open ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
        }`}
      >
        <SlidersHorizontal size={16} />
        Hyperparams
        {problems.length > 0 && <span className="h-2 w-2 rounded-full bg-red-500" title={problems.join('\n')} />}
        <ChevronDown size={14} />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onPointerDown={() => setOpen(false)} />
          <div className="absolute top-full left-0 z-50 mt-1 w-80 rounded-lg border border-slate-200 bg-white p-3 shadow-xl">
            <div className="mb-2 text-[11px] font-semibold tracking-wide text-slate-400 uppercase">Global hyperparameters</div>
            <div className="grid grid-cols-[1fr_auto_96px] items-center gap-x-2 gap-y-1.5 text-xs">
              {HYPERPARAM_INFO.map((info) => (
                <Row key={info.key} label={info.key} symbol={info.symbol} help={info.help}>
                  <NumberField
                    value={hp[info.key]}
                    integer={info.kind === 'int'}
                    min={info.kind === 'int' ? 1 : Number.MIN_VALUE}
                    onCommit={(n) => setHyperparam(info.key, n)}
                    className="w-full"
                  />
                </Row>
              ))}
              <Row label="dtype" symbol="" help="Element type for weights and activations (int64 token ids are always 8 bytes).">
                <select
                  value={hp.dtype}
                  onChange={(e) => setHyperparam('dtype', e.target.value as FloatDType)}
                  className="w-full rounded border border-slate-300 px-1 py-0.5 text-xs outline-none focus:border-indigo-400"
                >
                  {FLOAT_DTYPES.map((d) => (
                    <option key={d} value={d}>
                      {d} ({d === 'fp32' ? 4 : 2} B)
                    </option>
                  ))}
                </select>
              </Row>
              <Row label="d_head" symbol="d_model / H" help="Derived: size of each attention head.">
                <span className="px-1.5 font-mono text-slate-500">{Number.isInteger(dHead) ? dHead : dHead.toFixed(2)}</span>
              </Row>
              <Row label="num_layers" symbol="L" help="Derived: the number of Transformer Block groups on the canvas. Add a layer by dragging a Transformer Block from the palette or duplicating one (⌘D).">
                <span className="px-1.5 font-mono text-slate-500">
                  {layers} <span className="font-sans text-[11px] text-slate-400">= blocks</span>
                </span>
              </Row>
            </div>
            {problems.length > 0 && (
              <ul className="mt-3 space-y-0.5 rounded-md bg-red-50 p-2 text-[11px] text-red-700">
                {problems.map((p) => (
                  <li key={p}>⚠ {p}</li>
                ))}
              </ul>
            )}
            <p className="mt-3 text-[11px] leading-snug text-slate-400">
              Parts with a 🔗 param follow these values. Unlink a param in the Details drawer to override it locally.
            </p>
          </div>
        </>
      )}
    </div>
  )
}

function Row({ label, symbol, help, children }: { label: string; symbol: string; help: string; children: React.ReactNode }) {
  return (
    <>
      <span className="font-mono text-slate-700" title={help}>
        {label}
      </span>
      <span className="text-[11px] text-slate-400">{symbol}</span>
      {children}
    </>
  )
}
