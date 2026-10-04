import { AlertTriangle, SlidersHorizontal } from 'lucide-react'
import { numLayers } from '../../canvas/groupTemplates'
import type { AppNode } from '../../canvas/types'
import { formatCount } from '../../engine/format'
import { isProxyType } from '../../engine/groups'
import { HYPERPARAM_INFO } from '../../engine/hyperparams'
import { nodeTitle } from '../../store/inference'
import { useCanvasStore } from '../../store/useCanvasStore'
import { CONNECTED_RULE } from '../analysis/ParamsTab'
import { DrawerBody, DrawerHeader, Pill, Section } from './ui'
import { useFocusNode } from './useFocusNode'

const HOW_TO: string[] = [
  'Drag a part or a group (Transformer Block, attention, SwiGLU) from the palette onto the canvas.',
  'Connect: drag from a green output (bottom) to a blue input (top), or drop onto a part’s body. Drop on empty canvas to pick a new part that gets connected.',
  'Click a port dot to see the tensor shape, dtype and size flowing through it.',
  'Zoom in (pinch or ⌘ + scroll) to open the blocks, and further to open attention and SwiGLU.',
  'Click a part or group to read its docs and edit its parameters here.',
  '⌘D duplicates the selection; Delete / Backspace (or the trash button in this panel) removes it; ⌘Z / ⇧⌘Z undo and redo. Shift + drag box-selects. Right-click anything for more.',
]

/** "Title › path" of a node, e.g. "Block 1 › attn › q_proj". */
function nodePath(n: AppNode, byId: Map<string, AppNode>): string {
  const parts = [nodeTitle(n)]
  for (let p = n.parentId ? byId.get(n.parentId) : undefined; p; p = p.parentId ? byId.get(p.parentId) : undefined) parts.unshift(nodeTitle(p))
  return parts.join(' › ')
}

/** Drawer content when nothing is selected: model stats, problems, hyperparams, how to use. */
export function ModelSummary() {
  const nodes = useCanvasStore((s) => s.nodes)
  const inference = useCanvasStore((s) => s.inference)
  const hp = useCanvasStore((s) => s.hyperparams)
  const setHyperparamsOpen = useCanvasStore((s) => s.setHyperparamsOpen)
  const openAnalysis = useCanvasStore((s) => s.openAnalysis)
  const setHighlight = useCanvasStore((s) => s.setHighlight)
  const focus = useFocusNode()

  const byId = new Map(nodes.map((n) => [n.id, n]))
  const parts = nodes.filter((n) => n.type === 'part' && !isProxyType(n.data.partType))
  const groups = nodes.filter((n) => n.type === 'group')
  const { total: totalParams, unconnected, unconnectedIds } = inference.params
  const layers = numLayers(nodes)
  const problems = nodes.flatMap((n) => {
    const r = n.type === 'part' ? inference.nodes[n.id] : undefined
    return r?.status === 'error' ? r.errors.map((error) => ({ id: n.id, where: nodePath(n, byId), error })) : []
  })
  const dHead = hp.d_model / hp.num_heads

  return (
    <>
      <DrawerHeader
        color="#6366f1"
        title="Model summary"
        subtitle="Nothing selected — click a part to see its details"
        badge={
          problems.length > 0 ? (
            <Pill className="bg-red-50 text-red-700 ring-red-200">
              <AlertTriangle size={11} /> {problems.length} problem{problems.length > 1 ? 's' : ''}
            </Pill>
          ) : undefined
        }
      />
      <DrawerBody>
        <Section id="summary-model" title="Model">
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Parameters" value={formatCount(totalParams)} title={`${totalParams.toLocaleString()} parameters. ${CONNECTED_RULE}`} />
            <Stat label="Layers (num_layers)" value={String(layers)} title="Number of Transformer Blocks on the canvas" />
            <Stat label="Parts" value={String(parts.length)} title="Parts on the canvas (inside groups too)" />
            <Stat label="Problems" value={String(problems.length)} tone={problems.length > 0 ? 'bad' : 'good'} />
          </div>
          <p className="mt-2 text-[11px] text-slate-400">
            {groups.length} group{groups.length === 1 ? '' : 's'} · {totalParams.toLocaleString()} parameters in the model{' '}
            <button type="button" onClick={() => openAnalysis('params')} className="text-indigo-600 hover:underline">
              breakdown
            </button>
          </p>
          {unconnected > 0 && (
            <button
              type="button"
              onClick={() => setHighlight('unconnected')}
              title={`${CONNECTED_RULE} Click to highlight them.`}
              className="mt-1 block text-left text-[11px] text-amber-700 hover:underline"
            >
              +{unconnected.toLocaleString()} params in {unconnectedIds.length} unconnected part{unconnectedIds.length === 1 ? '' : 's'} (not counted — they don’t feed Logits / Loss)
            </button>
          )}
        </Section>

        {problems.length > 0 && (
          <Section id="summary-problems" title="Problems" meta={problems.length}>
            <ul className="space-y-1">
              {problems.map((p) => (
                <li key={`${p.id}:${p.error}`}>
                  <button
                    type="button"
                    onClick={() => focus(p.id)}
                    title="Select and show on the canvas"
                    className="-mx-1.5 block w-[calc(100%+0.75rem)] rounded px-1.5 py-1 text-left text-xs hover:bg-red-50"
                  >
                    <div className="font-medium break-words text-slate-700">{p.where}</div>
                    <div className="break-words text-red-600">{p.error}</div>
                  </button>
                </li>
              ))}
            </ul>
          </Section>
        )}

        <Section id="summary-hp" title="Hyperparameters">
          <dl className="grid grid-cols-[1fr_auto_auto] items-baseline gap-x-3 gap-y-1 text-xs">
            {HYPERPARAM_INFO.map((info) => (
              <HpRow key={info.key} name={info.key} symbol={info.symbol} value={hp[info.key].toLocaleString()} help={info.help} />
            ))}
            <HpRow name="dtype" symbol="" value={hp.dtype} help="Element type for weights and activations." />
            <HpRow name="d_head" symbol="d_model / H" value={Number.isInteger(dHead) ? String(dHead) : dHead.toFixed(2)} help="Derived: size of each attention head." />
            <HpRow name="num_layers" symbol="L" value={String(layers)} help="Derived: number of Transformer Blocks on the canvas." />
          </dl>
          <button
            type="button"
            onClick={() => setHyperparamsOpen(true)}
            className="mt-3 flex items-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1 text-xs text-slate-600 hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-700"
          >
            <SlidersHorizontal size={13} /> Edit hyperparameters
          </button>
        </Section>

        <Section id="summary-howto" title="How to use">
          <ol className="list-decimal space-y-1 pl-4 text-[13px] leading-snug text-slate-600 marker:text-slate-400">
            {HOW_TO.map((t) => (
              <li key={t} className="break-words">
                {t}
              </li>
            ))}
          </ol>
        </Section>
      </DrawerBody>
    </>
  )
}

function Stat({ label, value, title, tone }: { label: string; value: string; title?: string; tone?: 'good' | 'bad' }) {
  const color = tone === 'bad' ? 'text-red-600' : tone === 'good' ? 'text-emerald-600' : 'text-slate-800'
  return (
    <div className="rounded-md border border-slate-100 bg-slate-50 px-2.5 py-1.5" title={title}>
      <div className="text-[11px] text-slate-400">{label}</div>
      <div className={`text-base font-semibold tabular-nums ${color}`}>{value}</div>
    </div>
  )
}

function HpRow({ name, symbol, value, help }: { name: string; symbol: string; value: string; help: string }) {
  return (
    <>
      <dt className="font-mono text-slate-700" title={help}>
        {name}
      </dt>
      <span className="text-[11px] text-slate-400">{symbol}</span>
      <dd className="text-right font-mono text-slate-600 tabular-nums">{value}</dd>
    </>
  )
}
