import { formatBytes } from '../engine/format'
import { dtypeName } from '../engine/hyperparams'
import { formatConcrete, formatSymbolic, numel, shapeBytes } from '../engine/shape'
import type { Hyperparams, Shape } from '../engine/types'
import { getNodeDef } from '../nodes/registry'
import { useCanvasStore } from '../store/useCanvasStore'

const WIDTH = 250

/** Small popover shown when a port is clicked: symbolic + concrete shape, dtype, size. */
export function PortPopover() {
  const popover = useCanvasStore((s) => s.portPopover)
  const setPortPopover = useCanvasStore((s) => s.setPortPopover)
  const node = useCanvasStore((s) => (popover ? s.nodes.find((n) => n.id === popover.nodeId) : undefined))
  const result = useCanvasStore((s) => (popover ? s.inference.nodes[popover.nodeId] : undefined))
  const hp = useCanvasStore((s) => s.hyperparams)
  if (!popover || node?.type !== 'part' || !result) return null
  const def = getNodeDef(node.data.partType)
  if (!def) return null

  const ports = popover.kind === 'in' ? def.inputs : def.outputs
  const index = ports.findIndex((p) => p.id === popover.portId)
  const port = ports[index]
  const shape = (popover.kind === 'in' ? result.inputShapes : result.outputShapes)[index] ?? null
  const close = () => setPortPopover(null)

  let note = ''
  if (!shape) {
    if (popover.kind === 'in' && result.errors.some((e) => e.includes(`'${port?.label}' is not connected`))) note = 'Not connected.'
    else if (result.status === 'error') note = 'Unknown — this part has an error.'
    else note = 'Unknown — something upstream has an error.'
  }

  const left = Math.min(popover.x + 12, window.innerWidth - WIDTH - 8)
  const top = Math.min(popover.y + 12, window.innerHeight - 190)

  return (
    <>
      {/* Click / scroll anywhere else closes the popover. */}
      <div className="fixed inset-0 z-40" onPointerDown={close} onWheel={close} />
      <div
        role="dialog"
        tabIndex={-1}
        ref={(el) => el?.focus()}
        onKeyDown={(e) => e.key === 'Escape' && close()}
        className="fixed z-50 rounded-lg border border-slate-200 bg-white p-3 text-xs shadow-xl outline-none"
        style={{ left, top, width: WIDTH }}
      >
        <div className="mb-2 flex items-baseline gap-1.5">
          <span className={`rounded px-1 py-px text-[10px] font-semibold text-white ${popover.kind === 'in' ? 'bg-blue-500' : 'bg-green-600'}`}>
            {popover.kind === 'in' ? 'IN' : 'OUT'}
          </span>
          <span className="font-semibold text-slate-800">{port?.label}</span>
          <span className="truncate text-slate-400">of {node.data.title || def.label}</span>
        </div>
        {shape ? <ShapeFacts shape={shape} hp={hp} /> : <p className="text-slate-500">{note}</p>}
      </div>
    </>
  )
}

export function ShapeFacts({ shape, hp }: { shape: Shape; hp: Hyperparams }) {
  return (
    <dl className="grid grid-cols-[70px_1fr] gap-y-1">
      <dt className="text-slate-400">Symbolic</dt>
      <dd className="font-mono text-slate-700">{formatSymbolic(shape)}</dd>
      <dt className="text-slate-400">Concrete</dt>
      <dd className="font-mono text-slate-700">{formatConcrete(shape).replaceAll('×', ' × ')}</dd>
      <dt className="text-slate-400">dtype</dt>
      <dd className="font-mono text-slate-700">{dtypeName(shape.dtype, hp)}</dd>
      <dt className="text-slate-400">Size</dt>
      <dd className="text-slate-700">
        {formatBytes(shapeBytes(shape, hp))}{' '}
        <span className="text-slate-400">({numel(shape).toLocaleString()} elements)</span>
      </dd>
    </dl>
  )
}
