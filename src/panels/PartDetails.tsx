import { Link2, Unlink } from 'lucide-react'
import type { PartNode } from '../canvas/types'
import { formatCount } from '../engine/format'
import { bindSymbol } from '../engine/hyperparams'
import { isBound } from '../engine/resolve'
import { formatConcrete, formatSymbolic } from '../engine/shape'
import type { NodeDef, ParamSchema, ParamValue, PortDef, ResolvedParam, Shape } from '../engine/types'
import { CATEGORY_INFO, getNodeDef } from '../nodes/registry'
import { useCanvasStore } from '../store/useCanvasStore'
import { NumberField } from './NumberField'

/** Minimal drawer content for a model part (Phase 4 turns this into full sections). */
export function PartDetails({ node, Section }: { node: PartNode; Section: (p: { title: string; children: React.ReactNode }) => React.ReactNode }) {
  const def = getNodeDef(node.data.partType)
  const result = useCanvasStore((s) => s.inference.nodes[node.id])
  const setTitle = useCanvasStore((s) => s.setTitle)
  if (!def || !result) return <p className="text-red-600">Unknown part type “{node.data.partType}”.</p>

  const cat = CATEGORY_INFO[def.category]

  return (
    <div className="flex flex-col gap-5">
      <div>
        <div className="mb-1 flex items-center gap-2 text-xs text-slate-500">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: cat.color }} />
          {def.label} · {cat.label}
        </div>
        <input
          value={node.data.title ?? ''}
          placeholder={def.label}
          onChange={(e) => setTitle(node.id, e.target.value)}
          className="w-full rounded border border-transparent px-1 py-0.5 text-base font-semibold text-slate-800 outline-none hover:border-slate-200 focus:border-indigo-400"
          title="Rename this part"
        />
        <p className="mt-1 px-1 text-xs leading-relaxed text-slate-500">{def.docs.overview}</p>
      </div>

      {result.status === 'error' && (
        <div className="rounded-md border border-red-200 bg-red-50 p-2 text-xs text-red-700">
          {result.errors.map((e) => (
            <div key={e}>⚠ {e}</div>
          ))}
        </div>
      )}
      {result.status === 'unknown' && (
        <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 p-2 text-xs text-slate-500">
          Shape unknown: something upstream has an error.
        </div>
      )}

      {def.params.length > 0 && (
        <Section title="Parameters">
          <div className="flex flex-col gap-2">
            {def.params.map((schema) => (
              <ParamRow
                key={schema.key}
                nodeId={node.id}
                schema={schema}
                value={node.data.params[schema.key] ?? schema.default}
                resolved={result.resolved[schema.key]}
              />
            ))}
          </div>
        </Section>
      )}

      <Section title="Shapes">
        <ShapeList title="In" ports={def.inputs} shapes={result.inputShapes} />
        <ShapeList title="Out" ports={def.outputs} shapes={result.outputShapes} />
      </Section>

      <Section title="Weights">
        <ParamCount def={def} count={result.paramCount} />
      </Section>
    </div>
  )
}

function ParamRow({ nodeId, schema, value, resolved }: { nodeId: string; schema: ParamSchema; value: ParamValue; resolved: ResolvedParam }) {
  const setPartParam = useCanvasStore((s) => s.setPartParam)
  const set = (v: ParamValue) => setPartParam(nodeId, schema.key, v)
  const defaultBind = isBound(schema.default) ? schema.default.bind : null

  return (
    <div className="grid grid-cols-[1fr_auto] items-center gap-x-2" title={schema.help}>
      <span className="truncate font-mono text-xs text-slate-700">{schema.label}</span>
      <div className="flex items-center gap-1">
        {isBound(value) ? (
          <>
            <span className="rounded bg-indigo-50 px-1.5 py-0.5 font-mono text-xs text-indigo-700" title={`Follows the global ${value.bind}`}>
              🔗 {bindSymbol(value.bind)} = {String(resolved.value)}
            </span>
            <IconButton title="Unlink: set a local value for this part only" onClick={() => set({ value: resolved.value })}>
              <Unlink size={13} />
            </IconButton>
          </>
        ) : (
          <>
            <LocalEditor key={`${nodeId}:${schema.key}`} schema={schema} value={value.value} onChange={(v) => set({ value: v })} />
            {defaultBind && (
              <IconButton title={`Relink to the global ${defaultBind}`} onClick={() => set({ bind: defaultBind })}>
                <Link2 size={13} />
              </IconButton>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function LocalEditor({ schema, value, onChange }: { schema: ParamSchema; value: number | boolean | string; onChange: (v: number | boolean | string) => void }) {
  if (schema.kind === 'bool')
    return <input type="checkbox" checked={value === true} onChange={(e) => onChange(e.target.checked)} className="h-4 w-4 accent-indigo-600" />
  if (schema.kind === 'enum')
    return (
      <select value={String(value)} onChange={(e) => onChange(e.target.value)} className="rounded border border-slate-300 px-1 py-0.5 text-xs">
        {schema.options?.map((o) => (
          <option key={o}>{o}</option>
        ))}
      </select>
    )
  return <NumberField value={Number(value)} integer={schema.kind === 'int'} min={schema.min} onCommit={onChange} className="w-24" />
}

function IconButton({ title, onClick, children }: { title: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" title={title} onClick={onClick} className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700">
      {children}
    </button>
  )
}

function ShapeList({ title, ports, shapes }: { title: string; ports: PortDef[]; shapes: (Shape | null)[] }) {
  if (ports.length === 0) return null
  return (
    <div className="mb-2">
      <div className="mb-0.5 text-[11px] text-slate-400">{title}</div>
      {ports.map((p, i) => {
        const s = shapes[i]
        return (
          <div key={p.id} className="flex items-baseline gap-2 text-xs">
            <span className="w-16 shrink-0 truncate font-mono text-slate-600">{p.label}</span>
            {s ? (
              <span className="min-w-0">
                <span className="font-mono text-slate-700">{formatSymbolic(s)}</span>
                <span className="ml-1.5 font-mono text-slate-400">{formatConcrete(s)}</span>
              </span>
            ) : (
              <span className="text-slate-400 italic">unknown</span>
            )}
          </div>
        )
      })}
    </div>
  )
}

function ParamCount({ def, count }: { def: NodeDef; count: { total: number; tensors: { name: string; dims: { size: number; label?: string }[] }[] } }) {
  if (count.total === 0)
    return <p className="text-xs text-slate-400">No learned parameters{def.type === 'rope' ? ' (cos/sin tables are buffers)' : ''}.</p>
  return (
    <div className="text-xs">
      <div className="mb-1 font-semibold text-slate-700">
        {count.total.toLocaleString()} <span className="font-normal text-slate-400">({formatCount(count.total)})</span>
      </div>
      {count.tensors.map((t) => (
        <div key={t.name} className="font-mono text-slate-500">
          {t.name}: {t.dims.map((d) => d.label ?? d.size).join(' × ')} = {t.dims.map((d) => d.size).join(' × ')}
        </div>
      ))}
    </div>
  )
}
