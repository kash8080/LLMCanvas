import { Link2, Unlink } from 'lucide-react'
import type { PartNode } from '../../canvas/types'
import { bindSymbol } from '../../engine/hyperparams'
import { isBound } from '../../engine/resolve'
import type { ParamSchema, ParamValue, ResolvedParam } from '../../engine/types'
import { CATEGORY_INFO, getNodeDef } from '../../nodes/registry'
import { useCanvasStore } from '../../store/useCanvasStore'
import { NumberField } from '../NumberField'
import { FormulaSection, OverviewSection, PointsSection, ReferenceSection } from './DocsSections'
import { ShapesSection } from './ShapesSection'
import { PartSizeSection } from './SizeSection'
import { DrawerBody, DrawerHeader, ErrorBox, Notice, Section, StatusBadge } from './ui'

/** Drawer for a model part: header, errors, Overview, Formula, Parameters, Shapes, Size, Points, CS336 ref. */
export function PartDetails({ node }: { node: PartNode }) {
  const def = getNodeDef(node.data.partType)
  const result = useCanvasStore((s) => s.inference.nodes[node.id])
  const setTitle = useCanvasStore((s) => s.setTitle)
  if (!def || !result) return <p className="p-4 text-sm text-red-600">Unknown part type “{node.data.partType}”.</p>

  const cat = CATEGORY_INFO[def.category]
  const { docs } = def

  return (
    <>
      <DrawerHeader
        color={cat.color}
        title={node.data.title ?? ''}
        placeholder={def.label}
        onRename={(t) => setTitle(node.id, t)}
        subtitle={`${def.label} · ${cat.label}`}
        badge={<StatusBadge status={result.status} errorCount={result.errors.length} />}
      />
      <DrawerBody>
        {result.status === 'error' && <ErrorBox errors={result.errors} />}
        {result.status === 'unknown' && <Notice>Shape unknown: something upstream has an error, so this part can’t be checked yet.</Notice>}

        <OverviewSection docs={docs} title={node.data.title} />
        <FormulaSection lines={docs.formula} />

        {def.params.length > 0 && (
          <Section id="params" title="Parameters">
            <div className="space-y-3">
              {def.params.map((schema) => (
                <ParamRow
                  key={schema.key}
                  nodeId={node.id}
                  schema={schema}
                  help={docs.paramHelp?.[schema.key] ?? schema.help}
                  value={node.data.params[schema.key] ?? schema.default}
                  resolved={result.resolved[schema.key]}
                />
              ))}
            </div>
            {def.params.some((p) => isBound(p.default)) && (
              <p className="mt-3 text-[11px] leading-snug text-slate-400">🔗 = follows the global hyperparameter. Unlink to set a value for this part only.</p>
            )}
          </Section>
        )}

        <ShapesSection
          inputs={def.inputs.map((p, i) => ({ port: p.label, shape: result.inputShapes[i] }))}
          outputs={def.outputs.map((p, i) => ({ port: p.label, shape: result.outputShapes[i] }))}
          unknownOut={result.status === 'error' ? 'not computed — fix the problem above' : undefined}
        />
        <PartSizeSection nodeId={node.id} title={node.data.title} count={result.paramCount} />
        <PointsSection points={docs.pointsToRemember} />
        <ReferenceSection docs={docs} />
      </DrawerBody>
    </>
  )
}

function ParamRow({ nodeId, schema, help, value, resolved }: { nodeId: string; schema: ParamSchema; help: string; value: ParamValue; resolved: ResolvedParam }) {
  const setPartParam = useCanvasStore((s) => s.setPartParam)
  const set = (v: ParamValue) => setPartParam(nodeId, schema.key, v)
  const defaultBind = isBound(schema.default) ? schema.default.bind : null

  return (
    <div>
      <div className="flex items-center gap-2">
        <span className="min-w-0 truncate font-mono text-xs font-medium text-slate-700" title={schema.label}>
          {schema.label}
        </span>
        <div className="ml-auto flex shrink-0 items-center gap-1">
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
              {defaultBind ? (
                <IconButton title={`Relink to the global ${defaultBind}`} onClick={() => set({ bind: defaultBind })}>
                  <Link2 size={13} />
                </IconButton>
              ) : (
                <span className="w-[21px]" />
              )}
            </>
          )}
        </div>
      </div>
      <p className="mt-0.5 text-[11px] leading-snug break-words text-slate-400">{help}</p>
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
