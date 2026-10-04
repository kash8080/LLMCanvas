import { LOD_ZOOM } from '../canvas/lod'
import { ModeToggle } from '../canvas/nodes/GroupNode'
import type { GroupNode } from '../canvas/types'
import { formatCount } from '../engine/format'
import { formatConcrete, formatSymbolic } from '../engine/shape'
import type { Shape } from '../engine/types'
import { GROUP_DEFS } from '../nodes/groups'
import { useCanvasStore } from '../store/useCanvasStore'

/** Drawer content for a group frame (Phase 4 adds the full docs sections). */
export function GroupDetails({ node, Section }: { node: GroupNode; Section: (p: { title: string; children: React.ReactNode }) => React.ReactNode }) {
  const def = GROUP_DEFS[node.data.groupType]
  const summary = useCanvasStore((s) => s.inference.groups[node.id])
  const setTitle = useCanvasStore((s) => s.setTitle)
  const autoAt = Math.round((def.expandAtLod === 1 ? LOD_ZOOM.blocks : LOD_ZOOM.internals) * 100)

  return (
    <div className="flex flex-col gap-5">
      <div>
        <div className="mb-1 flex items-center gap-2 text-xs text-slate-500">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: def.color }} />
          {def.label} · group
        </div>
        <input
          value={node.data.title ?? ''}
          placeholder={def.label}
          onChange={(e) => setTitle(node.id, e.target.value)}
          className="w-full rounded border border-transparent px-1 py-0.5 text-base font-semibold text-slate-800 outline-none hover:border-slate-200 focus:border-indigo-400"
          title="Rename this group"
        />
        <p className="mt-1 px-1 text-xs leading-relaxed text-slate-500">{def.docs.overview}</p>
      </div>

      {summary?.status === 'error' && (
        <div className="rounded-md border border-red-200 bg-red-50 p-2 text-xs text-red-700">
          {summary.errors.map((e) => (
            <div key={e}>⚠ {e}</div>
          ))}
        </div>
      )}

      <Section title="Display">
        <ModeToggle id={node.id} mode={node.data.mode} className="w-fit text-xs" />
        <p className="mt-1.5 text-[11px] leading-snug text-slate-400">
          Auto opens this group when you zoom in to {autoAt}% or more. Open / Closed ignore the zoom.
        </p>
      </Section>

      <Section title="Shapes">
        <ShapeRow label="in" shape={summary?.inShape ?? null} />
        <ShapeRow label="out" shape={summary?.outShape ?? null} />
      </Section>

      <Section title="Weights">
        <div className="mb-1 text-xs font-semibold text-slate-700">
          {(summary?.params ?? 0).toLocaleString()} <span className="font-normal text-slate-400">({formatCount(summary?.params ?? 0)}) = sum of the parts inside</span>
        </div>
        {def.docs.formula && <div className="mb-1.5 font-mono text-[11px] text-slate-500">{def.docs.formula}</div>}
        <div className="flex flex-col gap-0.5 text-xs">
          {summary?.breakdown.map((b) => (
            <div key={b.id} className="flex gap-2">
              <span className="truncate text-slate-600">{b.title}</span>
              <span className="ml-auto font-mono text-slate-500 tabular-nums">{b.params > 0 ? b.params.toLocaleString() : '—'}</span>
            </div>
          ))}
        </div>
      </Section>
    </div>
  )
}

function ShapeRow({ label, shape }: { label: string; shape: Shape | null }) {
  return (
    <div className="flex items-baseline gap-2 text-xs">
      <span className="w-10 shrink-0 font-mono text-slate-600">{label}</span>
      {shape ? (
        <span className="min-w-0">
          <span className="font-mono text-slate-700">{formatSymbolic(shape)}</span>
          <span className="ml-1.5 font-mono text-slate-400">{formatConcrete(shape)}</span>
        </span>
      ) : (
        <span className="text-slate-400 italic">unknown</span>
      )}
    </div>
  )
}
