import { X } from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import type { AnnotationData, AppNode, StickyNode, TextBoxNode } from '../canvas/types'
import { useCanvasStore } from '../store/useCanvasStore'
import { GroupDetails } from './GroupDetails'
import { PartDetails } from './PartDetails'

const BG_SWATCHES = ['transparent', '#ffffff', '#fef08a', '#fed7aa', '#fecdd3', '#e9d5ff', '#bfdbfe', '#bbf7d0', '#e2e8f0']
const TEXT_SWATCHES = ['#111827', '#1f2937', '#64748b', '#dc2626', '#2563eb', '#16a34a', '#ffffff']

/** Right-hand drawer. Phase 2: minimal part editor; Phase 4 adds the full sections (Overview, Size, …). */
export function DetailDrawer() {
  const open = useCanvasStore((s) => s.drawerOpen)
  const setOpen = useCanvasStore((s) => s.setDrawerOpen)
  const selected = useCanvasStore(useShallow((s) => s.nodes.filter((n) => n.selected)))

  if (!open) return null

  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-slate-200 bg-white">
      <div className="flex h-11 items-center border-b border-slate-200 px-4">
        <h2 className="text-sm font-semibold text-slate-700">Details</h2>
        <button
          type="button"
          onClick={() => setOpen(false)}
          title="Close"
          className="ml-auto rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
        >
          <X size={16} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 text-sm">
        {selected.length === 0 && <p className="text-slate-400">Select something on the canvas to see its details.</p>}
        {selected.length > 1 && <p className="text-slate-500">{selected.length} items selected.</p>}
        {selected.length === 1 && <NodeDetails node={selected[0]} />}
      </div>
    </aside>
  )
}

function NodeDetails({ node }: { node: AppNode }) {
  if (node.type === 'part') return <PartDetails node={node} Section={Section} />
  if (node.type === 'group') return <GroupDetails node={node} Section={Section} />
  return (
    <div className="flex flex-col gap-5">
      <Section title="Element">
        <dl className="grid grid-cols-[64px_1fr] gap-y-1 text-xs">
          <dt className="text-slate-400">Type</dt>
          <dd className="font-medium text-slate-700">{node.type}</dd>
          <dt className="text-slate-400">ID</dt>
          <dd className="truncate font-mono text-slate-500" title={node.id}>
            {node.id}
          </dd>
        </dl>
      </Section>
      {(node.type === 'sticky' || node.type === 'textbox') && <AnnotationSettings node={node} />}
    </div>
  )
}

function AnnotationSettings({ node }: { node: StickyNode | TextBoxNode }) {
  const updateAnnotation = useCanvasStore((s) => s.updateAnnotation)
  const update = (patch: Partial<AnnotationData>) => updateAnnotation(node.id, patch)
  const { bgColor, textColor, fontSize } = node.data

  return (
    <>
      <Section title="Background">
        <Swatches colors={BG_SWATCHES} value={bgColor} onChange={(c) => update({ bgColor: c })} />
      </Section>
      <Section title="Text color">
        <Swatches colors={TEXT_SWATCHES} value={textColor} onChange={(c) => update({ textColor: c })} />
      </Section>
      <Section title="Font size">
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={10}
            max={48}
            value={fontSize}
            onChange={(e) => update({ fontSize: Number(e.target.value) })}
            className="flex-1 accent-indigo-600"
          />
          <span className="w-10 text-right text-xs text-slate-500 tabular-nums">{fontSize}px</span>
        </div>
      </Section>
      <p className="text-xs text-slate-400">Double-click the element on the canvas to edit its text.</p>
    </>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3 className="mb-2 text-[11px] font-semibold tracking-wide text-slate-400 uppercase">{title}</h3>
      {children}
    </section>
  )
}

function Swatches({ colors, value, onChange }: { colors: string[]; value: string; onChange: (c: string) => void }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {colors.map((c) => (
        <button
          key={c}
          type="button"
          title={c}
          onClick={() => onChange(c)}
          className={`h-6 w-6 rounded-full border transition ${
            value === c ? 'border-indigo-500 ring-2 ring-indigo-200' : 'border-slate-300 hover:scale-110'
          } ${c === 'transparent' ? 'swatch-transparent' : ''}`}
          style={c === 'transparent' ? undefined : { backgroundColor: c }}
        />
      ))}
    </div>
  )
}
