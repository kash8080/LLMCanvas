import { StickyNote, Type } from 'lucide-react'
import { DND_MIME, type PaletteItemId } from '../canvas/types'
import { GROUP_DEF_LIST } from '../nodes/groups'
import { CATEGORY_INFO, CATEGORY_ORDER, NODE_DEFS } from '../nodes/registry'

function PaletteItem({ item, label, title, children }: { item: PaletteItemId; label: string; title: string; children: React.ReactNode }) {
  return (
    <div
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(DND_MIME, item)
        e.dataTransfer.effectAllowed = 'move'
      }}
      className="flex cursor-grab items-center gap-2 rounded-md border border-slate-200 bg-white px-2 py-1.5 text-[13px] shadow-xs transition hover:border-indigo-300 hover:bg-indigo-50/40 active:cursor-grabbing"
      title={title}
    >
      {children}
      <span className="truncate font-medium text-slate-700">{label}</span>
    </div>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <div className="mb-1 px-1 text-[11px] font-semibold tracking-wide text-slate-400 uppercase">{children}</div>
}

/** Left palette: model parts from the registry grouped by category, plus annotations. */
export function Palette() {
  return (
    <aside className="flex w-56 shrink-0 flex-col gap-3 overflow-y-auto border-r border-slate-200 bg-white p-3">
      <div>
        <SectionTitle>Groups</SectionTitle>
        <div className="flex flex-col gap-1">
          {GROUP_DEF_LIST.map((g) => (
            <PaletteItem key={g.type} item={`group:${g.type}`} label={g.label} title={`${g.docs.overview}\n\nDrag onto the canvas: adds the whole group with its parts.`}>
              <span className="h-3 w-3 shrink-0 rounded-sm border-2" style={{ borderColor: g.color, background: `${g.color}33` }} />
            </PaletteItem>
          ))}
        </div>
      </div>
      {CATEGORY_ORDER.map((cat) => {
        const defs = NODE_DEFS.filter((d) => d.category === cat)
        if (defs.length === 0) return null
        return (
          <div key={cat}>
            <SectionTitle>{CATEGORY_INFO[cat].label}</SectionTitle>
            <div className="flex flex-col gap-1">
              {defs.map((d) => (
                <PaletteItem key={d.type} item={`part:${d.type}`} label={d.label} title={`${d.docs.overview}\n\nDrag onto the canvas`}>
                  <span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: CATEGORY_INFO[cat].color }} />
                </PaletteItem>
              ))}
            </div>
          </div>
        )
      })}
      <div>
        <SectionTitle>Annotations</SectionTitle>
        <div className="flex flex-col gap-1">
          <PaletteItem item="sticky" label="Sticky note" title="Coloured note — drag onto the canvas">
            <StickyNote size={14} className="text-slate-500" />
          </PaletteItem>
          <PaletteItem item="textbox" label="Text box" title="Free text — drag onto the canvas">
            <Type size={14} className="text-slate-500" />
          </PaletteItem>
        </div>
      </div>
      <p className="mt-auto px-1 text-[11px] leading-snug text-slate-400">
        Drag items onto the canvas, or drop a connection on empty canvas to add a connected part. Shift+drag to box-select,
        ⌘D to duplicate, Delete to remove, ⌘Z to undo, right-click for more.
      </p>
    </aside>
  )
}
