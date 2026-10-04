import { Box, StickyNote, Type, type LucideIcon } from 'lucide-react'
import { DND_MIME, type NodeKind } from '../canvas/types'

interface PaletteItem {
  kind: NodeKind
  label: string
  hint: string
  icon: LucideIcon
}

interface PaletteSection {
  title: string
  items: PaletteItem[]
}

// Phase 2 adds the model parts (Embedding, RMSNorm, Linear, …) from the node registry.
const SECTIONS: PaletteSection[] = [
  {
    title: 'Parts',
    items: [{ kind: 'placeholder', label: 'Placeholder part', hint: '1 input, 1 output', icon: Box }],
  },
  {
    title: 'Annotate',
    items: [
      { kind: 'sticky', label: 'Sticky note', hint: 'Coloured note', icon: StickyNote },
      { kind: 'textbox', label: 'Text box', hint: 'Free text', icon: Type },
    ],
  },
]

export function Palette() {
  return (
    <aside className="flex w-56 shrink-0 flex-col gap-4 overflow-y-auto border-r border-slate-200 bg-white p-3">
      {SECTIONS.map((section) => (
        <div key={section.title}>
          <div className="mb-1.5 px-1 text-[11px] font-semibold tracking-wide text-slate-400 uppercase">{section.title}</div>
          <div className="flex flex-col gap-1.5">
            {section.items.map((item) => (
              <div
                key={item.kind}
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData(DND_MIME, item.kind)
                  e.dataTransfer.effectAllowed = 'move'
                }}
                className="flex cursor-grab items-center gap-2.5 rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-sm shadow-xs transition hover:border-indigo-300 hover:bg-indigo-50/40 active:cursor-grabbing"
                title="Drag onto the canvas"
              >
                <item.icon size={16} className="text-slate-500" />
                <div className="leading-tight">
                  <div className="font-medium text-slate-700">{item.label}</div>
                  <div className="text-[11px] text-slate-400">{item.hint}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
      <p className="mt-auto px-1 text-[11px] leading-snug text-slate-400">
        Drag items onto the canvas. Shift+drag to box-select, ⌘D to duplicate, Delete to remove.
      </p>
    </aside>
  )
}
