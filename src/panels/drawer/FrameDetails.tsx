import { Trash2 } from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import { itemsInFrame } from '../../canvas/frames'
import type { FrameNode } from '../../canvas/types'
import { useDeleteElements } from '../../canvas/useDelete'
import { nodeTitle } from '../../store/inference'
import { useCanvasStore } from '../../store/useCanvasStore'
import { Swatches } from './AnnotationDetails'
import { DrawerBody, DrawerHeader, Section, SubHeading } from './ui'
import { useFocusNode } from './useFocusNode'

/** Soft background tints (Tailwind *-50) and matching borders (*-400). */
const BG_SWATCHES = ['transparent', '#ffffff', '#f8fafc', '#eff6ff', '#f0fdf4', '#fefce8', '#fff7ed', '#fdf2f8', '#f5f3ff']
const BORDER_SWATCHES = ['#cbd5e1', '#94a3b8', '#60a5fa', '#4ade80', '#facc15', '#fb923c', '#f472b6', '#a78bfa', '#475569']

/** Drawer for a user-made frame: title, colours, what's inside, delete (frame only, or with its contents). */
export function FrameDetails({ node }: { node: FrameNode }) {
  const setTitle = useCanvasStore((s) => s.setTitle)
  const updateFrame = useCanvasStore((s) => s.updateFrame)
  const inside = useCanvasStore(useShallow((s) => itemsInFrame(s.nodes, node.id)))
  const nodes = useCanvasStore((s) => s.nodes)
  const deleteElements = useDeleteElements()
  const focus = useFocusNode()
  const byId = new Map(nodes.map((n) => [n.id, n]))

  return (
    <>
      <DrawerHeader
        color={node.data.borderColor}
        title={node.data.title ?? ''}
        placeholder="Frame"
        onRename={(t) => setTitle(node.id, t)}
        subtitle="Frame · visual only"
        onDelete={() => deleteElements([node.id])}
        deleteTitle="Delete the frame — what's inside stays (⌫)"
      />
      <DrawerBody>
        <Section id="appearance" title="Appearance">
          <SubHeading>Background</SubHeading>
          <Swatches colors={BG_SWATCHES} value={node.data.bgColor} onChange={(c) => updateFrame(node.id, { bgColor: c })} />
          <SubHeading>Border</SubHeading>
          <Swatches colors={BORDER_SWATCHES} value={node.data.borderColor} onChange={(c) => updateFrame(node.id, { borderColor: c })} />
        </Section>
        <Section id="frame-inside" title="Inside" meta={inside.length}>
          {inside.length === 0 ? (
            <p className="text-xs text-slate-400">Nothing is fully inside this frame yet. Drag items into it, or right-click a selection → Frame selection.</p>
          ) : (
            <>
              <ul className="space-y-0.5">
                {inside.map((id) => {
                  const n = byId.get(id)
                  if (!n) return null
                  return (
                    <li key={id}>
                      <button
                        type="button"
                        onClick={() => focus(id)}
                        className="-mx-1.5 flex w-[calc(100%+0.75rem)] items-baseline gap-2 rounded px-1.5 py-1 text-left text-xs hover:bg-slate-50"
                      >
                        <span className="min-w-0 truncate text-slate-700">{nodeTitle(n)}</span>
                        <span className="ml-auto shrink-0 text-slate-400">{n.type === 'part' ? 'part' : n.type}</span>
                      </button>
                    </li>
                  )
                })}
              </ul>
              <button
                type="button"
                onClick={() => deleteElements([node.id, ...inside])}
                className="mt-2 flex items-center gap-1.5 rounded px-1.5 py-1 text-xs text-red-600 hover:bg-red-50"
              >
                <Trash2 size={13} /> Delete frame and its {inside.length} item{inside.length > 1 ? 's' : ''}
              </button>
            </>
          )}
        </Section>
        <p className="px-4 pt-3 text-xs leading-relaxed text-slate-400">
          A frame only labels a region: it has no ports and doesn't change shapes, parameters or memory. Dragging it moves
          everything fully inside it. Double-click its title to rename; drag its corners (when selected) to resize. Delete
          removes just the frame.
        </p>
      </DrawerBody>
    </>
  )
}
