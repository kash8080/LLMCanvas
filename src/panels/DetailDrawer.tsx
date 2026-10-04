import { useShallow } from 'zustand/react/shallow'
import type { AppNode } from '../canvas/types'
import { nodeTitle } from '../store/inference'
import { useCanvasStore } from '../store/useCanvasStore'
import { AnnotationDetails } from './drawer/AnnotationDetails'
import { GroupDetails } from './drawer/GroupDetails'
import { ModelSummary } from './drawer/ModelSummary'
import { PartDetails } from './drawer/PartDetails'
import { DrawerBody, DrawerHeader, Section } from './drawer/ui'
import { useFocusNode } from './drawer/useFocusNode'

/**
 * Right-hand drawer (R4). Closable; clicking a node reopens it.
 * Nothing selected → model summary; one node → its sectioned details; several → a list.
 */
export function DetailDrawer() {
  const open = useCanvasStore((s) => s.drawerOpen)
  const selected = useCanvasStore(useShallow((s) => s.nodes.filter((n) => n.selected)))

  if (!open) return null

  return (
    <aside className="flex w-[340px] shrink-0 flex-col xl:w-[380px] border-l border-slate-200 bg-white">
      {selected.length === 0 && <ModelSummary />}
      {selected.length === 1 && <NodeDetails key={selected[0].id} node={selected[0]} />}
      {selected.length > 1 && <MultiSelection nodes={selected} />}
    </aside>
  )
}

function NodeDetails({ node }: { node: AppNode }) {
  if (node.type === 'part') return <PartDetails node={node} />
  if (node.type === 'group') return <GroupDetails node={node} />
  return <AnnotationDetails node={node} />
}

function MultiSelection({ nodes }: { nodes: AppNode[] }) {
  const focus = useFocusNode()
  return (
    <>
      <DrawerHeader color="#94a3b8" title={`${nodes.length} items selected`} subtitle="⌘D duplicates them, Delete removes them" />
      <DrawerBody>
        <Section id="selection" title="Selection">
          <ul className="space-y-0.5">
            {nodes.map((n) => (
              <li key={n.id}>
                <button
                  type="button"
                  onClick={() => focus(n.id)}
                  className="-mx-1.5 flex w-[calc(100%+0.75rem)] items-baseline gap-2 rounded px-1.5 py-1 text-left text-xs hover:bg-slate-50"
                >
                  <span className="min-w-0 truncate text-slate-700">{nodeTitle(n)}</span>
                  <span className="ml-auto shrink-0 text-slate-400">{n.type === 'part' ? 'part' : n.type}</span>
                </button>
              </li>
            ))}
          </ul>
        </Section>
      </DrawerBody>
    </>
  )
}
