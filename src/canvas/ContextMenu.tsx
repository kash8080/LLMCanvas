import { useReactFlow } from '@xyflow/react'
import { BoxSelect, Copy, Maximize, PanelRight, Plus, Redo2, SquareDashedMousePointer, Trash2, Undo2 } from 'lucide-react'
import type { ReactNode } from 'react'
import { useCanvasStore } from '../store/useCanvasStore'
import { nodeTitle } from '../store/inference'
import { ModeToggle } from './nodes/GroupNode'
import { deleteLabel, isLoneProxy, useDeleteElements } from './useDelete'

/** What was right-clicked: a node, an edge, the current multi-selection, or the empty canvas. */
export type ContextTarget = { kind: 'node'; id: string } | { kind: 'edge'; id: string } | { kind: 'selection' } | { kind: 'pane' }
export interface ContextMenuState {
  /** Screen (client) position. */
  x: number
  y: number
  target: ContextTarget
}

const WIDTH = 248

/** Right-click menu on the canvas. Closes on Esc, click outside, scroll, or after picking an item. */
export function ContextMenu({ menu, onClose }: { menu: ContextMenuState; onClose: () => void }) {
  const left = Math.min(menu.x, window.innerWidth - WIDTH - 8)
  const top = Math.min(menu.y, window.innerHeight - 260)
  return (
    <>
      <div
        className="fixed inset-0 z-40"
        onPointerDown={onClose}
        onWheel={onClose}
        onContextMenu={(e) => {
          e.preventDefault()
          onClose()
        }}
      />
      <div
        role="menu"
        tabIndex={-1}
        ref={(el) => el?.focus()}
        onKeyDown={(e) => e.key === 'Escape' && onClose()}
        onContextMenu={(e) => e.preventDefault()}
        className="fixed z-50 rounded-lg border border-slate-200 bg-white p-1 text-sm shadow-xl outline-none"
        style={{ left, top, width: WIDTH }}
      >
        <MenuBody menu={menu} close={onClose} />
      </div>
    </>
  )
}

function MenuBody({ menu, close }: { menu: ContextMenuState; close: () => void }) {
  const nodes = useCanvasStore((s) => s.nodes)
  const edges = useCanvasStore((s) => s.edges)
  const canUndo = useCanvasStore((s) => s.history.past.length > 0)
  const canRedo = useCanvasStore((s) => s.history.future.length > 0)
  const { duplicateSelection, selectNodes, selectOnly, setDrawerOpen, openQuickAdd, undo, redo } = useCanvasStore.getState()
  const deleteElements = useDeleteElements()
  const { fitView } = useReactFlow()
  const run = (fn: () => void) => () => {
    close()
    fn()
  }
  const { target } = menu

  if (target.kind === 'pane') {
    return (
      <>
        <Item icon={Plus} label="Add part here…" onClick={run(() => openQuickAdd({ x: menu.x, y: menu.y, from: null }))} />
        <Item icon={SquareDashedMousePointer} label="Select all" onClick={run(() => selectNodes(nodes.filter((n) => !n.parentId).map((n) => n.id)))} />
        <Item icon={Maximize} label="Fit view" onClick={run(() => void fitView({ padding: 0.1, duration: 300 }))} />
        <Divider />
        <Item icon={Undo2} label="Undo" hint="⌘Z" disabled={!canUndo} onClick={run(undo)} />
        <Item icon={Redo2} label="Redo" hint="⇧⌘Z" disabled={!canRedo} onClick={run(redo)} />
      </>
    )
  }

  if (target.kind === 'edge') {
    const edge = edges.find((e) => e.id === target.id)
    const src = nodes.find((n) => n.id === edge?.source)
    const tgt = nodes.find((n) => n.id === edge?.target)
    return (
      <>
        {src && tgt && <Caption>{`${nodeTitle(src)} → ${nodeTitle(tgt)}`}</Caption>}
        <Item icon={Trash2} label="Delete connection" hint="⌫" danger onClick={run(() => deleteElements([], [target.id]))} />
      </>
    )
  }

  if (target.kind === 'selection') {
    const selNodes = nodes.filter((n) => n.selected)
    const selEdges = edges.filter((e) => e.selected)
    const count = selNodes.length + selEdges.length
    return (
      <>
        <Caption>{`${count} items selected`}</Caption>
        {selNodes.length > 0 && <Item icon={Copy} label={`Duplicate ${selNodes.length} items`} hint="⌘D" onClick={run(duplicateSelection)} />}
        <Item
          icon={Trash2}
          label={`Delete ${count} items`}
          hint="⌫"
          danger
          onClick={run(() => deleteElements(selNodes.map((n) => n.id), selEdges.map((e) => e.id)))}
        />
      </>
    )
  }

  const node = nodes.find((n) => n.id === target.id)
  if (!node) return <Caption>Nothing here</Caption>
  const details = <Item icon={PanelRight} label="Show details" onClick={run(() => (selectOnly(node.id), setDrawerOpen(true)))} />

  if (isLoneProxy(node)) {
    return (
      <>
        <Caption>Group {node.id.endsWith('.in') ? 'input' : 'output'} pill</Caption>
        <Item icon={BoxSelect} label="Select its group" onClick={run(() => selectOnly(node.parentId!))} />
        <Item icon={Trash2} label="Delete" disabled hint="with its group" onClick={() => {}} />
      </>
    )
  }

  return (
    <>
      <Caption>{nodeTitle(node)}</Caption>
      {node.type === 'group' && (
        <div className="flex items-center justify-between gap-2 px-2.5 py-1.5 text-slate-600">
          <span className="text-xs">Display</span>
          <ModeToggle id={node.id} mode={node.data.mode} className="text-xs" />
        </div>
      )}
      {details}
      <Item icon={Copy} label="Duplicate" hint="⌘D" onClick={run(() => (selectOnly(node.id, false), duplicateSelection()))} />
      <Divider />
      <Item icon={Trash2} label={deleteLabel(node, nodes)} hint="⌫" danger onClick={run(() => deleteElements([node.id]))} />
    </>
  )
}

function Item({
  icon: Icon,
  label,
  hint,
  danger,
  disabled,
  onClick,
}: {
  icon: typeof Trash2
  label: string
  hint?: string
  danger?: boolean
  disabled?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      role="menuitem"
      disabled={disabled}
      onClick={onClick}
      className={`flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left disabled:cursor-default disabled:opacity-40 ${
        danger ? 'text-red-600 hover:bg-red-50 disabled:hover:bg-transparent' : 'text-slate-700 hover:bg-slate-100 disabled:hover:bg-transparent'
      }`}
    >
      <Icon size={15} className="shrink-0" />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {hint && <span className="shrink-0 text-xs text-slate-400">{hint}</span>}
    </button>
  )
}

function Caption({ children }: { children: ReactNode }) {
  return <div className="truncate px-2.5 pt-1 pb-1.5 text-[11px] font-medium text-slate-400">{children}</div>
}

function Divider() {
  return <div className="my-1 h-px bg-slate-100" />
}
