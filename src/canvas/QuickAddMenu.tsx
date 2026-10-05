import { useReactFlow } from '@xyflow/react'
import { Search } from 'lucide-react'
import { useState } from 'react'
import { nodeTitle } from '../store/inference'
import { useCanvasStore, type QuickAddState } from '../store/useCanvasStore'
import { nodePorts } from './connect'
import { filterQuickAdd, quickAddItems, quickAddPosition, type QuickAddItem } from './quickAdd'

const WIDTH = 260

/**
 * Searchable "add a part here" menu. Opened when a connection is dropped on empty canvas (the new
 * part gets connected to the dragged port) or from the canvas context menu ("Add part…").
 * Typing filters, ↑/↓ move, Enter picks, Esc / click outside cancels.
 */
export function QuickAddMenu() {
  const state = useCanvasStore((s) => s.quickAdd)
  if (!state) return null
  return <QuickAddPanel key={`${state.x},${state.y}`} state={state} />
}

function QuickAddPanel({ state }: { state: QuickAddState }) {
  const close = useCanvasStore((s) => s.closeQuickAdd)
  const addNode = useCanvasStore((s) => s.addNode)
  const fromNode = useCanvasStore((s) => (state.from ? s.nodes.find((n) => n.id === state.from!.nodeId) : undefined))
  const parentType = useCanvasStore((s) => {
    const parent = state.parentId ? s.nodes.find((n) => n.id === state.parentId) : undefined
    return parent?.type === 'group' ? parent.data.groupType : undefined
  })
  const { screenToFlowPosition, getInternalNode } = useReactFlow()
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)

  const from = state.from
  const items = filterQuickAdd(quickAddItems(from?.type ?? null, parentType), query)
  const current = Math.min(active, items.length - 1)

  const pick = (it: QuickAddItem | undefined) => {
    if (!it) return
    let pos = quickAddPosition(it.item, screenToFlowPosition({ x: state.x, y: state.y }), from?.type ?? null)
    if (state.parentId) {
      // Children are positioned relative to their group frame.
      const abs = getInternalNode(state.parentId)?.internals.positionAbsolute ?? { x: 0, y: 0 }
      pos = { x: pos.x - abs.x, y: pos.y - abs.y }
    }
    addNode(it.item, pos, { parentId: state.parentId, connectFrom: from ?? undefined })
  }

  const ports = nodePorts(fromNode)
  const portLabel = from && ports ? (from.type === 'source' ? ports.outputs : ports.inputs).find((p) => p.id === from.handleId)?.label : undefined
  const port = `${portLabel ? `“${portLabel}” of ` : ''}${fromNode ? nodeTitle(fromNode) : ''}`
  const caption = !from || !fromNode ? 'Add to the canvas' : from.type === 'source' ? `New part takes ${port} as input` : `New part’s output goes into ${port}`

  const left = Math.min(state.x, window.innerWidth - WIDTH - 8)
  const top = Math.max(8, Math.min(state.y, window.innerHeight - 420))

  return (
    <>
      <div
        className="fixed inset-0 z-40"
        onPointerDown={close}
        onWheel={close}
        onContextMenu={(e) => {
          e.preventDefault()
          close()
        }}
      />
      <div
        role="dialog"
        aria-label="Add a part"
        className="fixed z-50 flex flex-col rounded-lg border border-slate-200 bg-white text-sm shadow-xl"
        style={{ left, top, width: WIDTH }}
      >
        <div className="flex items-center gap-2 border-b border-slate-100 px-2.5 py-2">
          <Search size={14} className="shrink-0 text-slate-400" />
          <input
            autoFocus
            value={query}
            placeholder="Search parts…"
            onChange={(e) => {
              setQuery(e.target.value)
              setActive(0)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Escape') close()
              else if (e.key === 'Enter') pick(items[current])
              else if (e.key === 'ArrowDown') setActive(Math.min(current + 1, items.length - 1))
              else if (e.key === 'ArrowUp') setActive(Math.max(current - 1, 0))
              else return
              e.preventDefault()
            }}
            className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-slate-400"
          />
        </div>
        <div className="px-2.5 pt-1.5 text-[11px] leading-snug break-words text-slate-400">{caption}</div>
        <ul className="max-h-72 overflow-y-auto p-1" role="listbox">
          {items.length === 0 && <li className="px-2.5 py-2 text-xs text-slate-400">No matching part{from ? ' with a free port on that side' : ''}.</li>}
          {items.map((it, i) => (
            <li key={it.item}>
              <button
                type="button"
                role="option"
                aria-selected={i === current}
                onPointerEnter={() => setActive(i)}
                onClick={() => pick(it)}
                className={`flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left ${i === current ? 'bg-indigo-50 text-indigo-800' : 'text-slate-700'}`}
              >
                <span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: it.color }} />
                <span className="min-w-0 flex-1 truncate">{it.label}</span>
                <span className="shrink-0 truncate text-[11px] text-slate-400">{it.group}</span>
              </button>
            </li>
          ))}
        </ul>
        <div className="border-t border-slate-100 px-2.5 py-1.5 text-[11px] text-slate-400">↑↓ to move · Enter to add · Esc to cancel</div>
      </div>
    </>
  )
}
