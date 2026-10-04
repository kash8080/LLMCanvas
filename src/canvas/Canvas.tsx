import {
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  useReactFlow,
  useStore,
  type DefaultEdgeOptions,
  type IsValidConnection,
  type OnConnectEnd,
} from '@xyflow/react'
import { X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { formatBytes, formatCount } from '../engine/format'
import { isMemKey, type MemoryCategory } from '../engine/memory'
import { GROUP_DEFS } from '../nodes/groups'
import { CATEGORY_INFO, getNodeDef, highlightInfo } from '../nodes/registry'
import { selectMemory, useCanvasStore, withoutLoneProxies, type CanvasState } from '../store/useCanvasStore'
import { connectionProblem, connectionToBody } from './connect'
import { ContextMenu, type ContextMenuState } from './ContextMenu'
import { paletteItemSize } from './groupTemplates'
import { applyLod, hideEdgesOfHiddenNodes, lodSelector } from './lod'
import { isPaletteItemId, topNodes } from './nodeFactory'
import { edgeTypes, nodeTypes } from './nodeTypes'
import { PortPopover } from './PortPopover'
import { QuickAddMenu } from './QuickAddMenu'
import { DND_MIME, type AppEdge, type AppNode } from './types'

const defaultEdgeOptions: DefaultEdgeOptions = {
  markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: '#94a3b8' },
  style: { stroke: '#94a3b8', strokeWidth: 1.5 },
}

// output → input only, an occupied input gets replaced (see store.onConnect), no cycles.
const isValidConnection: IsValidConnection<AppEdge> = (c) => {
  const { nodes, edges } = useCanvasStore.getState()
  return connectionProblem(nodes, edges, c) === null
}

// Refused drops on a port explain why; drops on a node body connect to its first free input;
// drops on empty canvas open the quick-add menu (create a part there, already connected).
const onConnectEnd: OnConnectEnd = (event, state) => {
  if (state.isValid || !state.fromNode || !state.fromHandle) return
  const { nodes, edges, onConnect, showHint, openQuickAdd } = useCanvasStore.getState()
  const from = { nodeId: state.fromNode.id, handleId: state.fromHandle.id ?? null, type: state.fromHandle.type }

  // Snapped to a port of the opposite kind (output → input) but refused: say why.
  // (Snapping to a same-kind port, e.g. the target's own output, is treated as a body drop below.)
  if (state.toNode && state.toHandle && state.toHandle.type !== from.type) {
    const c =
      from.type === 'source'
        ? { source: from.nodeId, sourceHandle: from.handleId, target: state.toNode.id, targetHandle: state.toHandle.id ?? null }
        : { source: state.toNode.id, sourceHandle: state.toHandle.id ?? null, target: from.nodeId, targetHandle: from.handleId }
    showHint(connectionProblem(nodes, edges, c) ?? 'Connect an output (green, bottom) to an input (blue, top).')
    return
  }

  const point = 'changedTouches' in event ? event.changedTouches[0] : event
  const nodeEl = document.elementFromPoint(point.clientX, point.clientY)?.closest('.react-flow__node')
  const targetId = nodeEl?.getAttribute('data-id') ?? state.toNode?.id
  const fromParent = nodes.find((n) => n.id === from.nodeId)?.parentId
  const at = { x: point.clientX, y: point.clientY }
  if (!targetId) {
    // Empty canvas. A part inside a group can only connect inside it, so the new part can't go out here.
    if (fromParent) showHint('Parts inside a group only connect to each other — let go inside the group to add a part there.')
    else openQuickAdd({ ...at, from })
    return
  }
  if (targetId === from.nodeId) return
  // Let go on the empty area of the group the drag started in: add a part inside that group.
  if (fromParent === targetId) {
    openQuickAdd({ ...at, from, parentId: targetId })
    return
  }
  const r = connectionToBody(nodes, edges, from, targetId)
  if ('connection' in r) onConnect({ ...r.connection, sourceHandle: r.connection.sourceHandle ?? null, targetHandle: r.connection.targetHandle ?? null })
  else showHint(r.problem)
}

function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)
}

export function Canvas() {
  const nodes = useCanvasStore((s) => s.nodes)
  const edges = useCanvasStore((s) => s.edges)
  const hint = useCanvasStore((s) => s.hint)
  const onNodesChange = useCanvasStore((s) => s.onNodesChange)
  const onEdgesChange = useCanvasStore((s) => s.onEdgesChange)
  const onConnect = useCanvasStore((s) => s.onConnect)
  const addNode = useCanvasStore((s) => s.addNode)
  const setDrawerOpen = useCanvasStore((s) => s.setDrawerOpen)
  const { screenToFlowPosition } = useReactFlow()
  const [menu, setMenu] = useState<ContextMenuState | null>(null)

  // Semantic zoom: hide the insides of collapsed groups. `lod` only changes when the zoom crosses
  // a threshold, so this doesn't recompute on every zoom step.
  const lod = useStore(lodSelector)
  const visibleNodes = useMemo(() => applyLod(nodes, lod), [nodes, lod])
  const visibleEdges = useMemo(() => hideEdgesOfHiddenNodes(edges, visibleNodes), [edges, visibleNodes])

  // Cmd/Ctrl+D duplicates the selection (window listener so it also overrides the browser bookmark shortcut).
  // Cmd/Ctrl+Z undo, Shift+Cmd/Ctrl+Z or Ctrl+Y redo — not while typing (inputs keep their own undo).
  // Esc clears the parameter-category highlight (set from the analysis panel).
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()
      const mod = (e.metaKey || e.ctrlKey) && !isTypingTarget(e.target)
      if (mod && key === 'd') {
        e.preventDefault()
        useCanvasStore.getState().duplicateSelection()
      } else if (mod && (key === 'z' || key === 'y')) {
        e.preventDefault()
        const { undo, redo } = useCanvasStore.getState()
        if (key === 'y' || e.shiftKey) redo()
        else undo()
      } else if (e.key === 'Escape' && useCanvasStore.getState().highlight) {
        useCanvasStore.getState().setHighlight(null)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const item = e.dataTransfer.getData(DND_MIME)
    if (!isPaletteItemId(item)) return
    const p = screenToFlowPosition({ x: e.clientX, y: e.clientY })
    const size = paletteItemSize(item)
    addNode(item, { x: p.x - size.width / 2, y: p.y - size.height / 2 })
  }

  return (
    <>
      <ReactFlow<AppNode, AppEdge>
        nodes={visibleNodes}
        edges={visibleEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onConnectEnd={onConnectEnd}
        onBeforeDelete={async (els) => {
          const allowed = withoutLoneProxies(els)
          if (allowed.nodes.length < els.nodes.length) useCanvasStore.getState().showHint('A group’s in/out pill is removed together with its group.')
          return allowed
        }}
        onNodeClick={() => setDrawerOpen(true)}
        // Right-click menus. On a node that's part of a multi-selection, the menu acts on the whole selection.
        onNodeContextMenu={(e, node) => {
          e.preventDefault()
          const { nodes: all, edges: allEdges, selectNodes } = useCanvasStore.getState()
          const count = all.filter((n) => n.selected).length + allEdges.filter((x) => x.selected).length
          if (node.selected && count > 1) return setMenu({ x: e.clientX, y: e.clientY, target: { kind: 'selection' } })
          selectNodes([node.id])
          setMenu({ x: e.clientX, y: e.clientY, target: { kind: 'node', id: node.id } })
        }}
        onEdgeContextMenu={(e, edge) => {
          e.preventDefault()
          useCanvasStore.getState().selectNodes([], [edge.id])
          setMenu({ x: e.clientX, y: e.clientY, target: { kind: 'edge', id: edge.id } })
        }}
        onSelectionContextMenu={(e) => {
          e.preventDefault()
          setMenu({ x: e.clientX, y: e.clientY, target: { kind: 'selection' } })
        }}
        onPaneContextMenu={(e) => {
          e.preventDefault()
          setMenu({ x: e.clientX, y: e.clientY, target: { kind: 'pane' } })
        }}
        onDragOver={onDragOver}
        onDrop={onDrop}
        isValidConnection={isValidConnection}
        defaultEdgeOptions={defaultEdgeOptions}
        connectionRadius={50}
        // A click on a port opens the shape popover, so click-to-connect is off (drag to connect).
        connectOnClick={false}
        // Miro-like navigation: drag empty canvas = pan, two-finger scroll = pan,
        // pinch or Cmd/Ctrl+wheel = zoom, Shift+drag = box select.
        panOnScroll
        zoomOnDoubleClick={false}
        selectionKeyCode="Shift"
        multiSelectionKeyCode={['Shift', 'Meta', 'Control']}
        deleteKeyCode={['Backspace', 'Delete']}
        minZoom={0.05}
        maxZoom={4}
        fitView
        fitViewOptions={{ padding: 0.1, maxZoom: 1, nodes: topNodes(nodes) }}

        attributionPosition="top-right"
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1.5} color="#cbd5e1" />
        <Controls position="bottom-left" />
        <MiniMap position="bottom-right" pannable zoomable style={{ width: 160, height: 110 }} nodeStrokeWidth={2} nodeColor={minimapColor} />
      </ReactFlow>
      {hint && (
        <div className="pointer-events-none absolute top-3 left-1/2 z-10 -translate-x-1/2 rounded-md bg-slate-800/90 px-3 py-1.5 text-xs text-white shadow">
          {hint}
        </div>
      )}
      <HighlightChip />
      <PortPopover />
      <QuickAddMenu />
      {menu && <ContextMenu menu={menu} onClose={() => setMenu(null)} />}
    </>
  )
}

/** "Attention · 2.10M params" or "Attention probs · 268.4 MB" for the highlighted category. */
function highlightAmount(s: CanvasState): string {
  const key = s.highlight
  if (!key) return ''
  if (isMemKey(key)) return formatBytes(selectMemory(s).activations.byCategory[key.slice(4) as MemoryCategory])
  return `${formatCount(key === 'unconnected' ? s.inference.params.unconnected : s.inference.params.byCategory[key])} params`
}

/** "Highlighting: Attention · 2.10M params" over the canvas while a category is highlighted. */
function HighlightChip() {
  const key = useCanvasStore((s) => s.highlight)
  const amount = useCanvasStore(highlightAmount)
  const setHighlight = useCanvasStore((s) => s.setHighlight)
  if (!key) return null
  const info = highlightInfo(key)
  return (
    <div className="absolute top-3 left-3 z-10 flex items-center gap-2 rounded-full border border-slate-200 bg-white/95 py-1 pr-1 pl-3 text-xs text-slate-600 shadow">
      <span className="h-2.5 w-2.5 rounded-full" style={{ background: info.color }} />
      <span>
        Highlighting <span className="font-semibold text-slate-800">{info.label}</span> · {amount}
      </span>
      <button
        type="button"
        onClick={() => setHighlight(null)}
        title="Clear highlight (Esc)"
        className="flex items-center gap-1 rounded-full px-2 py-0.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
      >
        Esc <X size={12} />
      </button>
    </div>
  )
}

function minimapColor(node: AppNode): string {
  if (node.type === 'sticky') return node.data.bgColor
  if (node.type === 'textbox') return '#e2e8f0'
  if (node.type === 'group') return `${GROUP_DEFS[node.data.groupType].color}33`
  const def = getNodeDef(node.data.partType)
  return def ? CATEGORY_INFO[def.category].color : '#c7d2fe'
}
