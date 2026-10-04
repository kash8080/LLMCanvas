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
import { useEffect, useMemo } from 'react'
import { GROUP_DEFS } from '../nodes/groups'
import { CATEGORY_INFO, getNodeDef } from '../nodes/registry'
import { useCanvasStore, withoutLoneProxies } from '../store/useCanvasStore'
import { connectionProblem, connectionToBody } from './connect'
import { paletteItemSize } from './groupTemplates'
import { applyLod, hideEdgesOfHiddenNodes, lodSelector } from './lod'
import { isPaletteItemId, topNodes } from './nodeFactory'
import { edgeTypes, nodeTypes } from './nodeTypes'
import { PortPopover } from './PortPopover'
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

// Refused drops on a port explain why; drops on a node body connect to its first free input.
const onConnectEnd: OnConnectEnd = (event, state) => {
  if (state.isValid || !state.fromNode || !state.fromHandle) return
  const { nodes, edges, onConnect, showHint } = useCanvasStore.getState()
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
  if (!targetId || targetId === from.nodeId) return
  // Let go on the empty area of the group the drag started in: nothing to connect, no hint.
  if (nodes.find((n) => n.id === from.nodeId)?.parentId === targetId) return
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

  // Semantic zoom: hide the insides of collapsed groups. `lod` only changes when the zoom crosses
  // a threshold, so this doesn't recompute on every zoom step.
  const lod = useStore(lodSelector)
  const visibleNodes = useMemo(() => applyLod(nodes, lod), [nodes, lod])
  const visibleEdges = useMemo(() => hideEdgesOfHiddenNodes(edges, visibleNodes), [edges, visibleNodes])

  // Cmd/Ctrl+D duplicates the selection (window listener so it also overrides the browser bookmark shortcut).
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'd' && !isTypingTarget(e.target)) {
        e.preventDefault()
        useCanvasStore.getState().duplicateSelection()
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
      <PortPopover />
    </>
  )
}

function minimapColor(node: AppNode): string {
  if (node.type === 'sticky') return node.data.bgColor
  if (node.type === 'textbox') return '#e2e8f0'
  if (node.type === 'group') return `${GROUP_DEFS[node.data.groupType].color}33`
  const def = getNodeDef(node.data.partType)
  return def ? CATEGORY_INFO[def.category].color : '#c7d2fe'
}
