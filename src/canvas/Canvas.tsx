import {
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  useReactFlow,
  type DefaultEdgeOptions,
  type IsValidConnection,
} from '@xyflow/react'
import { useEffect } from 'react'
import { useCanvasStore } from '../store/useCanvasStore'
import { DEFAULT_SIZE } from './nodeFactory'
import { nodeTypes } from './nodeTypes'
import { DND_MIME, isNodeKind, type AppEdge, type AppNode } from './types'

const defaultEdgeOptions: DefaultEdgeOptions = {
  markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: '#94a3b8' },
  style: { stroke: '#94a3b8', strokeWidth: 1.5 },
}

// Phase 2 adds real validation (port compatibility, single input, cycles).
const isValidConnection: IsValidConnection<AppEdge> = (c) => c.source !== c.target

function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)
}

export function Canvas() {
  const nodes = useCanvasStore((s) => s.nodes)
  const edges = useCanvasStore((s) => s.edges)
  const onNodesChange = useCanvasStore((s) => s.onNodesChange)
  const onEdgesChange = useCanvasStore((s) => s.onEdgesChange)
  const onConnect = useCanvasStore((s) => s.onConnect)
  const addNode = useCanvasStore((s) => s.addNode)
  const setDrawerOpen = useCanvasStore((s) => s.setDrawerOpen)
  const { screenToFlowPosition } = useReactFlow()

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
    const kind = e.dataTransfer.getData(DND_MIME)
    if (!isNodeKind(kind)) return
    const p = screenToFlowPosition({ x: e.clientX, y: e.clientY })
    const size = DEFAULT_SIZE[kind]
    addNode(kind, { x: p.x - size.width / 2, y: p.y - size.height / 2 })
  }

  return (
    <ReactFlow<AppNode, AppEdge>
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onNodeClick={() => setDrawerOpen(true)}
      onDragOver={onDragOver}
      onDrop={onDrop}
      isValidConnection={isValidConnection}
      defaultEdgeOptions={defaultEdgeOptions}
      connectionRadius={40}
      // Miro-like navigation: drag empty canvas = pan, two-finger scroll = pan,
      // pinch or Cmd/Ctrl+wheel = zoom, Shift+drag = box select.
      panOnScroll
      zoomOnDoubleClick={false}
      selectionKeyCode="Shift"
      multiSelectionKeyCode={['Shift', 'Meta', 'Control']}
      deleteKeyCode={['Backspace', 'Delete']}
      minZoom={0.1}
      maxZoom={4}
      fitView
      fitViewOptions={{ padding: 0.3, maxZoom: 1 }}
      attributionPosition="top-right"
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1.5} color="#cbd5e1" />
      <Controls position="bottom-left" />
      <MiniMap position="bottom-right" pannable zoomable style={{ width: 160, height: 110 }} nodeStrokeWidth={2} nodeColor={minimapColor} />
    </ReactFlow>
  )
}

function minimapColor(node: AppNode): string {
  if (node.type === 'sticky') return node.data.bgColor
  if (node.type === 'textbox') return '#e2e8f0'
  return '#c7d2fe'
}
