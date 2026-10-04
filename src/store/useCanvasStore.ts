import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type EdgeChange,
  type NodeChange,
  type XYPosition,
} from '@xyflow/react'
import { create } from 'zustand'
import { createNode, newId } from '../canvas/nodeFactory'
import type { AnnotationData, AppEdge, AppNode, NodeKind } from '../canvas/types'
import { demoDocument } from '../defaults/demoGraph'
import { loadFromStorage, saveToStorage, toDocument, type CanvasDocument } from './persistence'

export interface CanvasState {
  nodes: AppNode[]
  edges: AppEdge[]
  // UI state
  drawerOpen: boolean
  analysisOpen: boolean

  // React Flow wiring
  onNodesChange: (changes: NodeChange<AppNode>[]) => void
  onEdgesChange: (changes: EdgeChange<AppEdge>[]) => void
  onConnect: (connection: Connection) => void

  // Actions
  addNode: (kind: NodeKind, position: XYPosition) => void
  updateAnnotation: (id: string, patch: Partial<AnnotationData>) => void
  duplicateSelection: () => void
  loadDocument: (doc: CanvasDocument) => void
  resetCanvas: () => void
  setDrawerOpen: (open: boolean) => void
  setAnalysisOpen: (open: boolean) => void
}

const DUPLICATE_OFFSET = 30

function docToState(doc: CanvasDocument): Pick<CanvasState, 'nodes' | 'edges'> {
  // structuredClone so the store never shares objects with the source document
  return { nodes: structuredClone(doc.nodes) as AppNode[], edges: structuredClone(doc.edges) }
}

export const useCanvasStore = create<CanvasState>()((set, get) => ({
  ...docToState(loadFromStorage() ?? demoDocument()),
  drawerOpen: true,
  analysisOpen: false,

  onNodesChange: (changes) => set({ nodes: applyNodeChanges(changes, get().nodes) }),
  onEdgesChange: (changes) => set({ edges: applyEdgeChanges(changes, get().edges) }),
  onConnect: (connection) => set({ edges: addEdge({ ...connection, id: newId('e') }, get().edges) }),

  addNode: (kind, position) => {
    const node = { ...createNode(kind, position), selected: true }
    set({
      nodes: [...get().nodes.map((n) => (n.selected ? { ...n, selected: false } : n)), node],
      drawerOpen: true,
    })
  },

  updateAnnotation: (id, patch) =>
    set({
      nodes: get().nodes.map((n) =>
        n.id === id && (n.type === 'sticky' || n.type === 'textbox') ? { ...n, data: { ...n.data, ...patch } } : n,
      ),
    }),

  duplicateSelection: () => {
    const { nodes, edges } = get()
    const selected = nodes.filter((n) => n.selected)
    if (selected.length === 0) return

    const idMap = new Map<string, string>()
    const copies = selected.map((n) => {
      const id = newId(n.type)
      idMap.set(n.id, id)
      const { measured: _measured, dragging: _dragging, ...rest } = n
      return {
        ...rest,
        id,
        data: structuredClone(n.data),
        position: { x: n.position.x + DUPLICATE_OFFSET, y: n.position.y + DUPLICATE_OFFSET },
        selected: true,
      } as AppNode
    })
    // Keep edges that run between two duplicated nodes.
    const edgeCopies = edges
      .filter((e) => idMap.has(e.source) && idMap.has(e.target))
      .map((e) => ({ ...e, id: newId('e'), source: idMap.get(e.source)!, target: idMap.get(e.target)!, selected: false }))

    set({
      nodes: [...nodes.map((n) => (n.selected ? { ...n, selected: false } : n)), ...copies],
      edges: [...edges.map((e) => (e.selected ? { ...e, selected: false } : e)), ...edgeCopies],
    })
  },

  loadDocument: (doc) => set(docToState(doc)),
  resetCanvas: () => set(docToState(demoDocument())),
  setDrawerOpen: (open) => set({ drawerOpen: open }),
  setAnalysisOpen: (open) => set({ analysisOpen: open }),
}))

/** Debounced autosave of nodes/edges to localStorage. Call once at startup. */
export function startAutosave(delayMs = 400): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined
  const unsubscribe = useCanvasStore.subscribe((state, prev) => {
    if (state.nodes === prev.nodes && state.edges === prev.edges) return
    clearTimeout(timer)
    timer = setTimeout(() => saveToStorage(toDocument(state.nodes, state.edges)), delayMs)
  })
  return () => {
    clearTimeout(timer)
    unsubscribe()
  }
}
