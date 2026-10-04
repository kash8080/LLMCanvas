import { applyEdgeChanges, applyNodeChanges, type Connection, type EdgeChange, type NodeChange, type XYPosition } from '@xyflow/react'
import { create } from 'zustand'
import { createNode, newId } from '../canvas/nodeFactory'
import type { AnnotationData, AppEdge, AppNode, PaletteItemId } from '../canvas/types'
import { cs336Document } from '../defaults/cs336Graph'
import type { Hyperparams, InferenceResult, ParamValue } from '../engine/types'
import { inferCanvas } from './inference'
import { loadFromStorage, saveToStorage, toDocument, type CanvasDocument } from './persistence'

/** A clicked port: which node/port, and where on screen to show the popover. */
export interface PortPopover {
  nodeId: string
  portId: string
  kind: 'in' | 'out'
  x: number
  y: number
}

export interface CanvasState {
  nodes: AppNode[]
  edges: AppEdge[]
  hyperparams: Hyperparams
  /** Derived: shape inference for the current nodes/edges/hyperparams (recomputed on every change). */
  inference: InferenceResult
  // UI state
  drawerOpen: boolean
  analysisOpen: boolean
  showEdgeShapes: boolean
  portPopover: PortPopover | null
  /** Short message shown briefly over the canvas (e.g. why a connection was refused). */
  hint: string | null

  // React Flow wiring
  onNodesChange: (changes: NodeChange<AppNode>[]) => void
  onEdgesChange: (changes: EdgeChange<AppEdge>[]) => void
  /** Adds an edge; an input accepts one edge, so an existing edge into the same input is replaced. */
  onConnect: (connection: Connection) => void

  // Actions
  addNode: (item: PaletteItemId, position: XYPosition) => void
  updateAnnotation: (id: string, patch: Partial<AnnotationData>) => void
  setPartParam: (id: string, key: string, value: ParamValue) => void
  setPartTitle: (id: string, title: string) => void
  setHyperparam: <K extends keyof Hyperparams>(key: K, value: Hyperparams[K]) => void
  duplicateSelection: () => void
  loadDocument: (doc: CanvasDocument) => void
  resetCanvas: () => void
  setDrawerOpen: (open: boolean) => void
  setAnalysisOpen: (open: boolean) => void
  setShowEdgeShapes: (show: boolean) => void
  setPortPopover: (popover: PortPopover | null) => void
  showHint: (message: string) => void
}

const DUPLICATE_OFFSET = 30
let hintTimer: ReturnType<typeof setTimeout> | undefined

type GraphState = Pick<CanvasState, 'nodes' | 'edges' | 'hyperparams'>

function docToState(doc: CanvasDocument): GraphState {
  // structuredClone so the store never shares objects with the source document
  return { nodes: structuredClone(doc.nodes) as AppNode[], edges: structuredClone(doc.edges), hyperparams: { ...doc.hyperparams } }
}

function withInference(g: GraphState): GraphState & { inference: InferenceResult } {
  return { ...g, inference: inferCanvas(g.nodes, g.edges, g.hyperparams) }
}

export const useCanvasStore = create<CanvasState>()((set, get) => {
  /** Update graph state (nodes / edges / hyperparams) and re-derive the inference. */
  const commit = (patch: Partial<GraphState>) => {
    const { nodes, edges, hyperparams } = get()
    set(withInference({ nodes, edges, hyperparams, ...patch }))
  }
  const mapPart = (id: string, fn: (n: AppNode & { type: 'part' }) => AppNode) =>
    commit({ nodes: get().nodes.map((n) => (n.id === id && n.type === 'part' ? fn(n) : n)) })

  return {
    ...withInference(docToState(loadFromStorage() ?? cs336Document())),
    drawerOpen: true,
    analysisOpen: false,
    showEdgeShapes: true,
    portPopover: null,
    hint: null,

    onNodesChange: (changes) => commit({ nodes: applyNodeChanges(changes, get().nodes) }),
    onEdgesChange: (changes) => commit({ edges: applyEdgeChanges(changes, get().edges) }),
    onConnect: (c) => {
      const kept = get().edges.filter((e) => !(e.target === c.target && (e.targetHandle ?? null) === (c.targetHandle ?? null)))
      commit({ edges: [...kept, { ...c, id: newId('e') }] })
    },

    addNode: (item, position) => {
      const node = { ...createNode(item, position), selected: true }
      commit({ nodes: [...get().nodes.map((n) => (n.selected ? { ...n, selected: false } : n)), node] })
      set({ drawerOpen: true })
    },

    updateAnnotation: (id, patch) =>
      commit({
        nodes: get().nodes.map((n) =>
          n.id === id && (n.type === 'sticky' || n.type === 'textbox') ? { ...n, data: { ...n.data, ...patch } } : n,
        ),
      }),

    setPartParam: (id, key, value) => mapPart(id, (n) => ({ ...n, data: { ...n.data, params: { ...n.data.params, [key]: value } } })),

    setPartTitle: (id, title) => mapPart(id, (n) => ({ ...n, data: { ...n.data, title: title || undefined } })),

    setHyperparam: (key, value) => commit({ hyperparams: { ...get().hyperparams, [key]: value } }),

    duplicateSelection: () => {
      const { nodes, edges } = get()
      const selected = nodes.filter((n) => n.selected)
      if (selected.length === 0) return

      const idMap = new Map<string, string>()
      const copies = selected.map((n) => {
        const id = newId(n.type === 'part' ? n.data.partType : n.type)
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

      commit({
        nodes: [...nodes.map((n) => (n.selected ? { ...n, selected: false } : n)), ...copies],
        edges: [...edges.map((e) => (e.selected ? { ...e, selected: false } : e)), ...edgeCopies],
      })
    },

    loadDocument: (doc) => {
      commit(docToState(doc))
      set({ portPopover: null })
    },
    resetCanvas: () => get().loadDocument(cs336Document()),
    setDrawerOpen: (open) => set({ drawerOpen: open }),
    setAnalysisOpen: (open) => set({ analysisOpen: open }),
    setShowEdgeShapes: (show) => set({ showEdgeShapes: show }),
    setPortPopover: (popover) => set({ portPopover: popover }),
    showHint: (message) => {
      clearTimeout(hintTimer)
      set({ hint: message })
      hintTimer = setTimeout(() => set({ hint: null }), 3000)
    },
  }
})

/** Debounced autosave of nodes/edges/hyperparams to localStorage. Call once at startup. */
export function startAutosave(delayMs = 400): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined
  const unsubscribe = useCanvasStore.subscribe((state, prev) => {
    if (state.nodes === prev.nodes && state.edges === prev.edges && state.hyperparams === prev.hyperparams) return
    clearTimeout(timer)
    timer = setTimeout(() => saveToStorage(toDocument(state.nodes, state.edges, state.hyperparams)), delayMs)
  })
  return () => {
    clearTimeout(timer)
    unsubscribe()
  }
}
