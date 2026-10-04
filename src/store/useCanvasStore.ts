import { applyEdgeChanges, applyNodeChanges, type Connection, type EdgeChange, type NodeChange, type XYPosition } from '@xyflow/react'
import { create } from 'zustand'
import { instantiateGroup, isGroupItem, nextBlockTitle } from '../canvas/groupTemplates'
import { createNode, newId } from '../canvas/nodeFactory'
import type { AnnotationData, AppEdge, AppNode, GroupMode, PaletteItemId } from '../canvas/types'
import { cs336Document } from '../defaults/cs336Graph'
import { isProxyType } from '../engine/groups'
import type { MemoryHighlightKey, MemoryMode } from '../engine/memory'
import type { HighlightKey } from '../engine/params'
import type { Hyperparams, ParamValue } from '../engine/types'
import type { GroupType } from '../nodes/groups'
import { inferCanvas, memoryFor, type CanvasInference } from './inference'
import { loadFromStorage, saveToStorage, toDocument, type CanvasDocument } from './persistence'

/** A clicked port: which node/port, and where on screen to show the popover. */
export interface PortPopover {
  nodeId: string
  portId: string
  kind: 'in' | 'out'
  x: number
  y: number
}

/** Tabs of the bottom analysis panel. */
export type AnalysisTab = 'params' | 'memory'

/** What the analysis panel can highlight on the canvas: a param category / 'unconnected', or an activation category. */
export type CanvasHighlight = HighlightKey | MemoryHighlightKey

export interface CanvasState {
  nodes: AppNode[]
  edges: AppEdge[]
  hyperparams: Hyperparams
  /** Derived: shape inference + group summaries for the current nodes/edges/hyperparams (recomputed on every change). */
  inference: CanvasInference
  // UI state
  drawerOpen: boolean
  analysisOpen: boolean
  analysisTab: AnalysisTab
  /** Param / activation category highlighted on the canvas from the analysis panel; Esc clears it. */
  highlight: CanvasHighlight | null
  /** Memory estimate settings (R8): mode and activation checkpointing (CS336 `checkpoint_blocks`). UI state, not saved. */
  memoryMode: MemoryMode
  checkpointing: boolean
  /** Tint parts by the activation memory they hold. */
  memoryHeat: boolean
  showEdgeShapes: boolean
  portPopover: PortPopover | null
  /** Short message shown briefly over the canvas (e.g. why a connection was refused). */
  hint: string | null
  /** Toolbar Hyperparams popover (also opened from the drawer's model summary). */
  hyperparamsOpen: boolean
  /** Drawer sections the user collapsed, by section key (shared by every selection). */
  collapsedSections: Record<string, boolean>

  // React Flow wiring
  onNodesChange: (changes: NodeChange<AppNode>[]) => void
  onEdgesChange: (changes: EdgeChange<AppEdge>[]) => void
  /** Adds an edge; an input accepts one edge, so an existing edge into the same input is replaced. */
  onConnect: (connection: Connection) => void

  // Actions
  addNode: (item: PaletteItemId, position: XYPosition) => void
  updateAnnotation: (id: string, patch: Partial<AnnotationData>) => void
  setPartParam: (id: string, key: string, value: ParamValue) => void
  /** Rename a part or a group ('' = back to the default label). */
  setTitle: (id: string, title: string) => void
  setGroupMode: (id: string, mode: GroupMode) => void
  setHyperparam: <K extends keyof Hyperparams>(key: K, value: Hyperparams[K]) => void
  duplicateSelection: () => void
  loadDocument: (doc: CanvasDocument) => void
  resetCanvas: () => void
  setDrawerOpen: (open: boolean) => void
  /** Select exactly this node (deselects everything else) and open the drawer. */
  selectOnly: (id: string) => void
  setHyperparamsOpen: (open: boolean) => void
  toggleSection: (key: string) => void
  setAnalysisOpen: (open: boolean) => void
  /** Open the analysis panel on a tab. */
  openAnalysis: (tab: AnalysisTab) => void
  setHighlight: (key: CanvasHighlight | null) => void
  setMemoryMode: (mode: MemoryMode) => void
  setCheckpointing: (on: boolean) => void
  setMemoryHeat: (on: boolean) => void
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

function withInference(g: GraphState): GraphState & { inference: CanvasInference } {
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
    analysisTab: 'params',
    highlight: null,
    memoryMode: 'train',
    checkpointing: false,
    memoryHeat: false,
    showEdgeShapes: true,
    portPopover: null,
    hint: null,
    hyperparamsOpen: false,
    collapsedSections: {},

    onNodesChange: (changes) => commit({ nodes: applyNodeChanges(changes, get().nodes) }),
    onEdgesChange: (changes) => commit({ edges: applyEdgeChanges(changes, get().edges) }),
    onConnect: (c) => {
      const kept = get().edges.filter((e) => !(e.target === c.target && (e.targetHandle ?? null) === (c.targetHandle ?? null)))
      commit({ edges: [...kept, { ...c, id: newId('e') }] })
    },

    addNode: (item, position) => {
      const { nodes, edges } = get()
      const deselected = nodes.map((n) => (n.selected ? { ...n, selected: false } : n))
      if (isGroupItem(item)) {
        // A whole template: frame + children + internal edges; only the frame is selected.
        const built = instantiateGroup(item.slice(6) as GroupType, position, nodes)
        const [frame, ...children] = built.nodes
        commit({ nodes: [...deselected, { ...frame, selected: true }, ...children], edges: [...edges, ...built.edges] })
      } else {
        commit({ nodes: [...deselected, { ...createNode(item, position), selected: true }] })
      }
      set({ drawerOpen: true })
    },

    updateAnnotation: (id, patch) =>
      commit({
        nodes: get().nodes.map((n) =>
          n.id === id && (n.type === 'sticky' || n.type === 'textbox') ? { ...n, data: { ...n.data, ...patch } } : n,
        ),
      }),

    setPartParam: (id, key, value) => mapPart(id, (n) => ({ ...n, data: { ...n.data, params: { ...n.data.params, [key]: value } } })),

    setTitle: (id, title) =>
      commit({
        nodes: get().nodes.map((n) =>
          n.id === id && (n.type === 'part' || n.type === 'group') ? ({ ...n, data: { ...n.data, title: title || undefined } } as AppNode) : n,
        ),
      }),

    setGroupMode: (id, mode) =>
      commit({ nodes: get().nodes.map((n) => (n.id === id && n.type === 'group' ? { ...n, data: { ...n.data, mode } } : n)) }),

    setHyperparam: (key, value) => commit({ hyperparams: { ...get().hyperparams, [key]: value } }),

    duplicateSelection: () => {
      const { nodes, edges } = get()
      const selectedIds = new Set(nodes.filter((n) => n.selected).map((n) => n.id))
      if (selectedIds.size === 0) return

      // Copy the selection plus everything inside selected groups (parents come before children).
      // A proxy is only copied together with its group (a group has exactly one in / out proxy).
      const toCopy = new Set<string>()
      for (const n of nodes) if (selectedIds.has(n.id) || (n.parentId && toCopy.has(n.parentId))) toCopy.add(n.id)
      const copied = nodes.filter((n) => toCopy.has(n.id) && !(n.type === 'part' && isProxyType(n.data.partType) && !toCopy.has(n.parentId!)))
      if (copied.length === 0) return

      // Groups are big: put their copy beside the original instead of overlapping it.
      const groupWidths = copied.filter((n) => n.type === 'group' && !(n.parentId && toCopy.has(n.parentId))).map((n) => n.width ?? 0)
      const offset = groupWidths.length > 0 ? { x: Math.max(...groupWidths) + 80, y: 0 } : { x: DUPLICATE_OFFSET, y: DUPLICATE_OFFSET }

      const idMap = new Map<string, string>()
      let blocks = nodes
      const copies = copied.map((n) => {
        const id = newId(n.type === 'part' ? n.data.partType : n.type === 'group' ? n.data.groupType : n.type)
        idMap.set(n.id, id)
        const { measured: _measured, dragging: _dragging, hidden: _hidden, ...rest } = n
        const insideCopy = !!n.parentId && idMap.has(n.parentId)
        const copy = {
          ...rest,
          id,
          data: structuredClone(n.data),
          // Children keep their position relative to the (copied) frame.
          position: insideCopy ? n.position : { x: n.position.x + offset.x, y: n.position.y + offset.y },
          ...(insideCopy ? { parentId: idMap.get(n.parentId!) } : {}),
          selected: !insideCopy,
        } as AppNode
        if (copy.type === 'group' && copy.data.groupType === 'transformer_block') {
          copy.data = { ...copy.data, title: nextBlockTitle(blocks) }
          blocks = [...blocks, copy]
        }
        return copy
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
    selectOnly: (id) => {
      const { nodes, edges } = get()
      commit({
        nodes: nodes.map((n) => (!!n.selected !== (n.id === id) ? { ...n, selected: n.id === id } : n)),
        edges: edges.some((e) => e.selected) ? edges.map((e) => (e.selected ? { ...e, selected: false } : e)) : edges,
      })
      set({ drawerOpen: true })
    },
    setHyperparamsOpen: (open) => set({ hyperparamsOpen: open }),
    toggleSection: (key) => set((s) => ({ collapsedSections: { ...s.collapsedSections, [key]: !s.collapsedSections[key] } })),
    setAnalysisOpen: (open) => set({ analysisOpen: open }),
    openAnalysis: (tab) => set({ analysisOpen: true, analysisTab: tab }),
    setHighlight: (key) => set({ highlight: key }),
    setMemoryMode: (mode) => set({ memoryMode: mode }),
    setCheckpointing: (on) => set({ checkpointing: on }),
    setMemoryHeat: (on) => set({ memoryHeat: on }),
    setShowEdgeShapes: (show) => set({ showEdgeShapes: show }),
    setPortPopover: (popover) => set({ portPopover: popover }),
    showHint: (message) => {
      clearTimeout(hintTimer)
      set({ hint: message })
      hintTimer = setTimeout(() => set({ hint: null }), 3000)
    },
  }
})

/** The memory estimate for the current graph + mode (memoised; engine/memory.ts). Use as a selector. */
export const selectMemory = (s: CanvasState) => memoryFor(s.inference, s.hyperparams, s.memoryMode, s.checkpointing)

/**
 * Deleting: a group's in/out proxies can't be deleted on their own (the group would lose its port),
 * only together with their group. Edges are kept unless explicitly selected or attached to
 * something that is really deleted. Used as React Flow's `onBeforeDelete`.
 */
export function withoutLoneProxies(toDelete: { nodes: AppNode[]; edges: AppEdge[] }): { nodes: AppNode[]; edges: AppEdge[] } {
  const ids = new Set(toDelete.nodes.map((n) => n.id))
  const nodes = toDelete.nodes.filter((n) => !(n.type === 'part' && isProxyType(n.data.partType) && !ids.has(n.parentId ?? '')))
  if (nodes.length === toDelete.nodes.length) return toDelete
  const kept = new Set(nodes.map((n) => n.id))
  const edges = toDelete.edges.filter((e) => e.selected || kept.has(e.source) || kept.has(e.target))
  return { nodes, edges }
}

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
