import { applyEdgeChanges, applyNodeChanges, type Connection, type EdgeChange, type NodeChange, type XYPosition } from '@xyflow/react'
import { create } from 'zustand'
import { connectionToBody, type ConnectionLike } from '../canvas/connect'
import { instantiateGroup, isGroupItem, nextBlockTitle } from '../canvas/groupTemplates'
import { frameBoxAround, frameFollowChanges, topLevelIds } from '../canvas/frames'
import { createFrameNode, createNode, newId } from '../canvas/nodeFactory'
import type { AnnotationData, AppEdge, AppNode, FrameData, GroupMode, PaletteItemId } from '../canvas/types'
import { cs336Document } from '../defaults/cs336Graph'
import { isProxyType } from '../engine/groups'
import type { GenerationSettings, MemoryHighlightKey, MemoryMode } from '../engine/memory'
import type { HighlightKey } from '../engine/params'
import type { Hyperparams, ParamValue } from '../engine/types'
import type { GroupType } from '../nodes/groups'
import { emptyHistory, record, redo, undo, type History } from './history'
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

/** The end of a connection being dragged (React Flow handle): node, port, output ('source') or input ('target'). */
export interface ConnectionFrom {
  nodeId: string
  handleId: string | null
  type: 'source' | 'target'
}

/**
 * The quick-add menu: opened at a screen point (client coords) when a connection is dropped on
 * empty canvas (`from` set → the new part gets connected) or from the canvas context menu.
 * `parentId`: the drop was on the empty area of this group frame → the new part goes inside it.
 */
export interface QuickAddState {
  x: number
  y: number
  from: ConnectionFrom | null
  parentId?: string
}

/** Options for `addNode`: create inside a group frame and/or connect to a dragged port. */
export interface AddNodeOptions {
  parentId?: string
  connectFrom?: ConnectionFrom
}

/** Generation (KV cache) view of the Memory tab. */
export interface GenerationState {
  on: boolean
  length: number | null
  batch: number | null
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
  /**
   * Forward mode: estimate generation with a KV cache (Memory tab). `length` / `batch` null = follow
   * context_length / batch_size. UI state like the mode: not saved, not undoable.
   */
  generation: GenerationState
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
  /** Undo / redo stacks of graph snapshots (not persisted). */
  history: History<GraphSnapshot>
  quickAdd: QuickAddState | null

  // React Flow wiring
  onNodesChange: (changes: NodeChange<AppNode>[]) => void
  onEdgesChange: (changes: EdgeChange<AppEdge>[]) => void
  /** Adds an edge; an input accepts one edge, so an existing edge into the same input is replaced. */
  onConnect: (connection: Connection) => void

  // Actions
  /** Add a palette item (a group = its whole template); `position` is relative to `opts.parentId` if given. */
  addNode: (item: PaletteItemId, position: XYPosition, opts?: AddNodeOptions) => void
  updateAnnotation: (id: string, patch: Partial<AnnotationData>) => void
  /** Frame colours (title: `setTitle`). */
  updateFrame: (id: string, patch: Partial<Omit<FrameData, 'title'>>) => void
  /**
   * Wrap these items (default: the selection) in a new frame sized to their bounds + padding. Items inside
   * a group count as their top-level group. Selects the new frame; returns its id (null if nothing to frame).
   */
  frameSelection: (ids?: string[]) => string | null
  setPartParam: (id: string, key: string, value: ParamValue) => void
  /** Rename a part, group or frame ('' = back to the default label). */
  setTitle: (id: string, title: string) => void
  setGroupMode: (id: string, mode: GroupMode) => void
  setHyperparam: <K extends keyof Hyperparams>(key: K, value: Hyperparams[K]) => void
  duplicateSelection: () => void
  loadDocument: (doc: CanvasDocument) => void
  resetCanvas: () => void
  setDrawerOpen: (open: boolean) => void
  /** Select exactly this node (deselects everything else) and open the drawer (unless `openDrawer` is false). */
  selectOnly: (id: string, openDrawer?: boolean) => void
  /** Select exactly these nodes and edges (everything else deselected). */
  selectNodes: (ids: string[], edgeIds?: string[]) => void
  undo: () => void
  redo: () => void
  clearHistory: () => void
  openQuickAdd: (state: QuickAddState) => void
  closeQuickAdd: () => void
  setHyperparamsOpen: (open: boolean) => void
  toggleSection: (key: string) => void
  setAnalysisOpen: (open: boolean) => void
  /** Open the analysis panel on a tab. */
  openAnalysis: (tab: AnalysisTab) => void
  setHighlight: (key: CanvasHighlight | null) => void
  setMemoryMode: (mode: MemoryMode) => void
  setCheckpointing: (on: boolean) => void
  setGeneration: (patch: Partial<GenerationState>) => void
  setMemoryHeat: (on: boolean) => void
  setShowEdgeShapes: (show: boolean) => void
  setPortPopover: (popover: PortPopover | null) => void
  showHint: (message: string) => void
}

const DUPLICATE_OFFSET = 30
let hintTimer: ReturnType<typeof setTimeout> | undefined

type GraphState = Pick<CanvasState, 'nodes' | 'edges' | 'hyperparams'>
/** What undo / redo restores. */
export type GraphSnapshot = GraphState

/** Add edge `c`; an input takes one edge, so an existing edge into the same input is dropped. */
function withEdge(edges: AppEdge[], c: ConnectionLike): AppEdge[] {
  const kept = edges.filter((e) => !(e.target === c.target && (e.targetHandle ?? null) === (c.targetHandle ?? null)))
  return [...kept, { ...c, sourceHandle: c.sourceHandle ?? null, targetHandle: c.targetHandle ?? null, id: newId('e') }]
}

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

  // ---- Undo history (store/history.ts). Every undoable action calls `remember()` *before* changing the graph. ----
  const snapshot = (): GraphSnapshot => {
    const { nodes, edges, hyperparams } = get()
    return { nodes, edges, hyperparams }
  }
  /** Record the current graph as an undo step. Same `key` in quick succession = one step (typing). */
  const remember = (key?: string) => set({ history: record(get().history, snapshot(), { key }) })
  // React Flow deletes edges and nodes in two separate change calls: record once per tick.
  let removalRecorded = false
  const rememberRemoval = () => {
    if (removalRecorded) return
    removalRecorded = true
    queueMicrotask(() => (removalRecorded = false))
    remember()
  }
  // A drag / resize emits many changes; keep the state from its start and record it when it ends.
  let gestureStart: GraphSnapshot | null = null
  const trackGesture = (changes: NodeChange<AppNode>[]) => {
    const active = changes.some((c) => (c.type === 'position' && c.dragging) || (c.type === 'dimensions' && c.resizing))
    const ended = changes.some((c) => (c.type === 'position' && c.dragging === false) || (c.type === 'dimensions' && c.resizing === false))
    if (active && !gestureStart) gestureStart = snapshot()
    if (ended && gestureStart) {
      set({ history: record(get().history, gestureStart) })
      gestureStart = null
    }
  }
  // Frames drag their contents along (canvas/frames.ts): contents captured at drag start, per frame.
  const frameDrags = new Map<string, string[]>()
  const restore = (step: ((h: History<GraphSnapshot>, cur: GraphSnapshot) => { history: History<GraphSnapshot>; state: GraphSnapshot } | null)) => {
    const r = step(get().history, snapshot())
    if (!r) return
    gestureStart = null
    frameDrags.clear()
    set({ history: r.history, portPopover: null, quickAdd: null })
    commit(r.state)
  }

  return {
    ...withInference(docToState(loadFromStorage() ?? cs336Document())),
    drawerOpen: true,
    analysisOpen: false,
    analysisTab: 'params',
    highlight: null,
    memoryMode: 'train',
    checkpointing: false,
    generation: { on: false, length: null, batch: null },
    memoryHeat: false,
    showEdgeShapes: true,
    portPopover: null,
    hint: null,
    hyperparamsOpen: false,
    collapsedSections: {},
    history: emptyHistory(),
    quickAdd: null,

    onNodesChange: (changes) => {
      trackGesture(changes)
      if (changes.some((c) => c.type === 'remove')) rememberRemoval()
      const nodes = get().nodes
      const follow = frameFollowChanges(changes, nodes, frameDrags)
      commit({ nodes: applyNodeChanges(follow.length > 0 ? [...changes, ...follow] : changes, nodes) })
    },
    onEdgesChange: (changes) => {
      if (changes.some((c) => c.type === 'remove')) rememberRemoval()
      commit({ edges: applyEdgeChanges(changes, get().edges) })
    },
    onConnect: (c) => {
      remember()
      commit({ edges: withEdge(get().edges, c) })
    },

    addNode: (item, position, opts = {}) => {
      remember()
      const { nodes, edges } = get()
      const deselected = nodes.map((n) => (n.selected ? { ...n, selected: false } : n))
      const edgesOff = edges.map((e) => (e.selected ? { ...e, selected: false } : e))
      let added: AppNode[]
      let nextEdges: AppEdge[] = edgesOff
      if (isGroupItem(item)) {
        // A whole template: frame + children + internal edges; only the frame is selected.
        // With `parentId` it is nested (a sub-layer inside a Transformer Block, e.g. swapping its FFN).
        const built = instantiateGroup(item.slice(6) as GroupType, position, nodes, opts.parentId)
        const [frame, ...children] = built.nodes
        added = [{ ...frame, selected: true }, ...children]
        nextEdges = [...edgesOff, ...built.edges]
      } else {
        const node = { ...createNode(item, position), selected: true } as AppNode
        // Frames are visual only and always top-level (never inside a semantic group).
        added = [opts.parentId && node.type !== 'frame' ? ({ ...node, parentId: opts.parentId, extent: 'parent' } as AppNode) : node]
      }
      const nextNodes = [...deselected, ...added]
      if (opts.connectFrom) {
        const r = connectionToBody(nextNodes, nextEdges, opts.connectFrom, added[0].id)
        if ('connection' in r) nextEdges = withEdge(nextEdges, r.connection)
        else get().showHint(r.problem)
      }
      commit({ nodes: nextNodes, edges: nextEdges })
      set({ drawerOpen: true, quickAdd: null })
    },

    updateAnnotation: (id, patch) => {
      remember(`annot:${id}:${Object.keys(patch).sort().join(',')}`)
      commit({
        nodes: get().nodes.map((n) =>
          n.id === id && (n.type === 'sticky' || n.type === 'textbox') ? { ...n, data: { ...n.data, ...patch } } : n,
        ),
      })
    },

    updateFrame: (id, patch) => {
      remember(`frame:${id}:${Object.keys(patch).sort().join(',')}`)
      commit({ nodes: get().nodes.map((n) => (n.id === id && n.type === 'frame' ? { ...n, data: { ...n.data, ...patch } } : n)) })
    },

    frameSelection: (ids) => {
      const { nodes, edges } = get()
      const items = topLevelIds(nodes, ids ?? nodes.filter((n) => n.selected).map((n) => n.id))
      const box = frameBoxAround(nodes, items)
      if (!box) return null
      remember()
      const frame = { ...createFrameNode(box), selected: true }
      commit({
        nodes: [...nodes.map((n) => (n.selected ? { ...n, selected: false } : n)), frame],
        edges: edges.some((e) => e.selected) ? edges.map((e) => (e.selected ? { ...e, selected: false } : e)) : edges,
      })
      set({ drawerOpen: true })
      return frame.id
    },

    setPartParam: (id, key, value) => {
      // Typing a local value commits on every valid keystroke: one undo step per field edit.
      remember(`param:${id}:${key}:${'bind' in value ? 'bind' : 'value'}`)
      mapPart(id, (n) => ({ ...n, data: { ...n.data, params: { ...n.data.params, [key]: value } } }))
    },

    setTitle: (id, title) => {
      remember(`title:${id}`)
      commit({
        nodes: get().nodes.map((n) =>
          n.id === id && (n.type === 'part' || n.type === 'group' || n.type === 'frame') ? ({ ...n, data: { ...n.data, title: title || undefined } } as AppNode) : n,
        ),
      })
    },

    setGroupMode: (id, mode) => {
      const group = get().nodes.find((n) => n.id === id)
      if (group?.type !== 'group' || group.data.mode === mode) return
      remember()
      commit({ nodes: get().nodes.map((n) => (n.id === id && n.type === 'group' ? { ...n, data: { ...n.data, mode } } : n)) })
    },

    setHyperparam: (key, value) => {
      if (get().hyperparams[key] === value) return
      // Typed numbers coalesce into one step; a toggle (tie_embeddings) is one step per click.
      remember(typeof value === 'boolean' ? undefined : `hp:${key}`)
      commit({ hyperparams: { ...get().hyperparams, [key]: value } })
    },

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
      remember()

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
      remember()
      commit(docToState(doc))
      set({ portPopover: null, quickAdd: null })
    },
    resetCanvas: () => get().loadDocument(cs336Document()),
    setDrawerOpen: (open) => set({ drawerOpen: open }),
    selectOnly: (id, openDrawer = true) => {
      get().selectNodes([id])
      if (openDrawer) set({ drawerOpen: true })
    },
    selectNodes: (ids, edgeIds = []) => {
      const { nodes, edges } = get()
      const want = new Set([...ids, ...edgeIds])
      const toggle = <T extends { id: string; selected?: boolean }>(xs: T[]) =>
        xs.some((x) => !!x.selected !== want.has(x.id)) ? xs.map((x) => (!!x.selected !== want.has(x.id) ? { ...x, selected: want.has(x.id) } : x)) : xs
      commit({ nodes: toggle(nodes), edges: toggle(edges) })
    },
    undo: () => restore(undo),
    redo: () => restore(redo),
    clearHistory: () => set({ history: emptyHistory() }),
    openQuickAdd: (state) => set({ quickAdd: state, portPopover: null }),
    closeQuickAdd: () => set({ quickAdd: null }),
    setHyperparamsOpen: (open) => set({ hyperparamsOpen: open }),
    toggleSection: (key) => set((s) => ({ collapsedSections: { ...s.collapsedSections, [key]: !s.collapsedSections[key] } })),
    setAnalysisOpen: (open) => set({ analysisOpen: open }),
    openAnalysis: (tab) => set({ analysisOpen: true, analysisTab: tab }),
    setHighlight: (key) => set({ highlight: key }),
    setMemoryMode: (mode) => set({ memoryMode: mode }),
    setCheckpointing: (on) => set({ checkpointing: on }),
    setGeneration: (patch) => set({ generation: { ...get().generation, ...patch } }),
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

/** Generation settings for the engine when the KV-cache view is active (Forward mode + generation on), else null. */
export const selectGenerationSettings = (s: CanvasState): GenerationSettings | null => (s.memoryMode === 'forward' && s.generation.on ? s.generation : null)

/**
 * The memory estimate for the current graph + mode (memoised; engine/memory.ts). Use as a selector.
 * With generation on (Forward), it is the generation estimate: weights + buffers + KV cache + one decode step.
 */
export const selectMemory = (s: CanvasState) => memoryFor(s.inference, s.hyperparams, s.memoryMode, s.checkpointing, selectGenerationSettings(s))

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
