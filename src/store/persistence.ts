// Pure (no React) save/load helpers: document format, validation, localStorage.
import { isNodeKind, type AppEdge, type AppNode } from '../canvas/types'
import { DEFAULT_HYPERPARAMS, FLOAT_DTYPES, HYPERPARAM_INFO } from '../engine/hyperparams'
import type { Hyperparams } from '../engine/types'
import { getNodeDef } from '../nodes/registry'

export const STORAGE_KEY = 'llm-canvas:v1'
/** v1 = Phase 1 (placeholder parts, no hyperparams). v2 = Phase 2 (registry parts + hyperparams). */
export const DOC_VERSION = 2

/** A saved node: only the fields worth persisting (no selection / measured state). */
export type SavedNode = Pick<AppNode, 'id' | 'type' | 'position' | 'data' | 'width' | 'height'>
export type SavedEdge = Pick<AppEdge, 'id' | 'source' | 'target' | 'sourceHandle' | 'targetHandle'>

export interface CanvasDocument {
  app: 'llm-canvas'
  version: typeof DOC_VERSION
  hyperparams: Hyperparams
  nodes: SavedNode[]
  edges: SavedEdge[]
}

export function toDocument(nodes: AppNode[], edges: AppEdge[], hyperparams: Hyperparams): CanvasDocument {
  return {
    app: 'llm-canvas',
    version: DOC_VERSION,
    hyperparams: { ...hyperparams },
    nodes: nodes.map(({ id, type, position, data, width, height }) => ({
      id,
      type,
      position: { x: position.x, y: position.y },
      data,
      ...(width != null ? { width } : {}),
      ...(height != null ? { height } : {}),
    })) as SavedNode[],
    edges: edges.map(({ id, source, target, sourceHandle, targetHandle }) => ({
      id,
      source,
      target,
      sourceHandle: sourceHandle ?? null,
      targetHandle: targetHandle ?? null,
    })),
  }
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null
const isStr = (v: unknown): v is string => typeof v === 'string'
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

function parseHyperparams(raw: unknown): Hyperparams {
  const hp: Hyperparams = { ...DEFAULT_HYPERPARAMS }
  if (!isObj(raw)) return hp
  for (const { key } of HYPERPARAM_INFO) if (isNum(raw[key]) && raw[key] > 0) hp[key] = raw[key]
  if (FLOAT_DTYPES.includes(raw.dtype as Hyperparams['dtype'])) hp.dtype = raw.dtype as Hyperparams['dtype']
  return hp
}

/** Validates untrusted JSON (import / localStorage). Throws a readable Error if invalid. */
export function parseDocument(raw: unknown): CanvasDocument {
  if (!isObj(raw) || raw.app !== 'llm-canvas') throw new Error('Not an LLM Canvas file.')
  if (raw.version !== DOC_VERSION)
    throw new Error(`Unsupported file version: ${String(raw.version)} (this app reads version ${DOC_VERSION}).`)
  if (!Array.isArray(raw.nodes) || !Array.isArray(raw.edges)) throw new Error('File is missing nodes/edges.')

  const ids = new Set<string>()
  for (const n of raw.nodes) {
    if (!isObj(n) || !isStr(n.id)) throw new Error('A node is missing its id.')
    if (!isNodeKind(n.type)) throw new Error(`Unknown node type "${String(n.type)}".`)
    if (!isObj(n.position) || !isNum(n.position.x) || !isNum(n.position.y))
      throw new Error(`Node "${n.id}" has an invalid position.`)
    if (!isObj(n.data)) throw new Error(`Node "${n.id}" has no data.`)
    if (n.type === 'part') {
      if (!isStr(n.data.partType) || !getNodeDef(n.data.partType))
        throw new Error(`Node "${n.id}" has an unknown part type "${String(n.data.partType)}".`)
      if (!isObj(n.data.params)) throw new Error(`Node "${n.id}" has no params.`)
    }
    ids.add(n.id)
  }
  for (const e of raw.edges) {
    if (!isObj(e) || !isStr(e.id) || !isStr(e.source) || !isStr(e.target)) throw new Error('An edge is malformed.')
    if (!ids.has(e.source) || !ids.has(e.target)) throw new Error(`Edge "${e.id}" points to a missing node.`)
  }
  return { ...(raw as unknown as CanvasDocument), hyperparams: parseHyperparams(raw.hyperparams) }
}

/** Saved canvas, or null if none / unreadable / older format (caller falls back to the default graph). */
export function loadFromStorage(): CanvasDocument | null {
  try {
    const text = localStorage.getItem(STORAGE_KEY)
    return text ? parseDocument(JSON.parse(text)) : null
  } catch (err) {
    console.warn('Ignoring invalid saved canvas:', err)
    return null
  }
}

export function saveToStorage(doc: CanvasDocument): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(doc))
  } catch (err) {
    console.warn('Could not save canvas:', err)
  }
}
