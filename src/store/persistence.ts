// Pure (no React) save/load helpers: document format, validation, localStorage.
import { isNodeKind, type AppEdge, type AppNode } from '../canvas/types'

export const STORAGE_KEY = 'llm-canvas:v1'

/** A saved node: only the fields worth persisting (no selection / measured state). */
export type SavedNode = Pick<AppNode, 'id' | 'type' | 'position' | 'data' | 'width' | 'height'>
export type SavedEdge = Pick<AppEdge, 'id' | 'source' | 'target' | 'sourceHandle' | 'targetHandle'>

export interface CanvasDocument {
  app: 'llm-canvas'
  version: 1
  nodes: SavedNode[]
  edges: SavedEdge[]
}

export function toDocument(nodes: AppNode[], edges: AppEdge[]): CanvasDocument {
  return {
    app: 'llm-canvas',
    version: 1,
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

/** Validates untrusted JSON (import / localStorage). Throws a readable Error if invalid. */
export function parseDocument(raw: unknown): CanvasDocument {
  if (!isObj(raw) || raw.app !== 'llm-canvas') throw new Error('Not an LLM Canvas file.')
  if (raw.version !== 1) throw new Error(`Unsupported file version: ${String(raw.version)}`)
  if (!Array.isArray(raw.nodes) || !Array.isArray(raw.edges)) throw new Error('File is missing nodes/edges.')

  const ids = new Set<string>()
  for (const n of raw.nodes) {
    if (!isObj(n) || !isStr(n.id)) throw new Error('A node is missing its id.')
    if (!isNodeKind(n.type)) throw new Error(`Unknown node type "${String(n.type)}".`)
    if (!isObj(n.position) || !isNum(n.position.x) || !isNum(n.position.y))
      throw new Error(`Node "${n.id}" has an invalid position.`)
    if (!isObj(n.data)) throw new Error(`Node "${n.id}" has no data.`)
    ids.add(n.id)
  }
  for (const e of raw.edges) {
    if (!isObj(e) || !isStr(e.id) || !isStr(e.source) || !isStr(e.target)) throw new Error('An edge is malformed.')
    if (!ids.has(e.source) || !ids.has(e.target)) throw new Error(`Edge "${e.id}" points to a missing node.`)
  }
  return raw as unknown as CanvasDocument
}

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
