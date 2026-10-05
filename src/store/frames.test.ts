import { beforeEach, describe, expect, it } from 'vitest'
import { itemsInFrame, nodeBox } from '../canvas/frames'
import type { AppNode, FrameNode } from '../canvas/types'
import { selectMemory, useCanvasStore } from './useCanvasStore'
import { parseDocument, toDocument } from './persistence'

const s = () => useCanvasStore.getState()
const node = (id: string) => s().nodes.find((n) => n.id === id)!
const frames = () => s().nodes.filter((n): n is FrameNode => n.type === 'frame')

beforeEach(() => {
  s().resetCanvas()
  s().clearHistory()
})

/** What React Flow sends while dragging node `id` by (dx, dy) in `steps` steps, then the drag end. */
function drag(id: string, dx: number, dy: number, steps = 4) {
  const start = node(id).position
  for (let i = 1; i <= steps; i++)
    s().onNodesChange([{ type: 'position', id, position: { x: start.x + (dx * i) / steps, y: start.y + (dy * i) / steps }, dragging: true }])
  s().onNodesChange([{ type: 'position', id, position: { x: start.x + dx, y: start.y + dy }, dragging: false }])
}

describe('frames in the store', () => {
  it('frames a selection (a Transformer Block + a part) and frames change nothing in the model', () => {
    const params = s().inference.params.total
    const memory = selectMemory(s()).total
    const id = s().frameSelection(['b1.q_proj', 'embed'])! // a part inside Block 1 counts as Block 1
    const f = node(id) as FrameNode
    expect(f.type).toBe('frame')
    expect(f.selected).toBe(true)
    expect(f.parentId).toBeUndefined()
    expect(itemsInFrame(s().nodes, id).sort()).toEqual(['b1', 'embed'])
    expect(nodeBox(f).width).toBeGreaterThan(nodeBox(node('b1')).width)
    expect(s().nodes.filter((n) => n.selected).map((n) => n.id)).toEqual([id])
    expect(s().inference.params.total).toBe(params)
    expect(selectMemory(s()).total).toBe(memory)
    expect(s().history.past).toHaveLength(1)
    s().undo()
    expect(frames()).toHaveLength(0)
  })

  it('dragging a frame moves what is inside (one undo step); edges stay', () => {
    const id = s().frameSelection(['b1', 'embed'])!
    const edges = s().edges
    const b1 = node('b1').position
    const embed = node('embed').position
    const b2 = node('b2').position
    const childRel = node('b1.ln1').position
    drag(id, 300, 120)
    expect(node('b1').position).toEqual({ x: b1.x + 300, y: b1.y + 120 })
    expect(node('embed').position).toEqual({ x: embed.x + 300, y: embed.y + 120 })
    expect(node('b2').position).toEqual(b2) // not inside
    expect(node('b1.ln1').position).toEqual(childRel) // children are relative to their group
    expect(s().edges).toBe(edges)
    expect(s().inference.params.total).toBe(16_468_480)
    expect(s().history.past).toHaveLength(2) // frame selection + one drag
    s().undo()
    expect(node('b1').position).toEqual(b1)
    expect(node('embed').position).toEqual(embed)
    s().redo()
    expect(node('b1').position).toEqual({ x: b1.x + 300, y: b1.y + 120 })
  })

  it('resizing does not move the contents; rename, recolour, delete-frame-only are undoable', () => {
    const id = s().frameSelection(['embed'])!
    const embed = node('embed').position
    const f = node(id)
    // NodeResizer dragging the left edge: position (no dragging flag) + dimensions with resizing.
    s().onNodesChange([
      { type: 'position', id, position: { x: f.position.x - 100, y: f.position.y } },
      { type: 'dimensions', id, dimensions: { width: (f.width ?? 0) + 100, height: f.height ?? 0 }, resizing: true, setAttributes: true },
    ])
    s().onNodesChange([{ type: 'dimensions', id, dimensions: { width: (f.width ?? 0) + 100, height: f.height ?? 0 }, resizing: false }])
    expect(node('embed').position).toEqual(embed)
    expect(node(id).width).toBe((f.width ?? 0) + 100)

    for (const t of ['E', 'Em', 'Embedding']) s().setTitle(id, t)
    s().updateFrame(id, { bgColor: '#eff6ff' })
    expect((node(id) as FrameNode).data).toMatchObject({ title: 'Embedding', bgColor: '#eff6ff' })
    expect(s().history.past).toHaveLength(4) // frame, resize, rename, colour

    // Delete key on the frame: only the frame goes (no parenting → contents stay).
    s().onNodesChange([{ type: 'remove', id }])
    expect(frames()).toHaveLength(0)
    expect(node('embed')).toBeDefined()
    s().undo()
    expect((node(id) as FrameNode).data.title).toBe('Embedding')
    s().undo()
    expect((node(id) as FrameNode).data.bgColor).toBe('#f8fafc')
    s().undo()
    expect((node(id) as FrameNode).data.title).toBeUndefined()
  })

  it('palette / quick-add frames are top-level even when asked to go inside a group', () => {
    s().addNode('frame', { x: 10, y: 20 }, { parentId: 'b1' })
    const f = frames()[0]
    expect(f.parentId).toBeUndefined()
    expect(f).toMatchObject({ position: { x: 10, y: 20 }, width: 480, height: 320 })
  })

  it('duplicating a frame copies just the frame', () => {
    const id = s().frameSelection(['embed'])!
    const count = s().nodes.length
    s().duplicateSelection()
    expect(s().nodes).toHaveLength(count + 1)
    const copy = s().nodes.at(-1)!
    expect(copy).toMatchObject({ type: 'frame', selected: true, width: node(id).width })
    expect(copy.id).not.toBe(id)
  })

  it('round-trips frames through the saved document; frames must be top-level', () => {
    const id = s().frameSelection(['b1'])!
    s().setTitle(id, 'Layer 1')
    const doc = toDocument(s().nodes, s().edges, s().hyperparams)
    const parsed = parseDocument(JSON.parse(JSON.stringify(doc)))
    expect(parsed).toEqual(doc)
    const saved = parsed.nodes.find((n) => n.id === id)!
    expect(saved).toMatchObject({ type: 'frame', data: { title: 'Layer 1', bgColor: '#f8fafc', borderColor: '#94a3b8' } })
    s().loadDocument(parsed)
    expect(s().inference.params.total).toBe(16_468_480)

    const inGroup = { ...doc, nodes: doc.nodes.map((n) => (n.id === id ? { ...n, parentId: 'b2' } : n)) }
    // Parents must come first anyway; put the frame at the end so only the frame rule can fail.
    inGroup.nodes = [...inGroup.nodes.filter((n) => n.id !== id), inGroup.nodes.find((n) => n.id === id)!]
    expect(() => parseDocument(inGroup)).toThrow("can't be inside another node")
    const noSize = { ...doc, nodes: doc.nodes.map((n) => (n.id === id ? ({ ...n, width: undefined } as AppNode) : n)) }
    expect(() => parseDocument(JSON.parse(JSON.stringify(noSize)))).toThrow('has no size')
  })
})
