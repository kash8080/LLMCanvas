import { beforeEach, describe, expect, it } from 'vitest'
import { numLayers } from '../canvas/groupTemplates'
import { useCanvasStore } from './useCanvasStore'

const s = () => useCanvasStore.getState()
const total = () => s().inference.params.total
const node = (id: string) => s().nodes.find((n) => n.id === id)

beforeEach(() => {
  s().resetCanvas()
  s().clearHistory()
})

/** What React Flow does on Delete: remove the nodes (+ children) and every edge touching them, in two change calls. */
function deleteNodes(pred: (id: string) => boolean) {
  const ids = new Set(s().nodes.filter((n) => pred(n.id)).map((n) => n.id))
  const edges = s().edges.filter((e) => ids.has(e.source) || ids.has(e.target))
  s().onEdgesChange(edges.map((e) => ({ type: 'remove' as const, id: e.id })))
  s().onNodesChange([...ids].map((id) => ({ type: 'remove' as const, id })))
}

describe('undo / redo in the store', () => {
  it('walks a mixed sequence back and forth, with inference following', async () => {
    const start = total()
    expect(start).toBe(16_468_480)

    // 1. add a Linear
    s().addNode('part:linear', { x: 1500, y: 0 })
    const lin = s().nodes.find((n) => n.selected)!
    // 2. connect it from ln_final (it does not feed Logits → unconnected)
    s().onConnect({ source: 'ln_final', sourceHandle: 'out', target: lin.id, targetHandle: 'in' })
    expect(s().edges.some((e) => e.target === lin.id)).toBe(true)
    // 3. edit a param: typing "1024" commits 1, 10, 102, 1024 — one undo step
    for (const v of [1, 10, 102, 1024]) s().setPartParam(lin.id, 'out_features', { value: v })
    expect(s().inference.params.unconnected).toBe(512 * 1024)
    // 4. move it (one drag = many position changes = one step)
    for (let i = 1; i <= 5; i++) s().onNodesChange([{ type: 'position', id: lin.id, position: { x: 1500 + i * 10, y: 0 }, dragging: true }])
    s().onNodesChange([{ type: 'position', id: lin.id, position: { x: 1550, y: 0 }, dragging: false }])
    expect(node(lin.id)!.position.x).toBe(1550)
    // 5. delete Block 2 (frame + children + edges = one step)
    await Promise.resolve()
    deleteNodes((id) => id === 'b2' || id.startsWith('b2.'))
    expect(numLayers(s().nodes)).toBe(1)
    expect(s().history.past).toHaveLength(5)

    // Undo everything, checking each step.
    s().undo()
    expect(numLayers(s().nodes)).toBe(2)
    expect(total()).toBe(start)
    expect(s().inference.groups.b2.params).toBe(3_113_984)
    s().undo()
    expect(node(lin.id)!.position.x).toBe(1500)
    s().undo()
    expect(s().inference.params.unconnected).toBe(512 * 512)
    s().undo()
    expect(s().edges.some((e) => e.target === lin.id)).toBe(false)
    s().undo()
    expect(node(lin.id)).toBeUndefined()
    expect(s().history.past).toHaveLength(0)
    s().undo() // nothing left: no-op
    expect(s().history.future).toHaveLength(5)

    // Redo everything.
    for (let i = 0; i < 5; i++) s().redo()
    expect(numLayers(s().nodes)).toBe(1)
    expect(node(lin.id)!.position.x).toBe(1550)
    expect(s().inference.nodes[lin.id].paramCount.total).toBe(512 * 1024)
  })

  it('coalesces typing in a title / hyperparam, but not separate fields', () => {
    for (const t of ['q', 'qu', 'que', 'quer', 'query']) s().setTitle('b1.q_proj', t)
    for (const v of [7, 76, 768]) s().setHyperparam('d_model', v)
    expect(s().history.past).toHaveLength(2)
    s().undo()
    expect(s().hyperparams.d_model).toBe(512)
    s().undo()
    const q = node('b1.q_proj')!
    expect(q.type === 'part' && q.data.title).toBe('q_proj')
  })

  it('records group mode changes, annotation edits, duplicate, reset; not selection or no-op changes', () => {
    s().selectOnly('b1')
    s().setGroupMode('b1', 'auto') // unchanged → nothing
    s().setGroupMode('b1', 'collapsed')
    s().duplicateSelection()
    expect(numLayers(s().nodes)).toBe(3)
    s().addNode('sticky', { x: 0, y: 0 })
    const note = s().nodes.find((n) => n.selected)!
    for (const t of ['h', 'hi']) s().updateAnnotation(note.id, { text: t })
    s().updateAnnotation(note.id, { bgColor: '#ffffff' })
    s().resetCanvas()
    expect(s().history.past).toHaveLength(6)
    s().undo() // reset
    expect(s().nodes.some((n) => n.id === note.id)).toBe(true)
    s().undo() // colour
    s().undo() // text
    const n = node(note.id)!
    expect(n.type === 'sticky' && n.data.text).toBe('')
    s().undo() // sticky
    s().undo() // duplicate
    expect(numLayers(s().nodes)).toBe(2)
    s().undo() // mode
    const b1 = node('b1')!
    expect(b1.type === 'group' && b1.data.mode).toBe('auto')
  })

  it('a resize is one step (recorded when it ends); measuring / selecting is not recorded', () => {
    s().addNode('sticky', { x: 0, y: 0 })
    const id = s().nodes.find((n) => n.selected)!.id
    s().clearHistory()
    s().onNodesChange([{ type: 'dimensions', id, dimensions: { width: 200, height: 160 } }]) // measured
    s().onNodesChange([{ type: 'select', id, selected: false }])
    expect(s().history.past).toHaveLength(0)
    for (const w of [220, 240, 260]) s().onNodesChange([{ type: 'dimensions', id, resizing: true, setAttributes: true, dimensions: { width: w, height: 160 } }])
    s().onNodesChange([{ type: 'dimensions', id, resizing: false, dimensions: { width: 260, height: 160 } }])
    expect(s().history.past).toHaveLength(1)
    expect(node(id)!.width).toBe(260)
    s().undo()
    expect(node(id)!.width).toBe(200)
  })

  it('a new edit after undo drops the redo steps', () => {
    s().setHyperparam('num_heads', 8)
    s().undo()
    expect(s().history.future).toHaveLength(1)
    s().setHyperparam('d_ff', 2048)
    expect(s().history.future).toHaveLength(0)
    s().redo()
    expect(s().hyperparams.d_ff).toBe(2048)
  })

  it('quick-add: creates the part and connects it in one step (from an output / from an input)', () => {
    s().addNode('part:rmsnorm', { x: 1500, y: 0 }, { connectFrom: { nodeId: 'ln_final', handleId: 'out', type: 'source' } })
    const norm = s().nodes.find((n) => n.selected)!
    expect(s().edges.some((e) => e.source === 'ln_final' && e.target === norm.id && e.targetHandle === 'in')).toBe(true)
    // From an input that is already connected: the new part's output replaces that edge.
    s().addNode('part:silu', { x: 1500, y: 300 }, { connectFrom: { nodeId: 'logits', handleId: 'in', type: 'target' } })
    const silu = s().nodes.find((n) => n.selected)!
    expect(s().edges.filter((e) => e.target === 'logits').map((e) => e.source)).toEqual([silu.id])
    expect(s().history.past).toHaveLength(2)
    s().undo()
    expect(s().edges.filter((e) => e.target === 'logits').map((e) => e.source)).toEqual(['lm_head'])
  })

  it('quick-add inside a group frame creates a child that connects to the inner part', () => {
    s().addNode('part:silu', { x: 10, y: 10 }, { parentId: 'b1.attn', connectFrom: { nodeId: 'b1.output_proj', handleId: 'out', type: 'source' } })
    const silu = s().nodes.find((n) => n.selected)!
    expect(silu.parentId).toBe('b1.attn')
    expect(s().edges.some((e) => e.source === 'b1.output_proj' && e.target === silu.id)).toBe(true)
  })
})
