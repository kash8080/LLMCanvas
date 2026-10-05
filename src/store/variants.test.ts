// Phase 7c through the store: swapping a block's SwiGLU for a non-gated FFN, and the weight-tying toggle (undoable).
import { beforeEach, describe, expect, it } from 'vitest'
import { connectionProblem } from '../canvas/connect'
import { useCanvasStore } from './useCanvasStore'

const s = () => useCanvasStore.getState()
const total = () => s().inference.params.total

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

describe('swap Block 1’s SwiGLU for a non-gated FFN', () => {
  it('delete → quick-add the FFN group inside the block (connected from ln2) → wire it to add2', async () => {
    const swiglu = s().nodes.filter((n) => n.id === 'b1.ffn' || n.parentId === 'b1.ffn').map((n) => n.id)
    deleteNodes((id) => swiglu.includes(id))
    await Promise.resolve()
    // ln2 no longer feeds the output, so its 512 gains are "unconnected" for now.
    expect(total()).toBe(16_468_480 - 2_064_384 - 512)
    expect(s().inference.params.unconnectedIds).toEqual(['b1.ln2'])

    s().addNode('group:ffn', { x: 300, y: 1200 }, { parentId: 'b1', connectFrom: { nodeId: 'b1.ln2', handleId: 'out', type: 'source' } })
    const ffn = s().nodes.find((n) => n.selected)!
    expect(ffn).toMatchObject({ type: 'group', parentId: 'b1', extent: 'parent' })
    expect(s().edges.some((e) => e.source === 'b1.ln2' && e.target === ffn.id && e.targetHandle === 'in')).toBe(true)
    // Its parts are children of the new group (parents before children).
    const idx = (id: string) => s().nodes.findIndex((n) => n.id === id)
    expect(idx('b1')).toBeLessThan(idx(ffn.id))
    expect(idx(ffn.id)).toBeLessThan(idx(`${ffn.id}.w1`))

    const out = { source: ffn.id, sourceHandle: 'out', target: 'b1.add2', targetHandle: 'b' }
    expect(connectionProblem(s().nodes, s().edges, out)).toBeNull()
    s().onConnect(out)
    expect(total()).toBe(16_468_480 - 2_064_384 + 1_376_256)
    expect(s().inference.groups.b1.status).toBe('ok')
    expect(s().inference.params.formula.notes[0]).toMatch(/non-gated FFN/)

    // Each step is undoable: connect, add, delete.
    s().undo()
    s().undo()
    s().undo()
    expect(total()).toBe(16_468_480)
    expect(s().nodes.some((n) => n.id === 'b1.ffn')).toBe(true)
  })
})

describe('weight tying toggle', () => {
  it('is an undoable hyperparam (one step per click)', () => {
    s().setHyperparam('tie_embeddings', true)
    expect(total()).toBe(11_348_480)
    s().setHyperparam('tie_embeddings', false)
    s().setHyperparam('tie_embeddings', true)
    expect(s().history.past).toHaveLength(3)
    s().undo()
    expect(total()).toBe(16_468_480)
    s().undo()
    expect(total()).toBe(11_348_480)
    s().undo()
    expect(s().hyperparams.tie_embeddings).toBe(false)
    expect(total()).toBe(16_468_480)
    s().redo()
    expect(total()).toBe(11_348_480)
  })

  it('Reset brings back the untied CS336 default', () => {
    s().setHyperparam('tie_embeddings', true)
    s().resetCanvas()
    expect(s().hyperparams.tie_embeddings).toBe(false)
    expect(total()).toBe(16_468_480)
  })
})
