import { describe, expect, it } from 'vitest'
import { cs336Document } from '../defaults/cs336Graph'
import { connectionProblem, connectionToBody } from './connect'
import type { AppNode } from './types'

const doc = cs336Document()
const nodes = doc.nodes as AppNode[]
const edges = doc.edges

describe('connection rules', () => {
  it('allows output → input, including replacing an occupied input', () => {
    expect(connectionProblem(nodes, edges, { source: 'b1.q_proj', sourceHandle: 'out', target: 'b1.split_k', targetHandle: 'in' })).toBeNull()
  })

  it('group outer ports work like part ports; inner parts are off-limits from outside', () => {
    // top-level part → group outer input (replacing the existing edge)
    expect(connectionProblem(nodes, edges, { source: 'embed', sourceHandle: 'out', target: 'b2', targetHandle: 'in' })).toBeNull()
    // top-level part → a part inside a group: refused
    expect(connectionProblem(nodes, edges, { source: 'embed', sourceHandle: 'out', target: 'b1.ln1', targetHandle: 'in' })).toMatch(/group/)
    // siblings inside a block, incl. a nested group's outer port
    expect(connectionProblem(nodes, edges, { source: 'b1.add1', sourceHandle: 'out', target: 'b1.ffn', targetHandle: 'in' })).toBeNull()
    // proxies only expose their inner side
    expect(connectionProblem(nodes, edges, { source: 'embed', sourceHandle: 'out', target: 'b1.in', targetHandle: 'in' })).toMatch(/output/)
    // a cycle through groups: Block 2's output back into Block 1's input
    expect(connectionProblem(nodes, edges, { source: 'b2', sourceHandle: 'out', target: 'b1', targetHandle: 'in' })).toMatch(/cycle/)
    // body drop on a group frame picks its outer input
    expect(connectionToBody(nodes, edges, { nodeId: 'embed', handleId: 'out', type: 'source' }, 'b2')).toEqual({
      connection: { source: 'embed', sourceHandle: 'out', target: 'b2', targetHandle: 'in' },
    })
  })

  it('refuses self, wrong direction and cycles', () => {
    expect(connectionProblem(nodes, edges, { source: 'embed', sourceHandle: 'out', target: 'embed', targetHandle: 'in' })).toMatch(/itself/)
    expect(connectionProblem(nodes, edges, { source: 'embed', sourceHandle: 'in', target: 'b1.ln1', targetHandle: 'out' })).toMatch(/output/)
    expect(connectionProblem(nodes, edges, { source: 'b1.add2', sourceHandle: 'out', target: 'b1.ln1', targetHandle: 'in' })).toMatch(/cycle/)
  })

  it('a drop on a node body picks the input (replacing a single occupied one)', () => {
    const r = connectionToBody(nodes, edges, { nodeId: 'b1.k_proj', handleId: 'out', type: 'source' }, 'b1.split_k')
    expect(r).toEqual({ connection: { source: 'b1.k_proj', sourceHandle: 'out', target: 'b1.split_k', targetHandle: 'in' } })
    const full = connectionToBody(nodes, edges, { nodeId: 'embed', handleId: 'out', type: 'source' }, 'b1.add1')
    expect(full).toHaveProperty('problem')
  })
})
