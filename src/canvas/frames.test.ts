import type { NodeChange } from '@xyflow/react'
import { describe, expect, it } from 'vitest'
import { FRAME_TITLE_PX, FRAME_Z, FRAME_Z_STEP, frameBoxAround, frameFollowChanges, frameTitleFontSize, isInside, itemsInFrame, nodeBox, topLevelIds, withFrameLayers } from './frames'
import { createFrameNode, createNode, createPartNode } from './nodeFactory'
import type { AppNode } from './types'

const frame = (id: string, x: number, y: number, width: number, height: number): AppNode => ({ ...createFrameNode({ x, y, width, height }), id })
const part = (id: string, x: number, y: number, parentId?: string): AppNode => ({
  ...createPartNode('linear', { x, y }, id),
  measured: { width: 176, height: 60 },
  ...(parentId ? { parentId } : {}),
})
const group = (id: string, x: number, y: number, width: number, height: number): AppNode => ({
  id,
  type: 'group',
  position: { x, y },
  width,
  height,
  data: { groupType: 'transformer_block', mode: 'auto' },
})

describe('frame containment', () => {
  it('uses declared, then measured, then default sizes', () => {
    expect(nodeBox(frame('f', 1, 2, 300, 200))).toEqual({ x: 1, y: 2, width: 300, height: 200 })
    expect(nodeBox(part('p', 0, 0))).toEqual({ x: 0, y: 0, width: 176, height: 60 })
    expect(nodeBox(createNode('sticky', { x: 5, y: 5 }))).toMatchObject({ width: 200, height: 160 })
    const { measured: _m, ...unmeasured } = part('q', 0, 0) as AppNode & { measured?: unknown }
    expect(nodeBox(unmeasured as AppNode)).toMatchObject({ width: 176, height: 64 })
  })

  it('counts only top-level items fully inside the frame (touching the edge is inside)', () => {
    expect(isInside({ x: 0, y: 0, width: 10, height: 10 }, { x: 0, y: 0, width: 10, height: 10 })).toBe(true)
    const nodes = [
      frame('f', 0, 0, 1000, 1000),
      part('in', 100, 100),
      part('edge', 824, 940), // right / bottom exactly on the frame edge
      part('half', 900, 100), // sticks out on the right
      part('out', 2000, 0),
      group('g', 200, 200, 400, 400),
      part('child', 10, 10, 'g'), // moves with its group, not counted itself
      frame('inner', 50, 700, 200, 200), // nested frame comes along
      frame('big', -10, -10, 2000, 2000), // bigger frame around it: not inside
    ]
    expect(itemsInFrame(nodes, 'f')).toEqual(['in', 'edge', 'g', 'inner'])
    expect(itemsInFrame(nodes, 'big')).toEqual(['f', 'in', 'edge', 'half', 'g', 'inner'])
    expect(itemsInFrame(nodes, 'in')).toEqual([]) // not a frame
  })

  it('maps a selection to top-level items and pads their bounds', () => {
    const nodes = [group('g', 0, 0, 888, 2124), part('child', 10, 10, 'g'), part('a', 1000, 100), part('b', 1000, 300)]
    expect(topLevelIds(nodes, ['child', 'a', 'g'])).toEqual(['g', 'a'])
    expect(frameBoxAround(nodes, ['g', 'a'])).toEqual({ x: -40, y: -40, width: 1176 + 80, height: 2124 + 80 })
    expect(frameBoxAround(nodes, ['a', 'b'], 10)).toEqual({ x: 990, y: 90, width: 196, height: 280 })
    expect(frameBoxAround(nodes, [])).toBeNull()
  })
})

describe('frames follow drags', () => {
  const nodes = [frame('f', 0, 0, 1000, 1000), part('in', 100, 100), group('g', 200, 200, 400, 400), part('out', 2000, 0)]
  const at = (id: string, x: number, y: number, dragging?: boolean): NodeChange<AppNode> => ({ type: 'position', id, position: { x, y }, ...(dragging === undefined ? {} : { dragging }) })

  it('moves the contents captured at drag start by the same delta', () => {
    const captured = new Map<string, string[]>()
    expect(frameFollowChanges([at('f', 10, 5, true)], nodes, captured)).toEqual([at('in', 110, 105), at('g', 210, 205)])
    expect(captured.get('f')).toEqual(['in', 'g'])
    // An item that is now (mid-drag) inside the frame isn't picked up: the capture is fixed.
    const moved = nodes.map((n) => (n.id === 'f' ? { ...n, position: { x: 1500, y: 0 } } : n))
    expect(frameFollowChanges([at('f', 1510, 0, true)], moved, captured).map((c) => ('id' in c ? c.id : ''))).toEqual(['in', 'g'])
    // Drag end at the same position: nothing moves, capture dropped.
    expect(frameFollowChanges([at('f', 1500, 0, false)], moved, captured)).toEqual([])
    expect(captured.size).toBe(0)
  })

  it('does not move items that move anyway, and ignores resize / non-frame changes', () => {
    const captured = new Map<string, string[]>()
    expect(frameFollowChanges([at('f', 10, 0, true), at('in', 110, 100, true)], nodes, captured)).toEqual([at('g', 210, 200)])
    captured.clear()
    // NodeResizer position changes (dragging the left / top edge) carry no `dragging` flag.
    expect(frameFollowChanges([at('f', -50, 0)], nodes, captured)).toEqual([])
    expect(frameFollowChanges([at('in', 0, 0, true)], nodes, captured)).toEqual([])
    // Arrow-key nudge (dragging: false, no capture): contents computed on the spot.
    expect(frameFollowChanges([at('f', 5, 0, false)], nodes, captured)).toEqual([at('in', 105, 100), at('g', 205, 200)])
  })
})

describe('frame display', () => {
  it('puts frames below everything, bigger frames lowest, keeping identity otherwise', () => {
    const nodes = [part('p', 0, 0), frame('small', 0, 0, 100, 100), frame('large', 0, 0, 1000, 1000)]
    const out = withFrameLayers(nodes)
    expect(out[0]).toBe(nodes[0])
    expect(out.find((n) => n.id === 'large')!.zIndex).toBe(FRAME_Z)
    expect(out.find((n) => n.id === 'small')!.zIndex).toBe(FRAME_Z + FRAME_Z_STEP)
    expect(FRAME_Z + FRAME_Z_STEP).toBeGreaterThan(FRAME_Z + 1000) // selected large frame stays below the small one
    expect(withFrameLayers(out)).toEqual(out)
    expect(withFrameLayers(out)[1]).toBe(out[1])
  })

  it('keeps the title readable when zoomed out, but no wider than the frame', () => {
    expect(frameTitleFontSize(1, 480, 'Frame')).toBe(FRAME_TITLE_PX)
    expect(frameTitleFontSize(2, 480, 'Frame')).toBe(FRAME_TITLE_PX) // never smaller than at zoom 1
    expect(frameTitleFontSize(0.25, 2000, 'Encoder')).toBeCloseTo(FRAME_TITLE_PX * 4)
    expect(frameTitleFontSize(0.05, 1000, 'Two blocks')).toBe(1000 / (10 * 0.6 + 2))
  })
})
