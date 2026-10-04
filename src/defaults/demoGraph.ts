import type { CanvasDocument } from '../store/persistence'

/**
 * Tiny demo shown on first load / reset in Phase 1.
 * Phase 2+ replaces this with the CS336 TransformerLM graph (cs336Graph.ts).
 */
export function demoDocument(): CanvasDocument {
  return {
    app: 'llm-canvas',
    version: 1,
    nodes: [
      {
        id: 'welcome',
        type: 'sticky',
        position: { x: -320, y: -40 },
        width: 240,
        height: 170,
        data: {
          text: 'Welcome to LLM Canvas!\n\nDrag parts from the palette, connect an output (green, bottom) to an input (blue, top). Double-click notes to edit.',
          bgColor: '#fef08a',
          textColor: '#1f2937',
          fontSize: 14,
        },
      },
      { id: 'part-a', type: 'placeholder', position: { x: 0, y: 0 }, data: { label: 'Part A' } },
      { id: 'part-b', type: 'placeholder', position: { x: 0, y: 160 }, data: { label: 'Part B' } },
    ],
    edges: [{ id: 'e-a-b', source: 'part-a', sourceHandle: 'out', target: 'part-b', targetHandle: 'in' }],
  }
}
