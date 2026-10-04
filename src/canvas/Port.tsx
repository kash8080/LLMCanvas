import { Handle, Position } from '@xyflow/react'

/**
 * A node port. Inputs sit on top (hollow blue circle), outputs on the bottom
 * (filled green circle) so the graph reads top -> bottom (PLAN.md §2.4).
 */
export function Port({ kind, id, label, style }: { kind: 'in' | 'out'; id: string; label?: string; style?: React.CSSProperties }) {
  const isInput = kind === 'in'
  return (
    <Handle
      type={isInput ? 'target' : 'source'}
      position={isInput ? Position.Top : Position.Bottom}
      id={id}
      className={isInput ? 'port port-in' : 'port port-out'}
      title={label ?? (isInput ? `input: ${id}` : `output: ${id}`)}
      style={style}
    />
  )
}
