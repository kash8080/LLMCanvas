import { NodeResizer, useStore, type NodeProps } from '@xyflow/react'
import { useState } from 'react'
import { useCanvasStore } from '../../store/useCanvasStore'
import { frameTitleFontSize } from '../frames'
import type { FrameNode as FrameNodeType } from '../types'

/**
 * A user-made frame (src/canvas/frames.ts): a tinted, bordered rectangle drawn behind everything,
 * with its title just above the top-left corner (double-click to rename). The title keeps a readable
 * size on screen when zoomed out, so frames label regions of the overview. No ports.
 */
export function FrameNode({ id, data, selected, width }: NodeProps<FrameNodeType>) {
  const zoom = useStore((s) => s.transform[2])
  const setTitle = useCanvasStore((s) => s.setTitle)
  const [editing, setEditing] = useState(false)
  const title = data.title || 'Frame'
  const fontSize = frameTitleFontSize(zoom, width ?? 480, title)

  return (
    <>
      <NodeResizer isVisible={selected} minWidth={120} minHeight={80} color="#6366f1" />
      <div
        className={`h-full w-full rounded-xl border-2 ${selected ? 'ring-4 ring-indigo-200' : ''}`}
        style={{ background: data.bgColor, borderColor: data.borderColor }}
      />
      <div
        className="absolute bottom-full left-0 flex max-w-full items-center gap-[0.4em] pb-[0.35em] pl-[0.2em] leading-tight"
        style={{ fontSize }}
        onDoubleClick={() => setEditing(true)}
        title="Double-click to rename"
      >
        <span className="h-[0.8em] w-[0.8em] shrink-0 rounded-[0.2em]" style={{ background: data.borderColor }} />
        {editing ? (
          <input
            autoFocus
            value={data.title ?? ''}
            placeholder="Frame"
            onFocus={(e) => e.currentTarget.select()}
            onChange={(e) => setTitle(id, e.target.value)}
            onBlur={() => setEditing(false)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === 'Escape') e.currentTarget.blur()
            }}
            className="nodrag nopan min-w-[6em] rounded-[0.2em] border border-indigo-400 bg-white px-[0.2em] font-semibold text-slate-700 outline-none"
            style={{ width: `${Math.max(title.length, 6) + 2}ch` }}
          />
        ) : (
          <span className="truncate font-semibold whitespace-nowrap text-slate-600 select-none">{title}</span>
        )}
      </div>
    </>
  )
}
