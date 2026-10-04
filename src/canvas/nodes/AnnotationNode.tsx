import { NodeResizer, type NodeProps } from '@xyflow/react'
import { useState } from 'react'
import { useCanvasStore } from '../../store/useCanvasStore'
import type { StickyNode, TextBoxNode } from '../types'

/** Shared component for sticky notes and text boxes: resizable, double-click to edit. */
function AnnotationNode({ id, data, selected, variant }: NodeProps<StickyNode | TextBoxNode> & { variant: 'sticky' | 'textbox' }) {
  const updateAnnotation = useCanvasStore((s) => s.updateAnnotation)
  const [editing, setEditing] = useState(false)
  const isSticky = variant === 'sticky'
  const transparent = data.bgColor === 'transparent'

  const textStyle: React.CSSProperties = { color: data.textColor, fontSize: data.fontSize, lineHeight: 1.35 }

  return (
    <>
      <NodeResizer isVisible={selected} minWidth={60} minHeight={30} color="#6366f1" />
      <div
        onDoubleClick={() => setEditing(true)}
        className={`h-full w-full overflow-hidden ${isSticky ? 'rounded-sm p-3 shadow-md' : 'rounded p-1.5'} ${
          transparent && !selected ? 'border border-dashed border-transparent hover:border-slate-300' : ''
        }`}
        style={{ backgroundColor: data.bgColor }}
      >
        {editing ? (
          <textarea
            autoFocus
            onFocus={(e) => e.currentTarget.setSelectionRange(e.currentTarget.value.length, e.currentTarget.value.length)}
            className="nodrag nopan nowheel h-full w-full resize-none bg-transparent outline-none"
            style={textStyle}
            value={data.text}
            onChange={(e) => updateAnnotation(id, { text: e.target.value })}
            onBlur={() => setEditing(false)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') (e.target as HTMLTextAreaElement).blur()
            }}
          />
        ) : (
          <div className="h-full w-full whitespace-pre-wrap break-words select-none" style={textStyle}>
            {data.text || <span className="opacity-40">{isSticky ? 'Double-click to write…' : 'Double-click to edit text'}</span>}
          </div>
        )}
      </div>
    </>
  )
}

export function StickyNoteNode(props: NodeProps<StickyNode>) {
  return <AnnotationNode {...props} variant="sticky" />
}

export function TextBoxNode(props: NodeProps<TextBoxNode>) {
  return <AnnotationNode {...props} variant="textbox" />
}
