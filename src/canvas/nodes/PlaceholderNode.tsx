import type { NodeProps } from '@xyflow/react'
import { Box } from 'lucide-react'
import type { PlaceholderNode as PlaceholderNodeType } from '../types'
import { Port } from '../Port'

/** Temporary Phase-1 part with one input and one output, to demo connecting. */
export function PlaceholderNode({ data, selected }: NodeProps<PlaceholderNodeType>) {
  return (
    <div
      className={`flex min-w-[180px] items-center gap-2 rounded-lg border bg-white px-3 py-3 shadow-sm transition-shadow ${
        selected ? 'border-indigo-500 ring-2 ring-indigo-200' : 'border-slate-300 hover:shadow-md'
      }`}
    >
      <Port kind="in" id="in" label="input" />
      <div className="flex h-8 w-8 items-center justify-center rounded-md bg-slate-100 text-slate-500">
        <Box size={16} />
      </div>
      <div className="leading-tight">
        <div className="text-sm font-medium text-slate-800">{data.label}</div>
        <div className="text-[11px] text-slate-400">shape: —</div>
      </div>
      <Port kind="out" id="out" label="output" />
    </div>
  )
}
