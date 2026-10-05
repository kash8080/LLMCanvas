// Drawer sections rendered from a part's / group's NodeDocs (src/nodes/*, src/nodes/groups.ts).
import { FileCode } from 'lucide-react'
import type { NodeDocs } from '../../engine/types'
import { Section } from './ui'

export function OverviewSection({ docs, title }: { docs: NodeDocs; title?: string }) {
  const role = title ? docs.roles?.[title] : undefined
  return (
    <Section id="overview" title="Overview">
      <p className="leading-relaxed break-words text-slate-600">{docs.overview}</p>
      {role && <p className="mt-2 rounded-md bg-indigo-50 px-2.5 py-1.5 text-xs leading-relaxed break-words text-indigo-800">{role}</p>}
    </Section>
  )
}

export function FormulaSection({ lines }: { lines?: string[] }) {
  if (!lines?.length) return null
  return (
    <Section id="formula" title="Formula">
      <Formula lines={lines} />
    </Section>
  )
}

/** Monospace formula block; long lines wrap. */
export function Formula({ lines }: { lines: string[] }) {
  return (
    <div className="space-y-1 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-[12px] leading-relaxed text-slate-700">
      {lines.map((l) => (
        <div key={l} className="break-words whitespace-pre-wrap">
          {l}
        </div>
      ))}
    </div>
  )
}

export function PointsSection({ points }: { points: string[] }) {
  return (
    <Section id="points" title="Points to remember">
      <ul className="space-y-1.5">
        {points.map((p) => (
          <li key={p} className="flex gap-2 text-[13px] leading-snug text-slate-600">
            <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-400" />
            <span className="min-w-0 break-words">{p}</span>
          </li>
        ))}
      </ul>
    </Section>
  )
}

export function ReferenceSection({ docs }: { docs: NodeDocs }) {
  if (!docs.cs336Ref) {
    if (!docs.cs336Note) return null
    return (
      <Section id="cs336" title="CS336 reference">
        <p className="text-xs leading-relaxed break-words text-slate-500">{docs.cs336Note}</p>
      </Section>
    )
  }
  const { file, symbol } = docs.cs336Ref
  return (
    <Section id="cs336" title="CS336 reference">
      <div className="flex items-start gap-2 font-mono text-xs leading-relaxed">
        <FileCode size={14} className="mt-0.5 shrink-0 text-slate-400" />
        <span className="min-w-0 break-words">
          <span className="text-slate-500">cs336_basics/</span>
          <span className="text-slate-700">{file}</span>
          <span className="text-slate-400"> · </span>
          <span className="text-indigo-700">{symbol}</span>
        </span>
      </div>
    </Section>
  )
}
