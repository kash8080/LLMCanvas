// Shared building blocks for the detail drawer: header, collapsible sections, badges.
import { AlertTriangle, Check, ChevronRight, HelpCircle, X } from 'lucide-react'
import type { ReactNode } from 'react'
import type { NodeStatus } from '../../engine/types'
import { useCanvasStore } from '../../store/useCanvasStore'

/** Top of the drawer: colour chip, (editable) title, type name, status badge, close button. */
export function DrawerHeader({
  color,
  title,
  placeholder,
  onRename,
  subtitle,
  badge,
}: {
  color: string
  title: string
  /** Shown when the title is empty (the default label). */
  placeholder?: string
  /** When given, the title is an input. */
  onRename?: (title: string) => void
  subtitle: ReactNode
  badge?: ReactNode
}) {
  const setOpen = useCanvasStore((s) => s.setDrawerOpen)
  return (
    <header className="flex shrink-0 items-start gap-2.5 border-b border-slate-200 px-4 pt-3 pb-2.5">
      <span className="mt-1.5 h-3.5 w-3.5 shrink-0 rounded border border-black/10" style={{ background: color }} />
      <div className="min-w-0 flex-1">
        {onRename ? (
          <input
            value={title}
            placeholder={placeholder}
            onChange={(e) => onRename(e.target.value)}
            className="-mx-1 w-full rounded border border-transparent px-1 py-0.5 text-base font-semibold text-slate-800 outline-none placeholder:text-slate-800 hover:border-slate-200 focus:border-indigo-400 focus:placeholder:text-slate-300"
            title="Rename"
          />
        ) : (
          <div className="py-0.5 text-base font-semibold break-words text-slate-800">{title}</div>
        )}
        <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500">
          <span className="min-w-0 break-words">{subtitle}</span>
          {badge}
        </div>
      </div>
      <button
        type="button"
        onClick={() => setOpen(false)}
        title="Close"
        className="-mr-1 shrink-0 rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
      >
        <X size={16} />
      </button>
    </header>
  )
}

/** Scrolling area under the header. */
export function DrawerBody({ children }: { children: ReactNode }) {
  return <div className="min-h-0 flex-1 overflow-y-auto pb-6 text-sm">{children}</div>
}

/** ok / error / unknown pill. */
export function StatusBadge({ status, errorCount = 0 }: { status: NodeStatus; errorCount?: number }) {
  if (status === 'error')
    return (
      <Pill className="bg-red-50 text-red-700 ring-red-200">
        <AlertTriangle size={11} /> {errorCount === 1 ? '1 error' : `${errorCount} errors`}
      </Pill>
    )
  if (status === 'unknown')
    return (
      <Pill className="bg-slate-100 text-slate-500 ring-slate-200" title="Shape unknown: an input is missing or something upstream has an error">
        <HelpCircle size={11} /> unknown shape
      </Pill>
    )
  return (
    <Pill className="bg-emerald-50 text-emerald-700 ring-emerald-200">
      <Check size={11} /> ok
    </Pill>
  )
}

export function Pill({ className, title, children }: { className: string; title?: string; children: ReactNode }) {
  return (
    <span title={title} className={`inline-flex items-center gap-1 rounded-full px-1.5 py-px text-[11px] font-medium ring-1 ring-inset ${className}`}>
      {children}
    </span>
  )
}

/**
 * A collapsible drawer section. Collapsed state is kept per `id` in the store, so collapsing
 * "Points to remember" once keeps it collapsed for every part.
 */
export function Section({ id, title, meta, children }: { id: string; title: string; meta?: ReactNode; children: ReactNode }) {
  const collapsed = useCanvasStore((s) => !!s.collapsedSections[id])
  const toggle = useCanvasStore((s) => s.toggleSection)
  return (
    <section className="border-b border-slate-100 px-4 py-3">
      <button type="button" onClick={() => toggle(id)} className="flex w-full items-center gap-1 text-left" aria-expanded={!collapsed}>
        <ChevronRight size={14} className={`-ml-1 shrink-0 text-slate-400 transition-transform ${collapsed ? '' : 'rotate-90'}`} />
        <h3 className="text-[11px] font-semibold tracking-wide text-slate-500 uppercase">{title}</h3>
        {meta != null && <span className="ml-auto pl-2 text-xs text-slate-500 tabular-nums">{meta}</span>}
      </button>
      {!collapsed && <div className="mt-2.5">{children}</div>}
    </section>
  )
}

/** Small label inside a section (e.g. "In" / "Out", "Weights"). */
export function SubHeading({ children }: { children: ReactNode }) {
  return <div className="mt-3 mb-1.5 text-[11px] font-medium text-slate-400 first:mt-0">{children}</div>
}

/** Prominent error list shown right under the header. */
export function ErrorBox({ errors }: { errors: string[] }) {
  return (
    <div className="mx-4 mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
      <div className="mb-1 flex items-center gap-1.5 font-semibold">
        <AlertTriangle size={13} /> {errors.length === 1 ? 'Problem' : `${errors.length} problems`}
      </div>
      <ul className="space-y-1">
        {errors.map((e) => (
          <li key={e} className="break-words">
            {e}
          </li>
        ))}
      </ul>
    </div>
  )
}

/** Muted notice under the header (e.g. unknown shape). */
export function Notice({ children }: { children: ReactNode }) {
  return <div className="mx-4 mt-3 rounded-md border border-dashed border-slate-300 bg-slate-50 px-3 py-2 text-xs text-slate-500">{children}</div>
}
