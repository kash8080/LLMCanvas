import { BarChart3, ChevronDown, ChevronUp } from 'lucide-react'
import type { ReactNode } from 'react'
import { formatCount } from '../engine/format'
import { useCanvasStore, type AnalysisTab, type CanvasState } from '../store/useCanvasStore'
import { ParamsTab } from './analysis/ParamsTab'

/**
 * Tabs of the analysis panel. To add one (Phase 6: Memory): extend `AnalysisTab` in the store,
 * add an entry here (label + short summary shown on the tab) and a component in `TAB_BODY`.
 */
const TABS: { id: AnalysisTab; label: string; meta: (s: CanvasState) => string }[] = [
  { id: 'params', label: 'Parameters', meta: (s) => formatCount(s.inference.params.total) },
]

const TAB_BODY: Record<AnalysisTab, () => ReactNode> = {
  params: () => <ParamsTab />,
}

/** Bottom panel (collapsible): parameter breakdown (Phase 5), memory (Phase 6). */
export function AnalysisPanel() {
  const open = useCanvasStore((s) => s.analysisOpen)
  const tab = useCanvasStore((s) => s.analysisTab)
  const setOpen = useCanvasStore((s) => s.setAnalysisOpen)
  const openAnalysis = useCanvasStore((s) => s.openAnalysis)

  return (
    <section className="shrink-0 border-t border-slate-200 bg-white">
      <div className="flex h-9 items-center gap-1 pr-1 pl-3">
        <button type="button" onClick={() => setOpen(!open)} className="flex items-center gap-2 pr-2 text-sm font-medium text-slate-600 hover:text-slate-900">
          <BarChart3 size={15} className="text-slate-400" />
          Analysis
        </button>
        <div className="flex items-center gap-1" role="tablist">
          {TABS.map((t) => (
            <TabButton key={t.id} active={open && tab === t.id} onClick={() => (open && tab === t.id ? setOpen(false) : openAnalysis(t.id))} label={t.label} meta={t.meta} />
          ))}
        </div>
        <button
          type="button"
          onClick={() => setOpen(!open)}
          title={open ? 'Collapse' : 'Expand'}
          className="ml-auto flex h-7 flex-1 items-center justify-end rounded px-2 text-slate-400 hover:bg-slate-50"
        >
          {open ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
        </button>
      </div>
      {open && <div className="h-72 overflow-y-auto border-t border-slate-100">{TAB_BODY[tab]()}</div>}
    </section>
  )
}

function TabButton({ active, onClick, label, meta }: { active: boolean; onClick: () => void; label: string; meta: (s: CanvasState) => string }) {
  const value = useCanvasStore(meta)
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs transition ${active ? 'bg-indigo-50 text-indigo-700' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-800'}`}
    >
      {label}
      <span className={`tabular-nums ${active ? 'text-indigo-500' : 'text-slate-400'}`}>{value}</span>
    </button>
  )
}
