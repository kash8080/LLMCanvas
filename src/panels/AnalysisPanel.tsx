import { BarChart3, ChevronDown, ChevronUp } from 'lucide-react'
import { useCanvasStore } from '../store/useCanvasStore'

/** Bottom panel: parameter & memory breakdown (filled in by Phases 5 and 6). */
export function AnalysisPanel() {
  const open = useCanvasStore((s) => s.analysisOpen)
  const setOpen = useCanvasStore((s) => s.setAnalysisOpen)

  return (
    <section className="shrink-0 border-t border-slate-200 bg-white">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex h-9 w-full items-center gap-2 px-3 text-sm font-medium text-slate-600 hover:bg-slate-50"
      >
        <BarChart3 size={15} className="text-slate-400" />
        Analysis
        <span className="ml-auto text-slate-400">{open ? <ChevronDown size={16} /> : <ChevronUp size={16} />}</span>
      </button>
      {open && (
        <div className="h-48 overflow-y-auto border-t border-slate-100 px-4 py-3 text-sm text-slate-400">
          Parameter and memory breakdowns will appear here (Phases 5 and 6).
        </div>
      )}
    </section>
  )
}
