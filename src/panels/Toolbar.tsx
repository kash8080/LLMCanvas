import { useReactFlow } from '@xyflow/react'
import { Download, PanelBottom, PanelRight, RotateCcw, Upload, Workflow } from 'lucide-react'
import { useRef } from 'react'
import { parseDocument, toDocument } from '../store/persistence'
import { useCanvasStore } from '../store/useCanvasStore'

function ToolbarButton({ onClick, icon: Icon, label, active }: { onClick: () => void; icon: typeof Download; label: string; active?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm transition ${
        active ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
      }`}
    >
      <Icon size={16} />
      <span className="hidden lg:inline">{label}</span>
    </button>
  )
}

export function Toolbar() {
  const fileInput = useRef<HTMLInputElement>(null)
  const { fitView } = useReactFlow()
  const drawerOpen = useCanvasStore((s) => s.drawerOpen)
  const analysisOpen = useCanvasStore((s) => s.analysisOpen)
  const { loadDocument, resetCanvas, setDrawerOpen, setAnalysisOpen } = useCanvasStore.getState()

  const fitSoon = () => setTimeout(() => fitView({ padding: 0.3, maxZoom: 1, duration: 300 }), 50)

  const exportJson = () => {
    const { nodes, edges } = useCanvasStore.getState()
    const blob = new Blob([JSON.stringify(toDocument(nodes, edges), null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'llm-canvas.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const importJson = async (file: File) => {
    try {
      loadDocument(parseDocument(JSON.parse(await file.text())))
      fitSoon()
    } catch (err) {
      alert(`Could not import "${file.name}": ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  const reset = () => {
    if (confirm('Reset the canvas? Your current work will be replaced.')) {
      resetCanvas()
      fitSoon()
    }
  }

  return (
    <header className="flex h-12 shrink-0 items-center gap-1 border-b border-slate-200 bg-white px-3">
      <div className="mr-4 flex items-center gap-2 font-semibold text-slate-800">
        <Workflow size={18} className="text-indigo-600" />
        LLM Canvas
      </div>

      {/* Hyperparams / mode / dtype / totals arrive in Phases 2, 5 and 6 (PLAN.md §2.5). */}
      <div className="hidden items-center gap-3 text-xs text-slate-400 md:flex">
        <span>Params —</span>
        <span>Mem —</span>
      </div>

      <div className="ml-auto flex items-center gap-1">
        <ToolbarButton icon={Download} label="Export JSON" onClick={exportJson} />
        <ToolbarButton icon={Upload} label="Import JSON" onClick={() => fileInput.current?.click()} />
        <ToolbarButton icon={RotateCcw} label="Reset" onClick={reset} />
        <div className="mx-1 h-5 w-px bg-slate-200" />
        <ToolbarButton icon={PanelBottom} label="Analysis" active={analysisOpen} onClick={() => setAnalysisOpen(!analysisOpen)} />
        <ToolbarButton icon={PanelRight} label="Details" active={drawerOpen} onClick={() => setDrawerOpen(!drawerOpen)} />
      </div>

      <input
        ref={fileInput}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) void importJson(file)
          e.target.value = '' // allow re-importing the same file
        }}
      />
    </header>
  )
}
