import { getViewportForBounds, useReactFlow, useStoreApi } from '@xyflow/react'
import { Download, PanelBottom, PanelRight, RotateCcw, Ruler, Upload, Workflow } from 'lucide-react'
import { useRef } from 'react'
import { formatBytes, formatCount } from '../engine/format'
import { FLOAT_DTYPES } from '../engine/hyperparams'
import { MEMORY_MODES } from '../engine/memory'
import type { FloatDType } from '../engine/types'
import { numLayers } from '../canvas/groupTemplates'
import { startBounds } from '../canvas/nodeFactory'
import { parseDocument, toDocument } from '../store/persistence'
import { selectMemory, useCanvasStore } from '../store/useCanvasStore'
import { MODE_INFO } from './analysis/memoryText'
import { CONNECTED_RULE } from './analysis/ParamsTab'
import { HyperparamsMenu } from './HyperparamsMenu'

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
      <span className="hidden whitespace-nowrap 2xl:inline">{label}</span>
    </button>
  )
}

export function Toolbar() {
  const fileInput = useRef<HTMLInputElement>(null)
  const { setViewport } = useReactFlow()
  const flowStore = useStoreApi()
  const drawerOpen = useCanvasStore((s) => s.drawerOpen)
  const analysisOpen = useCanvasStore((s) => s.analysisOpen)
  const showEdgeShapes = useCanvasStore((s) => s.showEdgeShapes)
  const layers = useCanvasStore((s) => numLayers(s.nodes))
  const totalParams = useCanvasStore((s) => s.inference.params.total)
  const unconnected = useCanvasStore((s) => s.inference.params.unconnected)
  const { loadDocument, resetCanvas, setDrawerOpen, setAnalysisOpen, openAnalysis, setShowEdgeShapes } = useCanvasStore.getState()

  // Show the start of the model. Uses the nodes' declared positions/sizes rather than fitView, which
  // waits for freshly loaded nodes to be measured (unreliable while groups hide their children).
  const fitSoon = () => {
    const { width, height } = flowStore.getState()
    const bounds = startBounds(useCanvasStore.getState().nodes)
    if (bounds) void setViewport(getViewportForBounds(bounds, width, height, 0.05, 1, 0.1), { duration: 300 })
  }

  const exportJson = () => {
    const { nodes, edges, hyperparams } = useCanvasStore.getState()
    const blob = new Blob([JSON.stringify(toDocument(nodes, edges, hyperparams), null, 2)], { type: 'application/json' })
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
    if (confirm('Reset to the CS336 default model? Your current work will be replaced.')) {
      resetCanvas()
      fitSoon()
    }
  }

  return (
    <header className="flex h-12 shrink-0 items-center gap-1 border-b border-slate-200 bg-white px-3">
      <div className="mr-4 flex shrink-0 items-center gap-2 font-semibold whitespace-nowrap text-slate-800">
        <Workflow size={18} className="text-indigo-600" />
        LLM Canvas
      </div>

      <HyperparamsMenu />
      <ToolbarButton icon={Ruler} label="Shapes on edges" active={showEdgeShapes} onClick={() => setShowEdgeShapes(!showEdgeShapes)} />

      <MemoryControls />

      <div className="ml-3 hidden items-center gap-3 text-xs whitespace-nowrap text-slate-500 md:flex">
        <button
          type="button"
          onClick={() => openAnalysis('params')}
          className="-mx-1.5 rounded px-1.5 py-1 hover:bg-slate-100"
          title={`${totalParams.toLocaleString()} parameters in the model (${CONNECTED_RULE})${
            unconnected > 0 ? `\n+${unconnected.toLocaleString()} in unconnected parts, not counted.` : ''
          }\nClick for the breakdown.`}
        >
          Params <span className="font-semibold text-slate-700 tabular-nums">{formatCount(totalParams)}</span>
          {unconnected > 0 && <span className="ml-1 text-amber-600 tabular-nums">+{formatCount(unconnected)}</span>}
        </button>
        <span title="num_layers = Transformer Blocks on the canvas">
          Layers <span className="font-semibold text-slate-700 tabular-nums">{layers}</span>
        </span>
        <MemoryChip />
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

/** Mode (Forward / Fwd+Bwd / Train), dtype and activation checkpointing: the memory knobs (R8.3). */
function MemoryControls() {
  const mode = useCanvasStore((s) => s.memoryMode)
  const dtype = useCanvasStore((s) => s.hyperparams.dtype)
  const checkpointing = useCanvasStore((s) => s.checkpointing)
  const { setMemoryMode, setHyperparam, setCheckpointing } = useCanvasStore.getState()
  return (
    <div className="ml-2 flex items-center gap-2 border-l border-slate-200 pl-3 text-xs">
      <div className="flex overflow-hidden rounded-md border border-slate-300" role="radiogroup" aria-label="Memory mode">
        {MEMORY_MODES.map((m) => (
          <button
            key={m}
            type="button"
            role="radio"
            aria-checked={mode === m}
            onClick={() => setMemoryMode(m)}
            title={MODE_INFO[m].explain}
            className={`px-2 py-1 font-medium transition ${mode === m ? 'bg-indigo-600 text-white' : 'text-slate-600 hover:bg-slate-100'}`}
          >
            {MODE_INFO[m].label}
          </button>
        ))}
      </div>
      <select
        value={dtype}
        onChange={(e) => setHyperparam('dtype', e.target.value as FloatDType)}
        title="dtype of weights, gradients, optimizer state and activations (int64 token ids are always 8 bytes)"
        className="rounded-md border border-slate-300 bg-white px-1 py-1 text-xs text-slate-700 outline-none focus:border-indigo-400"
      >
        {FLOAT_DTYPES.map((d) => (
          <option key={d} value={d}>
            {d}
          </option>
        ))}
      </select>
      <label
        className={`flex items-center gap-1.5 ${mode === 'forward' ? 'text-slate-300' : 'text-slate-600'}`}
        title={
          mode === 'forward'
            ? 'Only matters with a backward pass (Fwd+Bwd / Train)'
            : 'Activation checkpointing (CS336 checkpoint_blocks): keep only each Transformer Block’s input and recompute the block during backward'
        }
      >
        <input type="checkbox" checked={checkpointing} disabled={mode === 'forward'} onChange={(e) => setCheckpointing(e.target.checked)} className="accent-indigo-600" />
        <span className="whitespace-nowrap">Act. checkpointing</span>
      </label>
    </div>
  )
}

/** "Mem 1.51 GB" — click opens the Memory tab. */
function MemoryChip() {
  const total = useCanvasStore((s) => selectMemory(s).total)
  const mode = useCanvasStore((s) => s.memoryMode)
  const openAnalysis = useCanvasStore((s) => s.openAnalysis)
  return (
    <button
      type="button"
      onClick={() => openAnalysis('memory')}
      className="-mx-1.5 rounded px-1.5 py-1 hover:bg-slate-100"
      title={`Estimated memory for ${MODE_INFO[mode].label}: ${total.toLocaleString()} bytes.\nClick for the breakdown.`}
    >
      Mem <span className="font-semibold text-slate-700 tabular-nums">{formatBytes(total)}</span>
    </button>
  )
}
