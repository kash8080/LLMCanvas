import { ReactFlowProvider } from '@xyflow/react'
import { Canvas } from './canvas/Canvas'
import { AnalysisPanel } from './panels/AnalysisPanel'
import { DetailDrawer } from './panels/DetailDrawer'
import { Palette } from './panels/Palette'
import { Toolbar } from './panels/Toolbar'

export default function App() {
  return (
    <ReactFlowProvider>
      <div className="flex h-screen w-screen flex-col bg-slate-50 text-slate-800">
        <Toolbar />
        <div className="flex min-h-0 flex-1">
          <Palette />
          <main className="flex min-w-0 flex-1 flex-col">
            <div className="relative min-h-0 flex-1">
              <Canvas />
            </div>
            <AnalysisPanel />
          </main>
          <DetailDrawer />
        </div>
      </div>
    </ReactFlowProvider>
  )
}
