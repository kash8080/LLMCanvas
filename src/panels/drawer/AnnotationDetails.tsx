import type { AnnotationData, StickyNode, TextBoxNode } from '../../canvas/types'
import { useCanvasStore } from '../../store/useCanvasStore'
import { DrawerBody, DrawerHeader, Section, SubHeading } from './ui'

const BG_SWATCHES = ['transparent', '#ffffff', '#fef08a', '#fed7aa', '#fecdd3', '#e9d5ff', '#bfdbfe', '#bbf7d0', '#e2e8f0']
const TEXT_SWATCHES = ['#111827', '#1f2937', '#64748b', '#dc2626', '#2563eb', '#16a34a', '#ffffff']

/** Drawer for a sticky note / text box: background, text colour, font size. */
export function AnnotationDetails({ node }: { node: StickyNode | TextBoxNode }) {
  const updateAnnotation = useCanvasStore((s) => s.updateAnnotation)
  const update = (patch: Partial<AnnotationData>) => updateAnnotation(node.id, patch)
  const { bgColor, textColor, fontSize, text } = node.data
  const label = node.type === 'sticky' ? 'Sticky note' : 'Text box'
  const firstLine = text.split('\n').find((l) => l.trim() !== '')?.trim()

  return (
    <>
      <DrawerHeader
        color={bgColor === 'transparent' ? '#ffffff' : bgColor}
        title={firstLine && firstLine.length <= 60 ? firstLine : label}
        subtitle={`${label} · annotation`}
      />
      <DrawerBody>
        <Section id="appearance" title="Appearance">
          <SubHeading>Background</SubHeading>
          <Swatches colors={BG_SWATCHES} value={bgColor} onChange={(c) => update({ bgColor: c })} />
          <SubHeading>Text color</SubHeading>
          <Swatches colors={TEXT_SWATCHES} value={textColor} onChange={(c) => update({ textColor: c })} />
          <SubHeading>Font size</SubHeading>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={10}
              max={48}
              value={fontSize}
              onChange={(e) => update({ fontSize: Number(e.target.value) })}
              className="flex-1 accent-indigo-600"
            />
            <span className="w-10 text-right text-xs text-slate-500 tabular-nums">{fontSize}px</span>
          </div>
        </Section>
        <p className="px-4 pt-3 text-xs text-slate-400">Double-click the element on the canvas to edit its text; drag its corners (when selected) to resize.</p>
      </DrawerBody>
    </>
  )
}

function Swatches({ colors, value, onChange }: { colors: string[]; value: string; onChange: (c: string) => void }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {colors.map((c) => (
        <button
          key={c}
          type="button"
          title={c}
          onClick={() => onChange(c)}
          className={`h-6 w-6 rounded-full border transition ${
            value === c ? 'border-indigo-500 ring-2 ring-indigo-200' : 'border-slate-300 hover:scale-110'
          } ${c === 'transparent' ? 'swatch-transparent' : ''}`}
          style={c === 'transparent' ? undefined : { backgroundColor: c }}
        />
      ))}
    </div>
  )
}
