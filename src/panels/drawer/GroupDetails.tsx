import { LOD_ZOOM } from '../../canvas/lod'
import { ModeToggle } from '../../canvas/nodes/GroupNode'
import type { GroupNode } from '../../canvas/types'
import { GROUP_DEFS } from '../../nodes/groups'
import { useCanvasStore } from '../../store/useCanvasStore'
import { FormulaSection, OverviewSection, PointsSection, ReferenceSection } from './DocsSections'
import { ShapesSection } from './ShapesSection'
import { GroupSizeSection } from './SizeSection'
import { DrawerBody, DrawerHeader, ErrorBox, Section, StatusBadge } from './ui'

/** Drawer for a group frame (Transformer Block / MHA / SwiGLU). Groups have no params: "Display" takes that slot. */
export function GroupDetails({ node }: { node: GroupNode }) {
  const def = GROUP_DEFS[node.data.groupType]
  const summary = useCanvasStore((s) => s.inference.groups[node.id])
  const setTitle = useCanvasStore((s) => s.setTitle)
  const autoAt = Math.round((def.expandAtLod === 1 ? LOD_ZOOM.blocks : LOD_ZOOM.internals) * 100)
  const errors = summary?.errors ?? []

  return (
    <>
      <DrawerHeader
        color={def.color}
        title={node.data.title ?? ''}
        placeholder={def.label}
        onRename={(t) => setTitle(node.id, t)}
        subtitle={`${def.label} · group`}
        badge={<StatusBadge status={summary?.status ?? 'unknown'} errorCount={errors.length} />}
      />
      <DrawerBody>
        {errors.length > 0 && <ErrorBox errors={errors} />}

        <OverviewSection docs={def.docs} />
        <FormulaSection lines={def.docs.formula} />

        <Section id="display" title="Display">
          <ModeToggle id={node.id} mode={node.data.mode} className="w-fit text-xs" />
          <p className="mt-1.5 text-[11px] leading-snug text-slate-400">
            Auto opens this group when you zoom in to {autoAt}% or more. Open / Closed ignore the zoom.
          </p>
        </Section>

        <ShapesSection
          inputs={[{ port: 'x', shape: summary?.inShape ?? null }]}
          outputs={[{ port: 'out', shape: summary?.outShape ?? null }]}
          unknownOut={errors.length > 0 ? 'not computed — fix the problems above' : undefined}
        />
        <GroupSizeSection nodeId={node.id} summary={summary} paramFormula={def.docs.paramFormula} color={def.color} />
        <PointsSection points={def.docs.pointsToRemember} />
        <ReferenceSection docs={def.docs} />
      </DrawerBody>
    </>
  )
}
