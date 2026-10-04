// Proxy parts that live inside every group and bridge its outer ports (see src/engine/groups.ts).
// Engine-side both are identity pass-throughs with one input and one output; on the canvas the
// group_input proxy only shows its output and the group_output proxy only shows its input.
import { GROUP_INPUT, GROUP_OUTPUT } from '../engine/groups'
import type { NodeDef } from '../engine/types'
import { NO_PARAMS, NOTHING_SAVED } from './common'

const passThrough: NodeDef['infer'] = ({ inputs: [x] }) => ({ outputs: [structuredClone(x)], errors: [] })

export const groupInput: NodeDef = {
  type: GROUP_INPUT,
  label: 'Group input',
  category: 'io',
  inputs: [{ id: 'in', label: 'group input' }],
  outputs: [{ id: 'out', label: 'in' }],
  params: [],
  infer: passThrough,
  paramCount: NO_PARAMS,
  savedForBackward: NOTHING_SAVED,
  docs: {
    overview: 'Where the tensor arriving at the group’s outer input port enters the group. A pass-through: its shape is what the group shows as “in”.',
    pointsToRemember: [
      'Connect things from outside to the group’s top port, not to the parts inside.',
      'No parameters; it only forwards the tensor.',
      'It is removed only together with its group.',
    ],
  },
}

export const groupOutput: NodeDef = {
  type: GROUP_OUTPUT,
  label: 'Group output',
  category: 'io',
  inputs: [{ id: 'in', label: 'out' }],
  outputs: [{ id: 'out', label: 'group output' }],
  params: [],
  infer: passThrough,
  paramCount: NO_PARAMS,
  savedForBackward: NOTHING_SAVED,
  docs: {
    overview: 'Whatever reaches this proxy leaves the group through its outer output port. A pass-through: its shape is what the group shows as “out”.',
    pointsToRemember: [
      'Connect things outside to the group’s bottom port, not to the parts inside.',
      'No parameters; it only forwards the tensor.',
      'It is removed only together with its group.',
    ],
  },
}

export const PROXY_DEFS: NodeDef[] = [groupInput, groupOutput]
