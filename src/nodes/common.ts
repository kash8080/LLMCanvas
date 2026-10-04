// Small helpers shared by node definitions.
import { formatConcrete, formatDim } from '../engine/shape'
import type { Dim, ParamCountResult, Shape } from '../engine/types'

export const NO_PARAMS = (): ParamCountResult => ({ total: 0, tensors: [] })
export const NOTHING_SAVED = () => []

/** Param count for a list of weight tensors. */
export function weights(...tensors: { name: string; dims: Dim[] }[]): ParamCountResult {
  return { total: tensors.reduce((acc, t) => acc + t.dims.reduce((a, d) => a * d.size, 1), 0), tensors }
}

export function last(shape: Shape): Dim | undefined {
  return shape.dims[shape.dims.length - 1]
}

/** Error unless `shape` is a float tensor with at least `rank` dims. */
export function expectFloat(shape: Shape, what: string, rank = 1): string | null {
  if (shape.dtype !== 'float') return `${what} must be a float tensor, got int64 token ids`
  if (shape.dims.length < rank) return `${what} needs at least ${rank} dims, got ${formatConcrete(shape)}`
  return null
}

/** `Linear.in_features = 768 but input last dim is 512 (d_model)` */
export function mismatch(param: string, expected: number, actual: Dim, where = 'input last dim'): string {
  return `${param} = ${expected} but ${where} is ${formatDim(actual)}`
}
