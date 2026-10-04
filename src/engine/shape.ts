// Shape helpers (pure TS).
import { bytesPerElement } from './hyperparams'
import type { Dim, Hyperparams, Shape } from './types'

export function floatShape(dims: Dim[]): Shape {
  return { dims, dtype: 'float' }
}

export function numel(shape: Shape): number {
  return shape.dims.reduce((acc, d) => acc * d.size, 1)
}

export function shapeBytes(shape: Shape, hp: Pick<Hyperparams, 'dtype'>): number {
  return numel(shape) * bytesPerElement(shape.dtype, hp)
}

export function sameSizes(a: Shape, b: Shape): boolean {
  return a.dims.length === b.dims.length && a.dims.every((d, i) => d.size === b.dims[i].size)
}

/** `32×256×512`; scalar → `scalar`. */
export function formatConcrete(shape: Shape): string {
  return shape.dims.length === 0 ? 'scalar' : shape.dims.map((d) => d.size).join('×')
}

/** `B × T × d_model` (unlabelled dims show their size); scalar → `scalar`. */
export function formatSymbolic(shape: Shape): string {
  return shape.dims.length === 0 ? 'scalar' : shape.dims.map((d) => d.label ?? String(d.size)).join(' × ')
}

/** `512 (d_model)` or `768`. */
export function formatDim(d: Dim): string {
  return d.label ? `${d.size} (${d.label})` : String(d.size)
}
