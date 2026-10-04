// Resolve a node's params: global binding (🔗) vs local override.
import { bindSymbol, bindValue } from './hyperparams'
import type { BindKey, Hyperparams, NodeDef, ParamValue, ResolvedParams } from './types'

export function isBound(v: ParamValue): v is { bind: BindKey } {
  return 'bind' in v
}

/** Fresh params for a new node: each schema default. */
export function defaultParams(def: NodeDef): Record<string, ParamValue> {
  return Object.fromEntries(def.params.map((s) => [s.key, structuredClone(s.default)]))
}

/** Resolve every param in the schema (missing values fall back to the schema default). */
export function resolveParams(def: NodeDef, values: Record<string, ParamValue>, hp: Hyperparams): ResolvedParams {
  const out: ResolvedParams = {}
  for (const schema of def.params) {
    const v = values[schema.key] ?? schema.default
    out[schema.key] = isBound(v)
      ? { value: bindValue(v.bind, hp), label: bindSymbol(v.bind), bound: true }
      : { value: v.value, bound: false }
  }
  return out
}

/** Validation of local overrides (bound values are validated in the hyperparams panel). */
export function validateParams(def: NodeDef, values: Record<string, ParamValue>): string[] {
  const errors: string[] = []
  for (const schema of def.params) {
    const v = values[schema.key] ?? schema.default
    if (isBound(v)) continue
    const x = v.value
    if (schema.kind === 'int' || schema.kind === 'float') {
      if (typeof x !== 'number' || !Number.isFinite(x)) errors.push(`${schema.key} must be a number`)
      else if (schema.kind === 'int' && !Number.isInteger(x)) errors.push(`${schema.key} must be an integer (got ${x})`)
      else if (schema.min != null && x < schema.min) errors.push(`${schema.key} must be ≥ ${schema.min} (got ${x})`)
    } else if (schema.kind === 'bool' && typeof x !== 'boolean') errors.push(`${schema.key} must be true/false`)
    else if (schema.kind === 'enum' && !schema.options?.includes(String(x))) errors.push(`${schema.key} has an invalid option`)
  }
  return errors
}

/** Helpers for node defs. */
export function num(p: ResolvedParams, key: string): number {
  return Number(p[key]?.value)
}

/** A param as a Dim (size + symbol label when bound). */
export function paramDim(p: ResolvedParams, key: string): { size: number; label?: string } {
  const r = p[key]
  return r?.label ? { size: Number(r.value), label: r.label } : { size: Number(r?.value) }
}
