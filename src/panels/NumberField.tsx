import { useState } from 'react'

/**
 * Number input that keeps the user's text while typing and commits only valid values.
 * Invalid text (empty, not a number, below min / above max, non-integer when `integer`) gets a red border.
 */
export function NumberField({
  value,
  onCommit,
  integer = false,
  min,
  max,
  className = '',
}: {
  value: number
  onCommit: (n: number) => void
  integer?: boolean
  min?: number
  max?: number
  className?: string
}) {
  const [draft, setDraft] = useState(String(value))
  const [seen, setSeen] = useState(value)
  // Value changed from outside (reset, import, relink…): show it, unless the draft already means it.
  if (value !== seen) {
    setSeen(value)
    if (Number(draft) !== value) setDraft(String(value))
  }

  const isValid = (text: string) => {
    const n = Number(text)
    return text.trim() !== '' && Number.isFinite(n) && (!integer || Number.isInteger(n)) && (min == null || n >= min) && (max == null || n <= max)
  }

  return (
    <input
      type="text"
      inputMode={integer ? 'numeric' : 'decimal'}
      value={draft}
      onChange={(e) => {
        setDraft(e.target.value)
        if (isValid(e.target.value)) onCommit(Number(e.target.value))
      }}
      onBlur={() => !isValid(draft) && setDraft(String(value))}
      className={`rounded border px-1.5 py-0.5 font-mono text-xs tabular-nums outline-none focus:ring-2 ${
        isValid(draft) ? 'border-slate-300 focus:border-indigo-400 focus:ring-indigo-100' : 'border-red-400 focus:ring-red-100'
      } ${className}`}
    />
  )
}
