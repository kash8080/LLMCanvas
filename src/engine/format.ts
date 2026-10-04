// Pure formatting helpers (no React). Used by param / memory displays in later phases.

/** 16468480 -> "16.47M", 5120 -> "5.12K", 42 -> "42". */
export function formatCount(n: number): string {
  const abs = Math.abs(n)
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${(n / 1e3).toFixed(2)}K`
  return String(n)
}

/** Bytes in decimal units (as in PLAN.md §5 examples): 134217728 -> "134.2 MB", 1250721792 -> "1.25 GB". */
export function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let i = 0
  while (Math.abs(value) >= 1000 && i < units.length - 1) {
    value /= 1000
    i += 1
  }
  return i === 0 ? `${value} B` : `${value.toFixed(i >= 3 ? 2 : 1)} ${units[i]}`
}
