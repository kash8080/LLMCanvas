import { describe, expect, it } from 'vitest'
import { formatBytes, formatCount } from './format'

describe('formatCount', () => {
  it('formats with K/M/B suffixes', () => {
    expect(formatCount(42)).toBe('42')
    expect(formatCount(5120)).toBe('5.12K')
    expect(formatCount(16_468_480)).toBe('16.47M')
    expect(formatCount(1_500_000_000)).toBe('1.50B')
  })
})

describe('formatBytes', () => {
  it('uses decimal units', () => {
    expect(formatBytes(512)).toBe('512 B')
    // attention probs for one layer with CS336 defaults: 32·16·256·256·4 bytes
    expect(formatBytes(32 * 16 * 256 * 256 * 4)).toBe('134.2 MB')
  })
})
