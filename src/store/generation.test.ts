import { beforeEach, describe, expect, it } from 'vitest'
import { selectGenerationSettings, selectMemory, useCanvasStore } from './useCanvasStore'

const s = () => useCanvasStore.getState()
const KV = 67_108_864

beforeEach(() => {
  s().resetCanvas()
  s().clearHistory()
  s().setMemoryMode('train')
  s().setGeneration({ on: false, length: null, batch: null })
})

describe('generation (KV cache) view in the store', () => {
  it('only applies in Forward mode, and the Mem total includes the cache', () => {
    s().setGeneration({ on: true })
    expect(selectGenerationSettings(s())).toBeNull() // Train mode: ignored
    expect(selectMemory(s()).generation).toBeUndefined()
    s().setMemoryMode('forward')
    const plain = selectMemory({ ...s(), generation: { ...s().generation, on: false } }).total
    const r = selectMemory(s())
    expect(r.byComponent.kv_cache).toBe(KV)
    expect(r.total).toBe(r.byComponent.weights + r.byComponent.buffers + KV + r.activations.total)
    expect(r.total).not.toBe(plain)
  })

  it('length / batch settings follow the hyperparams until set; UI state only (not undoable)', () => {
    s().setMemoryMode('forward')
    s().setGeneration({ on: true })
    s().setHyperparam('context_length', 512)
    expect(selectMemory(s()).generation!.kv.total).toBe(2 * KV)
    s().setGeneration({ length: 128, batch: 8 })
    expect(selectMemory(s()).generation!.kv.total).toBe(KV / 8)
    expect(s().history.past).toHaveLength(1) // only the context_length edit
    s().undo()
    expect(s().generation).toEqual({ on: true, length: 128, batch: 8 })
  })
})
