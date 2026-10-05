// Weight tying (Phase 7c, hyperparam `tie_embeddings`). Pure TS, no React.
//
// When on, the LM head — a Linear whose output goes straight into a Logits part (same rule as the
// "LM head" category in params.ts) — reuses the token Embedding's matrix instead of owning one:
//   embedding.weight: vocab_size × d_model      lm_head.weight: out_features × in_features
// so they can only be shared when out_features = vocab_size and in_features = d_model.
//   valid    → the LM head's param count becomes 0 (`ParamCountResult.tied`), so totals, categories,
//              group sums and memory (P) all count the matrix once;
//   mismatch → an error on the LM head part (like any other shape error) and nothing is tied.
// The Embedding used is one that feeds the LM head (walking edges backwards). CS336 does not tie
// (`tie_embeddings` defaults to false).
import { resolveParams, num } from './resolve'
import type { GraphModel, Hyperparams, NodeDef } from './types'

export interface TyingCheck {
  /** LM head part id → id of the Embedding part whose weight it shares (valid ties only). */
  tied: Record<string, string>
  /** Problems per LM head part id (shown as part errors). */
  errors: Record<string, string[]>
}

export const NO_TYING: TyingCheck = { tied: {}, errors: {} }

/** Which LM heads share the embedding matrix, or why they can't. `graph` is the flattened graph. */
export function checkTying(graph: GraphModel, hp: Hyperparams, defs: Record<string, NodeDef>): TyingCheck {
  if (!hp.tie_embeddings) return NO_TYING
  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]))
  const edges = graph.edges.filter((e) => nodeById.has(e.source) && nodeById.has(e.target))
  const lmHeads = [...new Set(edges.filter((e) => nodeById.get(e.target)?.type === 'logits' && nodeById.get(e.source)?.type === 'linear').map((e) => e.source))]
  if (lmHeads.length === 0) return NO_TYING

  const predecessors = new Map<string, string[]>()
  for (const e of edges) {
    if (!predecessors.has(e.target)) predecessors.set(e.target, [])
    predecessors.get(e.target)!.push(e.source)
  }
  /** Embedding parts upstream of `id`, nearest first. */
  const upstreamEmbeddings = (id: string): string[] => {
    const seen = new Set<string>([id])
    const queue = [id]
    const found: string[] = []
    while (queue.length > 0) {
      const cur = queue.shift()!
      for (const p of predecessors.get(cur) ?? []) {
        if (seen.has(p)) continue
        seen.add(p)
        if (nodeById.get(p)?.type === 'embedding') found.push(p)
        queue.push(p)
      }
    }
    return found
  }

  const out: TyingCheck = { tied: {}, errors: {} }
  for (const id of lmHeads) {
    const embId = upstreamEmbeddings(id)[0]
    if (!embId) {
      out.errors[id] = ['Weight tying is on, but no Embedding feeds this LM head, so there is no matrix to share.']
      continue
    }
    const head = resolveParams(defs.linear, nodeById.get(id)!.params, hp)
    const emb = resolveParams(defs.embedding, nodeById.get(embId)!.params, hp)
    const [outF, inF] = [num(head, 'out_features'), num(head, 'in_features')]
    const [V, d] = [num(emb, 'vocab_size'), num(emb, 'd_model')]
    if (outF !== V || inF !== d) {
      out.errors[id] = [
        `Weight tying: this LM head's weight is ${outF} × ${inF} (out × in) but the Embedding is ${V} × ${d} (vocab_size × d_model); they must match to share one matrix.`,
      ]
      continue
    }
    out.tied[id] = embId
  }
  return out
}
