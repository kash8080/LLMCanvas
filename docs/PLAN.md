# LLM Canvas — Implementation Plan

> **Status: APPROVED (2026-10-04)** with the decisions in §8.
> Requirements: [REQUIREMENTS.md](REQUIREMENTS.md). Session log: [PROGRESS.md](PROGRESS.md).

---

## 1. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Build | **Vite + React + TypeScript** | Fast, zero-config, static output (browser-only). |
| Canvas | **React Flow (`@xyflow/react` v12)** | Gives pan/zoom, minimap, ports ("handles"), edges, drag-to-connect, parent/child nodes, node resizer out of the box. Writing our own canvas is the #1 way to sink this project. |
| State | **Zustand** | One small store, easy to read from anywhere. |
| Styling | **Tailwind CSS** | Quick, consistent UI. |
| Icons | `lucide-react` | |
| Charts | Plain divs (stacked bars) | Breakdown bars don't need a chart library. |
| Tests | **Vitest** on the pure engine only | Shape inference, param counts and memory math are the parts worth testing. |
| Package manager | pnpm | |

No backend. Persistence = `localStorage` + JSON export/import.

---

## 2. Architecture

The key rule: **the engine is pure TypeScript with no React**, and **each node type is one
self-contained definition file** (ports, params, shape logic, param count, saved activations,
docs content). Adding a new part type = adding one file + registering it.

```
src/
  engine/                  # pure TS, unit-tested, no React
    types.ts               # Shape, Dim, ParamValue, NodeDef, GraphModel, Diagnostics
    hyperparams.ts         # global hyperparams + defaults (CS336 train.py)
    resolve.ts             # resolve param values (global binding vs override)
    infer.ts               # topological shape propagation + error collection
    params.ts              # parameter counting + breakdown
    memory.ts              # memory estimation per mode + breakdown
  nodes/                   # one file per node type (NodeDef objects)
    embedding.ts, rmsnorm.ts, linear.ts, rope.ts, sdpa.ts, splitHeads.ts, mergeHeads.ts,
    silu.ts, multiply.ts, add.ts, softmax.ts, crossEntropy.ts, input.ts, output.ts,
    groups.ts              # Transformer Block / MHA / SwiGLU composite templates
    registry.ts
  defaults/cs336Graph.ts   # the default CS336 TransformerLM graph + layout
  store/                   # zustand store (graph, hyperparams, selection, UI state, persistence)
  canvas/                  # React Flow wiring: node components, ports, edges, LOD logic
  panels/                  # Palette (left), Toolbar (top), DetailDrawer (right), AnalysisPanel (bottom)
```

### 2.1 Data model (engine)

- **Dim** = `{ size: number, label?: string }` — e.g. `{size: 512, label: "d_model"}` so we can show
  `B × T × d_model` *and* `32 × 256 × 512`.
- **Shape** = `Dim[]`, plus dtype.
- **ParamValue** = `{ bind: "d_model" }` (follows a global hyperparam — shown with a 🔗 link icon)
  or `{ value: 768 }` (local override). Unlinking/relinking is one click in the drawer.
- **NodeDef** (per type):
  ```ts
  {
    type, label, category, color,
    inputs:  PortDef[], outputs: PortDef[],
    params:  ParamSchema[],          // fixed list; type: int | float | bool | enum; default binding
    infer(inputs: Shape[], p, hp) => { outputs: Shape[], errors: string[] },
    paramCount(p, hp) => { total, tensors: [{name, shape}] },   // e.g. W_q: d_model × d_model
    savedForBackward(inputs, outputs, p, hp) => [{name, elements, which: "input0"|"output0"|"internal"}],
    docs: { overview, roles?, formula?: string[], paramFormula?, pointsToRemember[], paramHelp?, cs336Ref?: {file, symbol} }
  }
  ```

### 2.2 Shape inference (R2.4, R2.6)
- Graph evaluated in topological order on every change (graph is tiny — no memoization tricks needed).
- Each node gets `inputShapes`, `outputShapes`, `errors[]`. Error kinds:
  missing input, shape mismatch (`Linear.in_features=512 but input last dim is 768`),
  divisibility (`d_model % num_heads != 0`), RoPE needs even `head_dim`, `seq_len > context_length`,
  residual Add shapes differ, cycle detected.
- Nodes downstream of an error show "unknown shape" (grey/dashed) rather than cascading red.
- UI: error node gets red border + ⚠ badge; message in drawer and on hover.

### 2.3 Groups & semantic zoom (R5)
Composite parts (**Transformer Block**, **Multi-Head Self-Attention**, **SwiGLU FFN**) are
*subgraphs*:
- A group node has its own outer `in` / `out` ports. Inside, two small proxy nodes
  (`Block In`, `Block Out`) bridge the outer ports to the inner nodes. External edges always
  attach to the group's outer ports, so collapsing/expanding never rewires anything.
- Shape inference stays flat: `flattenGroups` redirects edges on a group's outer ports onto its proxies.
- **Frame size is fixed** whether collapsed or expanded. Collapsed = the frame renders a summary
  card (name, `×N`, params, in/out shape) and children are hidden. This avoids re-layout jumps.
- **Level of detail by zoom** (thresholds in `src/canvas/lod.ts`, tuned for the default graph):
  - zoom < 0.25 → Transformer Block shown as one card
  - 0.25–0.6 → block internals visible (RMSNorm, MHA card, Add, RMSNorm, SwiGLU card, Add)
  - ≥ 0.6 → MHA / SwiGLU internals visible (Q/K/V proj, split heads, RoPE, SDPA, merge, out proj …)
- Each group has a manual override: *auto / always expanded / always collapsed*.
- **Layers:** each layer is its **own Transformer Block group** on the canvas (no `×N` repeat).
  `num_layers` is **derived** = number of Transformer Block groups in the graph (shown read-only in
  the hyperparams panel). Users add a layer by duplicating a block / dragging one from the palette.
  Default graph has **2 blocks** (`Block 1`, `Block 2`).

### 2.4 Canvas UX (R1, R2.5, R2.6, R3)
- Inputs on **top** (hollow circle, blue), outputs on **bottom** (filled circle, green) — flow
  reads top→bottom like the CS336 diagram. Port labels on hover.
- Connect: drag from port to port; dropping anywhere on a node body snaps to its first free
  compatible input (React Flow `connectionRadius` + custom snap). Invalid connections are
  refused with a hint (output→output, input already connected, cycle).
- Click a port → small popover: symbolic shape, concrete shape, dtype, size in bytes.
- Edge labels (toggleable): concrete shape, e.g. `32×256×512`.
- Sticky note / text box: resizable, background color, text color, font size; edit by double-click.
- Keyboard: Delete, Cmd+D duplicate, Space+drag pan. (Undo/redo deferred.)

### 2.5 Layout of the app
```
┌─────────────────────────── Toolbar ─────────────────────────────────────────┐
│ Hyperparams ▾ | Mode: [Fwd | Fwd+Bwd | Train] | dtype | Params 22.7M | Mem 1.4 GB │
├────────┬──────────────────────────────────────────────────────┬─────────────┤
│Palette │                    Canvas                            │ Detail      │
│(drag)  │                                                      │ drawer (x)  │
│        ├──────────────────────────────────────────────────────┤             │
│        │ Analysis panel (collapsible): params & memory breakdown│             │
└────────┴──────────────────────────────────────────────────────┴─────────────┘
```

---

## 3. Node catalog (v1)

Mirrors CS336 `cs336_basics`. Shapes use `B`=batch, `T`=seq len, `d`=d_model, `H`=num_heads,
`dh`=d/H, `F`=d_ff, `V`=vocab.

| Part | Params (editable) | In → Out | Params count | CS336 file |
|---|---|---|---|---|
| Data Batch | batch_size, seq_len | — → `input_ids B×T`, `targets B×T` (int) | 0 | DataLoading.py |
| Embedding | vocab_size, d_model | `B×T` → `B×T×d` | V·d | Embedding.py |
| RMSNorm | d_model, eps | `B×T×d` → same | d | RMSNorm.py |
| Linear | in_features, out_features (no bias) | `…×in` → `…×out` | in·out | Linear.py |
| Split Heads | num_heads | `B×T×d` → `B×H×T×dh` | 0 | MHA reshape |
| RoPE | head_dim, theta, max_seq_len | `B×H×T×dh` → same | 0 (buffers only) | RotaryPositionalEmbedding.py |
| Scaled Dot-Product Attn | causal (bool) | Q,K,V `B×H×T×dh` → `B×H×T×dh` | 0 | ScaledDotProductAttention.py |
| Merge Heads | — | `B×H×T×dh` → `B×T×d` | 0 | MHA reshape |
| SiLU | — | same → same | 0 | SwiGLU.silu |
| Multiply (⊙) | — | 2 same-shape → same | 0 | SwiGLU |
| Add (residual) | — | 2 same-shape → same | 0 | TransformerBlock |
| Softmax | dim | same → same | 0 | Softmax.py |
| Logits | — | `B×T×V` → `B×T×V` (labelled pass-through marking the model output) | 0 | TransformerLM output |
| Cross-Entropy | — | logits `B×T×V` + targets `B×T` → loss (scalar) | 0 | CrossEntropy.py |
| Loss | — | scalar sink | 0 | |
| **Group:** Multi-Head Self-Attention | d_model, num_heads, rope on/off | `B×T×d` → `B×T×d` | 4·d² | MultiHeadSelfAttention.py |
| **Group:** SwiGLU FFN | d_model, d_ff | `B×T×d` → `B×T×d` | 3·d·F | SwiGLU.py |
| **Group:** Transformer Block | — | `B×T×d` → `B×T×d` | 4d² + 3dF + 2d | TransformerBlock.py |
| Sticky note / Text box | bg color, text color, font size | — | — | |

**Default graph (CS336 TransformerLM):**
`Data Batch.input_ids → Embedding → Block 1 → Block 2 → RMSNorm (ln_final) → Linear (lm_head, d→V) → Logits → Cross-Entropy (+ Data Batch.targets) → Loss`

Default hyperparams = CS336 `train.py` except **2 layers** (user decision):
`vocab_size=10000, context_length=256, d_model=512, num_heads=16, d_ff=1344, rope_theta=10000, batch_size=32, dtype=fp32`.

Sanity check (unit test): default params = 5,120,000 (emb) + 2 × 3,113,984 (blocks) + 512 (ln_final)
+ 5,120,000 (lm_head) = **16,468,480** (no weight tying, as in CS336). With 4 blocks it would be 22,696,448.

---

## 4. Parameter accounting (R7)
- Per node: total + list of weight tensors with shapes (e.g. `W1: 1344 × 512`).
- Groups: sum of children.
- Analysis panel: total with the formula written out
  `V·d + L·(4d² + 3dF + 2d) + d + d·V`, and a stacked bar by category:
  Embedding / Attention / FFN / Norms / LM head. Clicking a category highlights those nodes on canvas.
- **Model = connected parts** (`src/engine/params.ts`): a part counts when it feeds a Logits or Loss part
  (reverse walk over the flattened graph). Others are "unconnected" — shown as `+X in N unconnected parts,
  not counted` (toolbar, summary, panel, amber badges). No Logits/Loss on the canvas → everything counts.
- Categories: Embedding (Embedding parts), Attention (inside an MHA group), FFN (inside a SwiGLU group),
  Norms (RMSNorm), LM head (a Linear feeding Logits), Other (anything else with weights).
- `L` in the formula = connected Transformer Blocks. The formula "matches" only when every category equals
  its term; otherwise it is labelled "standard CS336 formula" and the per-category differences are listed
  (the per-part sum is always the real count).

## 5. Memory estimation (R8)
Educational estimate, not allocator-exact (`src/engine/memory.ts`). `bytes(dtype)` = 4 (fp32) / 2 (bf16/fp16);
int64 tensors (token ids, targets) are always 8. P = connected params (§4). Only **connected** parts count
(`ParamReport.connectedIds`).

| Component | Fwd (inference) | Fwd+Bwd | Train (AdamW) |
|---|---|---|---|
| Weights | P·b | P·b | P·b |
| Gradients | – | P·b | P·b |
| Optimizer (m, v) | – | – | 2·P·b (CS336 uses `zeros_like(p)` → same dtype as p) |
| Activations | peak live set: max over parts of (unique inputs + outputs + internal temporaries such as the attention probs) | sum of tensors **saved for backward** | same as Fwd+Bwd |
| Buffers | RoPE cos + sin, **once per attention group** (`2 · max_seq_len · d_head/2 · b`) | same | same |

RoPE buffers: CS336 builds one `RotaryPositionalEmbedding` per MHA and uses it for q and k; the canvas has two
RoPE parts per attention group, so RoPE parts inside the same MHA group share one cos/sin pair (a RoPE outside an
MHA group counts on its own). Default: 2 · 2 · 256 · 16 · 4 = 65,536 B.

**Saved-for-backward** is declared per node type and counted **once per unique tensor**. A tensor is identified by
its producing part + output port after skipping pass-throughs (group in/out proxies, Logits); internal tensors by the
part that creates them. So the RMSNorm output feeding Q, K and V is stored once — a nice teaching point.
Linear → its input; RMSNorm → input + rms; SiLU → input; Multiply → both inputs;
SDPA → Q, K, V and the attention probabilities `B×H×T×T`; Softmax → output;
Cross-Entropy → logits `B×T×V` + targets; Embedding → token ids; Add / reshape / RoPE → nothing.
Attribution: a tensor belongs to the part that **produced** it (internal ones: the part that saves them; Data Batch
outputs: the part that saves them). Categories: Attention probs (B·H·T·T), Attention other, FFN, Norms, Embedding,
Logits / loss, Residual / other. Rows = top-level group (Transformer Block) or the part itself.

**Activation checkpointing toggle** (CS336 `checkpoint_blocks`, one `torch.utils.checkpoint` per block; ignored in
Forward mode): tensors whose savers all sit inside one Transformer Block are dropped; each block's input is kept
(`L·B·T·d`), and the largest block's saved tensors are added once (recomputed during backward; its input is not
double-counted). Everything outside the blocks is saved as usual.

Defaults (fp32, 2 blocks): Forward 410.4 MB (peak at lm_head: input + logits = 344.5 MB); Fwd+Bwd 1.38 GB
(activations 1,250,721,792 B); Train 1.51 GB; Train + checkpointing 1.09 GB (activations 822,837,248 B).

**Breakdown views (Memory tab):** (a) by component with formulas, (b) activations by category (click = highlight on
canvas), by layer and the biggest tensors (click = focus) — with defaults the attention probabilities
(`32·16·256·256·4B ≈ 134 MB/layer`) and the logits (`32·256·10000·4B ≈ 328 MB`) dominate, which is exactly the
"where to optimise" insight. Each node's drawer "Size" section shows its own contribution. Mode and checkpointing are
UI state (not saved in the document); dtype is a hyperparam (saved).

Not modelled: allocator overhead / fragmentation, CUDA context, transient activation gradients, extra autograd
intermediates of hand-written ops (CS336's SDPA/softmax/cross-entropy keep several B·H·T² / B·T·V temporaries in
real PyTorch), fused kernels (FlashAttention), mixed-precision master weights.

---

## 6. Phases

Each phase ends with: app runs, tests pass, PROGRESS.md updated.

### Phase 0 — Docs ✅
- [x] CLAUDE.md, REQUIREMENTS.md, PLAN.md, PROGRESS.md

### Phase 1 — Scaffold & basic canvas (R1, R3) ✅
- [x] Vite + React + TS + Tailwind + React Flow + Zustand + Vitest; `git init`
- [x] App shell layout (toolbar, palette, canvas, drawer, analysis panel placeholders)
- [x] Pan/zoom/fit/minimap; drag from palette; select/move/delete/duplicate
- [x] Sticky note & text box (resizable, colors, font size, inline edit)
- [x] Autosave to localStorage; export/import JSON

### Phase 2 — Engine: node types + shape inference (R2, R6) ✅
- [x] `engine/types`, `hyperparams`, `resolve`, `infer` (+ `shape` helpers) + unit tests
- [x] Primitive node defs (table in §3) and generic node component (ports, title, shape, param count badge)
      — every def also has `paramCount`, `savedForBackward` and short `docs` (expanded in Phase 4)
- [x] Port styles, connection validation, snap-to-body connect
- [x] Error highlighting + messages; unknown-shape propagation
- [x] Port click popover & edge shape labels
- [x] Hyperparameters panel (global), param binding/override (`num_layers` shown read-only, derived in Phase 3)
- [x] Default CS336 graph, **flat** (no groups yet)

### Phase 3 — Groups & semantic zoom (R5) ✅
- [x] Subgraph groups with outer ports + inner proxies; inference via `flattenGroups` (engine stays flat)
- [x] MHA, SwiGLU, Transformer Block templates (in palette too); `num_layers` derived from block count
- [x] Zoom-based LOD (3 levels, thresholds 0.25 / 0.6) + per-group override; fixed frame size
- [x] Default graph switched to grouped version

### Phase 4 — Detail drawer (R4) ✅
- [x] Drawer with sections: Overview, Parameters (editable, 🔗 binding), Shapes, Size, Points to remember, Formula
      (+ header with status badge, errors on top, CS336 reference; collapsible sections). Size shows params only —
      the per-part memory sub-section is a Phase 6 hook (`MemoryContribution` in `src/panels/drawer/SizeSection.tsx`).
- [x] Docs content for every node type and group (from CS336 code + handout)
- [x] Empty-selection state = model summary

### Phase 5 — Parameter accounting (R7) ✅
- [x] `engine/params` + tests (16,468,480 for the 2-block default; formula gives 22,696,448 for 4 blocks)
- [x] Per-node badge, analysis panel total + formula + category bar, click-to-highlight
      (+ per-layer list with click-to-focus, insights, connected-model rule, Analysis tabs ready for Memory)

### Phase 6 — Memory estimation (R8) ✅
- [x] `engine/memory` + tests
- [x] Mode / dtype / activation-checkpointing controls (toolbar; Mem chip opens the Memory tab)
- [x] Breakdown by component and by part; per-node contribution in drawer; optional "heat" tint on nodes by activation memory
      (+ formulas, activations by category/layer, top tensors, "where to optimise" insights, category highlight)

### Phase 7 — Extras (approved 2026-10-05)
- [ ] 7a. Removing items made obvious: Delete button in drawer header, right-click context menu (delete / duplicate / …), delete for edges too (Delete/Backspace already works)
- [ ] 7a. Undo/redo (Cmd+Z / Shift+Cmd+Z + toolbar buttons) covering all graph edits
- [ ] 7a. Drop a connection on empty canvas → quick-add menu that creates a part and auto-connects it
- [ ] 7b. User-made visual frames (Miro-style, no ports): titled, coloured, resizable, moves what's inside
- [ ] 7c. Extra part variants: LayerNorm, GELU, ReLU, non-gated FFN (CS336 `SiLU.py`) + docs; weight-tying toggle (lm_head shares embedding)
- [ ] 7d. KV-cache estimate for generation (memory)

---

## 7. Simplifications / dropped (for viability)
1. ~~Layers as one ×N block~~ — reversed: N separate blocks (user decision).
2. **Fixed-size group frames** across zoom levels — no automatic re-layout.
3. **Only predefined composite groups** (Block / MHA / SwiGLU) have ports and LOD. User-made grouping
   is a stretch item and would be visual-only frames.
4. **Memory is an estimate**: no allocator fragmentation, no fused-kernel savings (e.g. FlashAttention),
   no mixed-precision master weights, single dtype for everything. Formulas are shown so it stays honest.
5. Tokenizer/BPE, data loading, LR schedule, gradient clipping are **not** canvas parts (they don't
   change shapes/params). AdamW appears only as the optimizer-memory setting.
6. No collaboration/cloud save; localStorage + JSON file.

## 8. Decisions (2026-10-04)
- D1. Layers = **N separate Transformer Block groups**, default **2**. `num_layers` is derived from the graph.
- D2. `batch_size` is a global hyperparam (default 32).
- D3. Default graph ends `… → Logits → Cross-Entropy (+ targets) → Loss`.
- D4. ~~Phase 7 skipped~~ → Phase 7 approved on 2026-10-05, plus a visible way to remove items.
- D5. Work is delegated to subagents phase by phase; commit locally after each phase, **never push**.
- D6. Only read files inside this project and the CS336 folder — nowhere else on the machine.
