# LLM Canvas — Implementation Plan

> **Status: DRAFT — awaiting approval.** Do not start Phase 1 until the user approves.
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
    docs: { overview, formula?, pointsToRemember[], paramHelp{} , cs336Ref }
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
- Shape inference recurses into the subgraph.
- **Frame size is fixed** whether collapsed or expanded. Collapsed = the frame renders a summary
  card (name, `×N`, params, in/out shape) and children are hidden. This avoids re-layout jumps.
- **Level of detail by zoom** (thresholds tunable):
  - zoom < ~0.5 → Transformer Block shown as one card
  - 0.5–1.1 → block internals visible (RMSNorm, MHA card, Add, RMSNorm, SwiGLU card, Add)
  - \> 1.1 → MHA / SwiGLU internals visible (Q/K/V proj, split heads, RoPE, SDPA, merge, out proj …)
- Each group has a manual override: *auto / always expanded / always collapsed*.
- **Layers:** the Transformer Block group has a `repeat` param bound to `num_layers` and shows a
  stacked-card "×4" look. Params/memory multiply by `repeat`. (Users can still duplicate a block
  and set `repeat=1` on each if they want to see them separately.)

### 2.4 Canvas UX (R1, R2.5, R2.6, R3)
- Inputs on **top** (hollow circle, blue), outputs on **bottom** (filled circle, green) — flow
  reads top→bottom like the CS336 diagram. Port labels on hover.
- Connect: drag from port to port; dropping anywhere on a node body snaps to its first free
  compatible input (React Flow `connectionRadius` + custom snap). Invalid connections are
  refused with a hint (output→output, input already connected, cycle).
- Click a port → small popover: symbolic shape, concrete shape, dtype, size in bytes.
- Edge labels (toggleable): concrete shape, e.g. `32×256×512`.
- Sticky note / text box: resizable, background color, text color, font size; edit by double-click.
- Keyboard: Delete, Cmd+D duplicate, Cmd+Z/Shift+Cmd+Z undo/redo (Phase 7), Space+drag pan.

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
| Token Input | batch_size, seq_len | — → `B×T` (int) | 0 | DataLoading.py |
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
| Cross-Entropy Loss | — | logits `B×T×V` + targets `B×T` → scalar | 0 | CrossEntropy.py |
| Output / Logits | — | sink | 0 | |
| **Group:** Multi-Head Self-Attention | d_model, num_heads, rope on/off | `B×T×d` → `B×T×d` | 4·d² | MultiHeadSelfAttention.py |
| **Group:** SwiGLU FFN | d_model, d_ff | `B×T×d` → `B×T×d` | 3·d·F | SwiGLU.py |
| **Group:** Transformer Block | repeat (=num_layers) | `B×T×d` → `B×T×d` | 4d² + 3dF + 2d | TransformerBlock.py |
| Sticky note / Text box | bg color, text color, font size | — | — | |

**Default graph (CS336 TransformerLM):**
`Token Input → Embedding → [Transformer Block ×num_layers] → RMSNorm (ln_final) → Linear (lm_head, d→V) → Cross-Entropy (+ targets) / Logits`

Sanity check (unit test): default params = 5,120,000 (emb) + 4 × 3,113,984 (blocks) + 512 (ln_final)
+ 5,120,000 (lm_head) = **22,696,448** (no weight tying, as in CS336).

---

## 4. Parameter accounting (R7)
- Per node: total + list of weight tensors with shapes (e.g. `W1: 1344 × 512`).
- Groups: sum of children × `repeat`.
- Analysis panel: total with the formula written out
  `V·d + L·(4d² + 3dF + 2d) + d + d·V`, and a stacked bar by category:
  Embedding / Attention / FFN / Norms / LM head. Clicking a category highlights those nodes on canvas.

## 5. Memory estimation (R8)
Educational estimate, not allocator-exact. `bytes(dtype)` = 4 (fp32) / 2 (bf16/fp16). P = total params.

| Component | Fwd (inference) | Fwd+Bwd | Train (AdamW) |
|---|---|---|---|
| Weights | P·b | P·b | P·b |
| Gradients | – | P·b | P·b |
| Optimizer (m, v) | – | – | 2·P·b (CS336 uses `zeros_like(p)`) |
| Activations | peak live tensors only (max over nodes of inputs+output) | sum of tensors **saved for backward** | same as Fwd+Bwd |
| Buffers | RoPE cos/sin per layer | same | same |

**Saved-for-backward** is declared per node type and counted **once per unique tensor** (e.g. the
RMSNorm output feeding Q, K and V is stored once — a nice teaching point):
Linear → its input; RMSNorm → input (+ rms); SiLU → input; Multiply → both inputs;
SDPA → Q, K, V and the attention probabilities `B×H×T×T`; Softmax → output;
Cross-Entropy → logits `B×T×V`; Add / reshape / RoPE → nothing.

**Activation checkpointing toggle** (CS336 `checkpoint_blocks`): saved = each block's input
(`L·B·T·d`) + one block's full saved activations (recomputed during backward).

**Breakdown views:** (a) by component (weights/grads/optimizer/activations), (b) activations by
part — e.g. with defaults the attention probabilities (`32·16·256·256·4B ≈ 134 MB/layer`) and the
logits (`32·256·10000·4B ≈ 328 MB`) dominate, which is exactly the "where to optimise" insight.
Each node's drawer "Size" section shows its own contribution.

---

## 6. Phases

Each phase ends with: app runs, tests pass, PROGRESS.md updated.

### Phase 0 — Docs ✅
- [x] CLAUDE.md, REQUIREMENTS.md, PLAN.md, PROGRESS.md

### Phase 1 — Scaffold & basic canvas (R1, R3)
- [ ] Vite + React + TS + Tailwind + React Flow + Zustand + Vitest; `git init`
- [ ] App shell layout (toolbar, palette, canvas, drawer, analysis panel placeholders)
- [ ] Pan/zoom/fit/minimap; drag from palette; select/move/delete/duplicate
- [ ] Sticky note & text box (resizable, colors, font size, inline edit)
- [ ] Autosave to localStorage; export/import JSON

### Phase 2 — Engine: node types + shape inference (R2, R6)
- [ ] `engine/types`, `hyperparams`, `resolve`, `infer` + unit tests
- [ ] Primitive node defs (table in §3) and generic node component (ports, title, shape, param count badge)
- [ ] Port styles, connection validation, snap-to-body connect
- [ ] Error highlighting + messages; unknown-shape propagation
- [ ] Port click popover & edge shape labels
- [ ] Hyperparameters panel (global), param binding/override
- [ ] Default CS336 graph, **flat** (no groups yet)

### Phase 3 — Groups & semantic zoom (R5)
- [ ] Subgraph groups with outer ports + inner proxies; recursive inference
- [ ] MHA, SwiGLU, Transformer Block templates (in palette too); `repeat` ×N with stacked look
- [ ] Zoom-based LOD (3 levels) + per-group override; fixed frame size
- [ ] Default graph switched to grouped version

### Phase 4 — Detail drawer (R4)
- [ ] Drawer with sections: Overview, Parameters (editable, 🔗 binding), Shapes, Size, Points to remember, Formula
- [ ] Docs content for every node type (from CS336 code + handout)
- [ ] Empty-selection state = model summary

### Phase 5 — Parameter accounting (R7)
- [ ] `engine/params` + tests (22,696,448 default)
- [ ] Per-node badge, analysis panel total + formula + category bar, click-to-highlight

### Phase 6 — Memory estimation (R8)
- [ ] `engine/memory` + tests
- [ ] Mode / dtype / activation-checkpointing controls
- [ ] Breakdown by component and by part; per-node contribution in drawer; optional "heat" tint on nodes by activation memory

### Phase 7 — Polish (stretch, pick as time allows)
- [ ] Undo/redo (zundo)
- [ ] Drop a connection on empty canvas → quick-add menu that auto-connects
- [ ] Plain visual "frames" for user-made grouping (Miro-style, no ports)
- [ ] Extra parts for experimentation: LayerNorm, GELU/ReLU FFN, non-gated SiLU FFN (CS336 `SiLU.py`), weight tying toggle
- [ ] KV-cache estimate for generation

---

## 7. Simplifications / dropped (for viability)
1. **Layers shown as one block ×N**, not N separate copies (duplicating is still possible).
2. **Fixed-size group frames** across zoom levels — no automatic re-layout.
3. **Only predefined composite groups** (Block / MHA / SwiGLU) have ports and LOD. User-made grouping
   is a stretch item and would be visual-only frames.
4. **Memory is an estimate**: no allocator fragmentation, no fused-kernel savings (e.g. FlashAttention),
   no mixed-precision master weights, single dtype for everything. Formulas are shown so it stays honest.
5. Tokenizer/BPE, data loading, LR schedule, gradient clipping are **not** canvas parts (they don't
   change shapes/params). AdamW appears only as the optimizer-memory setting.
6. No collaboration/cloud save; localStorage + JSON file.

## 8. Open questions (defaults assumed if not answered)
- Q1. Layers as one `×N` block (default) vs N separate block copies on canvas?
- Q2. Include `batch_size` in global hyperparams (default yes, 32 from `train.py`).
- Q3. Show the loss (Cross-Entropy + targets) in the default graph, or end at logits? (default: include it, since it matters for training memory.)
- Q4. Stretch items in Phase 7 — any you want promoted into the core?
