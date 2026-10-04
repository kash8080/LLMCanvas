# Progress Log

Append one entry per working session (newest at the bottom). Keep it short: what was done,
what's next, any decisions or gotchas. Tick boxes in [PLAN.md](PLAN.md) as phases progress.

---

## 2026-10-04 — Session 1: Planning
- Read CS336 reference implementation (`cs336_basics`): TransformerLM, TransformerBlock (pre-norm),
  MHA (+RoPE), SDPA (causal), SwiGLU, RMSNorm, Linear (no bias), Embedding, CrossEntropy, AdamW,
  and `train.py` defaults.
- Wrote CLAUDE.md, docs/REQUIREMENTS.md, docs/PLAN.md (draft), this log.
- **Next:** user reviews PLAN.md → answer open questions (§8) → start Phase 1.

## 2026-10-04 — Session 2: Phase 1 (scaffold & basic canvas)
**Done**
- Hand-written Vite scaffold (no `create vite`, dir wasn't empty). Versions: Vite 8, React 19, TS 7,
  `@xyflow/react` 12.12, zustand 5, Tailwind 4 (via `@tailwindcss/vite`, just `@import 'tailwindcss'`
  in `src/index.css`), lucide-react, Vitest 5. Single `tsconfig.json` (noEmit, bundler resolution);
  `pnpm build` = `tsc && vite build`. Vitest config lives in `vite.config.ts` (`src/**/*.test.ts`, node env).
- App shell: Toolbar (top) / Palette (left) / Canvas / AnalysisPanel (bottom, collapsible placeholder) /
  DetailDrawer (right, closable; opens on node click; shows type/id + sticky/text settings).
- Canvas: dots background, Controls, MiniMap, drop-from-palette at cursor (`screenToFlowPosition`),
  Delete/Backspace removes, Cmd/Ctrl+D duplicates selection (incl. edges between duplicated nodes),
  `connectionRadius=40`, self-connections refused.
- Nodes: `sticky`, `textbox` (shared `AnnotationNode`: NodeResizer when selected, double-click → textarea,
  Esc/blur ends edit, bg/text colour swatches + font-size slider in drawer) and a temporary
  `placeholder` part (1 input top, 1 output bottom) to demo connecting.
- Ports: `src/canvas/Port.tsx` → inputs top hollow blue, outputs bottom filled green (CSS in `index.css`).
- Store `src/store/useCanvasStore.ts` (nodes, edges, drawerOpen, analysisOpen + actions). Autosave:
  `startAutosave()` (debounced 400 ms) → `localStorage["llm-canvas:v1"]`, restored on load.
  `src/store/persistence.ts` = pure document format `{app:'llm-canvas', version:1, nodes, edges}`,
  `toDocument` (strips selected/measured), `parseDocument` (validates imports, readable errors).
  Toolbar: Export JSON, Import JSON, Reset (confirm → tiny demo in `src/defaults/demoGraph.ts`).
- Tests: `src/engine/format.test.ts` (formatCount / formatBytes helpers for later phases) and
  `src/store/persistence.test.ts`. `pnpm test` 4/4 pass, `pnpm build` passes. Manually checked in
  browser: drop, connect, select, multi-select, duplicate, delete, edit, colours, autosave/reload,
  import, reset.

**Layout notes**
- `src/canvas/types.ts` holds `NODE_KINDS` / `AppNode` union — Phase 2 must add its node kind(s) there
  (persistence rejects unknown `type`s). `nodeFactory.ts` creates nodes + `DEFAULT_SIZE`; `nodeTypes.ts`
  maps kinds → components.
- `src/nodes/registry.ts` is an empty placeholder; `src/engine/` only has `format.ts` so far.

**Gotchas / decisions**
- Navigation is Miro/trackpad style: drag empty canvas = pan, two-finger scroll = pan (`panOnScroll`),
  pinch or Cmd/Ctrl+wheel = zoom, Shift+drag = box select, Shift/Cmd+click = add to selection,
  double-click zoom disabled (double-click is for editing). A plain mouse wheel pans instead of zooming.
- React Flow tracks modifier keys via keydown, so synthetic clicks with only `shiftKey` don't multi-select
  in automated tests — real keyboard works.
- React Flow attribution kept (moved to top-right); hiding it is meant for Pro subscribers.
- `.claude/launch.json` has a "dev" config for the preview browser.
- Demo graph replaced by the CS336 default in Phase 2 (`resetCanvas` / initial state use `demoDocument()`).

**Next:** Phase 2 — engine types/hyperparams/resolve/infer + tests, primitive node defs, generic part
node component, connection validation, default flat CS336 graph.

## 2026-10-04 — Session 3: Phase 2 (engine, node types, shape inference)
**Done**
- Engine (`src/engine/`, pure TS): `types.ts` (Dim/Shape with `dims`+`dtype`, ParamValue bind/value,
  ParamSchema, NodeDef, GraphModel, NodeResult/InferenceResult), `hyperparams.ts` (CS336 defaults,
  symbols B/T/d_model/H/d_head/d_ff/V/θ, derived `d_head`, `validateHyperparams`, bytes per dtype),
  `resolve.ts` (bind vs override, local-value validation, `num`/`paramDim` helpers), `shape.ts`
  (numel, bytes, `32×256×512` / `B × T × d_model` formatting), `infer.ts` (`inferShapes(graph, hp, defs)`
  + `wouldCreateCycle`).
- 15 node defs in `src/nodes/` (one file each) + `registry.ts` (`NODE_DEFS`, `nodeRegistry`, `getNodeDef`,
  `CATEGORY_INFO` colours, `CATEGORY_ORDER`). Every def has `infer`, `paramCount`, `savedForBackward`
  (per PLAN §5) and short `docs`.
- UI: generic `PartNode` (category strip, labelled multi-ports, shape line, param badge, red error +
  hover tooltip, dashed/greyed unknown), `ShapeEdge` (smooth-step + concrete-shape label, dashed when
  unknown), `PortPopover` (symbolic, concrete, dtype, bytes), toolbar Hyperparams ▾ popover (+ d_head,
  read-only num_layers, problems list), "Shapes on edges" toggle, live Params total, palette from the
  registry, drawer part editor (rename, 🔗 bound value + Unlink, local editor + Relink, errors, shapes,
  weights).
- Connection rules (`src/canvas/connect.ts`): output→input only, connecting to an occupied input
  **replaces** its edge, no cycles (refused with a hint toast), `connectOnClick=false` so a click on a
  port opens the popover. Dropping on a node body connects to the first free input (or replaces the
  only input); `connectionRadius=50`.
- Default graph `src/defaults/cs336Graph.ts` (flat, 2 blocks, 47 parts / 57 edges) used for first load
  and Reset. Persistence bumped to **version 2** (adds `hyperparams`; part nodes are `type:'part'` with
  `data: {partType, params, title?}`); v1 saves are rejected → default graph.
- Tests: 22 passing (engine: default-graph shapes, SDPA probs `32×16×256×256`, total params
  **16,468,480**, Linear mismatch + downstream unknown, d_model % H, odd head dim, seq_len > ctx,
  missing input, Add mismatch, cycles, local param validation; persistence; connection rules).
  `pnpm build` passes. Checked in the browser: default graph all-green with edge shapes, d_model=500 →
  3 split-heads errors + 38 unknown, q_proj in_features unlink → 768 → mismatch error + downstream
  unknown, relink, port popover, replace-edge, cycle hint, body-drop, palette drop, edge-label toggle.

**Layout notes**
- Inference is derived in the store: every graph mutation goes through `commit()` in
  `useCanvasStore.ts`, which calls `inferCanvas()` (`src/store/inference.ts`). That memoises on node
  `data` refs + `edges` + `hyperparams`, so dragging nodes doesn't re-infer. Components read
  `s.inference.nodes[id]` / `s.inference.edges[id]`. No `useEffect`.
- Node param values live in `node.data.params` (full set from schema defaults on creation; missing keys
  fall back to the schema default in `resolveParams`).
- Default graph layout: residual stream straight down column x=0 (Adds there), branches to the right
  (q/k/v at 190/420/650), Data Batch + Cross-Entropy + Loss on a lane at x=900 so `targets` drops
  straight down. Rows are 110 px; `PART_WIDTH` = 176. Block labels are plain text boxes (Phase 3 groups
  replace them). Initial fit/Reset fit to the top 14 nodes (`topNodes`) so the start is readable.
- `ParamTensor.dims` for Linear are `out × in` (CS336 storage).

**Gotchas**
- React Flow picks the closest handle within `connectionRadius` of *any* type; a drop near a target's
  own output handle comes back as invalid with `toHandle.type === fromHandle.type` — `onConnectEnd`
  treats that as a body drop.
- Built-in browser automation: ref-based clicks on canvas nodes don't select them; coordinate clicks do.
  Controls zoom animates, so wait before reading handle positions.
- Embedding saves its int64 token ids and Cross-Entropy saves logits + targets for backward (small
  int64 tensors; PLAN §5 only lists logits) — Phase 6 should use each tensor's dtype for bytes.
- Toolbar Params total currently sums every part on the canvas (including stray unconnected ones).

**Intended approach for Phase 3 (groups)**
- Keep the engine flat. A group (Transformer Block / MHA / SwiGLU) is a React Flow parent node
  (`type: 'group'`, children have `parentId`); it is *not* a part, so `toGraphModel` skips it (it
  already passes `parentId` through).
- Each group gets inner proxy parts, ordinary NodeDefs: `group_input` (no inputs → one output) and
  `group_output` (one input → no outputs), both identity/0-params, one pair per outer port.
- Before inference, a small `flattenGroups(graph)` rewrites edges: an edge targeting group G's outer
  port `p` is redirected to G's `group_input[p]` proxy (which then takes it as its input), and an edge
  sourced from G's outer port `p` is re-sourced from G's `group_output[p]` proxy. Proxies therefore
  need a pass-through input/output on the engine side even though the UI shows only one side.
  Outer-port shapes for the group card = the proxies' shapes; group param count = sum of children.
- Collapse/LOD is purely visual (hide children), so inference never changes with zoom.
- `num_layers` = count of `transformer_block` groups; show it in the Hyperparams popover.

**Next:** Phase 3 — groups & semantic zoom.

## 2026-10-04 — Session 4: Phase 3 (groups & semantic zoom)
**Done**
- Engine: `src/engine/groups.ts` — `flattenGroups(graph)` redirects edges on a group's outer ports
  (`<groupId>.in` / `.out` handles) onto its `group_input` / `group_output` proxies; `inferShapes` is unchanged.
  Proxies are ordinary identity NodeDefs (`src/nodes/groupProxy.ts`, in `nodeRegistry` but not the palette).
- Groups: `src/nodes/groups.ts` (GroupDef: label, colour, `expandAtLod`, short docs; `GROUP_PORTS` = one `in`,
  one `out`). React Flow node kind `'group'` with `data: {groupType, title?, mode: 'auto'|'expanded'|'collapsed'}`
  and fixed `width`/`height`; children have `parentId` + `extent: 'parent'`.
- Templates: `src/canvas/groupTemplates.ts` (`buildGroup` for transformer_block / mha / swiglu, nested; layout
  constants + `GROUP_LAYOUT` sizes; `instantiateGroup` for the palette; `numLayers`, `nextBlockTitle`).
  Default graph (`cs336Graph.ts`) now = Data → Embedding → Block 1 → Block 2 → ln_final → lm_head → Logits →
  Cross-Entropy (+targets) → Loss, built from the same templates (66 nodes / 69 edges). Text-box block labels removed;
  welcome sticky mentions zooming.
- Inference bridge (`src/store/inference.ts`): `inferGraph` = flatten + infer + `summarizeGroups` →
  `inference.groups[id]` = {params (sum of descendants), status, errors (prefixed `attn › q_proj: …`), in/out shape
  (from proxies), contains, per-child breakdown}.
- LOD (`src/canvas/lod.ts`): levels by zoom (<0.25 blocks are cards; <0.6 MHA/SwiGLU are cards; else all open),
  read via `useStore(lodSelector)` (bucketed → re-render only on level change). `applyLod` / `hideEdgesOfHiddenNodes`
  set `hidden` on the nodes/edges passed to React Flow (useMemo in Canvas; store nodes stay clean).
- `GroupNode` (`src/canvas/nodes/GroupNode.tsx`): expanded = tinted frame + header (title, label, error count, params,
  Auto/Open/Closed toggle); collapsed = summary card (text scales with frame width; shapes, params, contains, problems).
  Outer ports grow when zoomed out (56/26/16 px) so cards can be wired. Drawer: `GroupDetails` (rename, display mode,
  shapes, weights breakdown). Proxies render as small dashed pills (`in 32×256×512`).
- Store: palette drop of `group:<type>` instantiates the whole template (new blocks titled "Block N"); ⌘D copies
  groups with all descendants + internal edges (copy placed to the right; block copies renumbered); deleting a group
  deletes its children (React Flow); a proxy alone can't be deleted (`withoutLoneProxies` via `onBeforeDelete`, hint).
  `setPartTitle` → `setTitle` (parts + groups), new `setGroupMode`.
- Connections (`connect.ts`): `nodePorts()` covers parts, proxies (one side only) and groups; edges must join nodes
  with the same `parentId` (outside ↔ group outer port only, hint otherwise); cycle check runs on the flattened graph.
  Port popover works on group ports (shape from the proxies); edge labels work (flattened edges keep their ids).
- `num_layers` shown in the Hyperparams popover (read-only, `= blocks`) and as "Layers N" in the toolbar.
- Persistence: doc **version 3**; saves `parentId` / `extent`; validates group type/mode/size and parents-before-children.
  v2 (flat) saves fall back to the default graph.
- Tests: 32 passing. New `src/store/groups.test.ts` (flattenGroups; grouped default = no errors, **16,468,480**;
  per-group 3,113,984 / 1,048,576 / 2,064,384; nested error propagation; num_layers = 2; palette block → 3 layers,
  **19,582,464**, "Block 3"; duplicate copies children + internal edges; proxy delete guard). Connect + persistence
  tests extended. `pnpm build` passes.
- Browser-checked at 1440×900: the three LOD levels, manual Open override, dragging a block moves its children,
  ⌘D / Delete of a block, palette drop + wiring a new block in (body drop on a group, port drop on ln_final),
  group port popover, outside→inner refusal hint, proxy delete guard, Reset.

**Layout notes**
- Block: residual column on the left (proxies, Adds), ln1/MHA and ln2/SwiGLU on a branch column to the right.
  Sizes: Block 888×2124, MHA 616×822, SwiGLU 432×602 (all derived from constants in groupTemplates.ts).
  Blocks are placed so their residual column is x = 0; Data Batch / Cross-Entropy / Loss on a lane right of them.
- Start view / Reset / Import fit to the top 4 top-level nodes (welcome, data, embed, Block 1) → ~0.3 zoom at
  1440×900 = level 1. Reset/Import now compute the viewport from declared positions (`startBounds` +
  `getViewportForBounds`): `fitView` waits for freshly loaded nodes to be measured and stalled at level 0.

**Gotchas**
- React Flow's built-in `group` node type comes with default CSS (`.react-flow__node-group` padding/border/bg) —
  overridden in `index.css`.
- Handles must be rendered *after* a positioned card in the node, otherwise the card paints over them and a drag
  from the port moves the node instead.
- Edges touching child nodes get their z-index elevated by React Flow, so inner edges draw above the frame;
  hidden nodes' edges must be hidden explicitly.
- React Flow only deletes children whose parent comes earlier in the array — keep parents first everywhere.
- Parts dropped from the palette onto an expanded group frame become top-level nodes (no re-parenting), so they
  can't be wired to the group's inner parts. Fine for now.
- Group frames are big, so whole-model views are ~0.1–0.15 zoom where plain parts are unreadable; that's the
  fixed-frame trade-off (PLAN §7.2).

**Next:** Phase 4 — detail drawer sections + docs content (parts *and* groups: `GROUP_DEFS[...].docs`).

## 2026-10-04 — Session 5: Phase 4 (detail drawer)
**Done**
- `NodeDocs` (engine/types.ts) extended: `overview`, `roles?` (extra sentence keyed by a part's default title —
  `q_proj`, `ln_final`, `w1`, `RoPE q`, `x + attn`…), `formula?: string[]` (lines), `paramFormula?` (groups),
  `pointsToRemember[]`, `paramHelp?` (per param key, falls back to `ParamSchema.help`), `cs336Ref?: {file, symbol}`.
- Docs rewritten for all 15 parts, both proxies and the 3 groups, checked against `cs336_basics` (pre-norm, RMSNorm
  eps 1e-5 + fp32 upcast + gain init 1, Linear no bias + trunc-normal σ²=2/(in+out), Embedding trunc-normal N(0,1)
  in [−3,3], interleaved RoPE pairs + non-persistent cos/sin buffers, SDPA −∞ mask + max-subtracted softmax,
  SwiGLU w1/w3/w2 + d_ff ≈ 8/3·d rounded to 64, cross-entropy log-sum-exp).
- Drawer (`src/panels/DetailDrawer.tsx` → `src/panels/drawer/*`), 380 px wide, header fixed, body scrolls:
  - `ui.tsx`: `DrawerHeader` (colour chip, editable title, type name, ok/error/unknown badge, close), collapsible
    `Section` (collapsed state per section id in the store → stays collapsed across selections), `ErrorBox`, `Notice`.
  - Parts (`PartDetails.tsx`): Errors → Overview (+ role) → Formula → Parameters (🔗 bound / unlink / relink, help as
    caption) → Shapes (port, symbolic, concrete, dtype, bytes) → Size → Points to remember → CS336 reference.
  - Groups (`GroupDetails.tsx`): same order; "Display" (Auto/Open/Closed) sits where Parameters would be; Size has the
    group's `paramFormula`, total and a per-child breakdown with bars (click a row = select + focus that child).
  - Sticky / text box (`AnnotationDetails.tsx`): same header + "Appearance" section (bg, text colour, font size).
  - Nothing selected (`ModelSummary.tsx`): params / layers / parts / problems, clickable problem list (select +
    zoom to the part), read-only hyperparams + "Edit hyperparameters" (opens the toolbar popover), "How to use".
  - Several selected: clickable list.
- Store: `selectOnly(id)`, `hyperparamsOpen` (popover state lifted from `HyperparamsMenu`), `collapsedSections` +
  `toggleSection`. `useFocusNode()` (drawer) = `selectOnly` + `setCenter` (parts, zoom ≥ 0.9 so Auto groups open) or
  `fitBounds` (groups).
- Test `src/nodes/docs.test.ts`: every registered part (incl. proxies) and group has an overview, ≥ 3 points, only
  documents params it has, and (non-proxies) a CS336 ref. 73 tests pass; `pnpm build` passes.
- Browser-checked at 1440×900: summary, Embedding, attn group → q_proj via breakdown, RoPE q, SDPA, ffn, Block 1,
  Cross-Entropy, sticky note, d_model = 500 → problems list → click focuses `split q`, unknown-shape notice, section
  collapse persisting across selections.

**Layout notes**
- Phase 6 hook: `MemoryContribution({ nodeId })` in `src/panels/drawer/SizeSection.tsx` is rendered at the end of
  both `PartSizeSection` and `GroupSizeSection` and returns `null` for now. Fill it in (e.g. `<SubHeading>Memory</SubHeading>`
  + saved-for-backward tensors / bytes for the current mode); the section header `meta` can also show bytes.
- Phase 5: the part Size section derives its formula from the weight tensors' dim labels
  (`params = d_model · d_model = 512 · 512 = 262,144`); groups use `GROUP_DEFS[t].docs.paramFormula`. The summary's
  total params = sum over `inference.nodes` (same as the toolbar, i.e. includes stray unconnected parts).

**Gotchas**
- Built-in browser `computer` clicks use the *screenshot* coordinate frame (800×500 for a 1440×900 viewport), not CSS px:
  divide `getBoundingClientRect()` values by 1.8.
- Editing a `src/nodes/*` file during `pnpm dev` triggers a full reload (store module re-initialises) — selection is lost.
- `useFocusNode` on a part hidden inside a collapsed group uses its declared size (`PART_WIDTH`, 60 px) since it was
  never measured; a group in manual *Closed* mode stays closed, so the part is selected but not visible.

**Next:** Phase 5 — parameter accounting (`engine/params` + tests, analysis panel total + formula + category bar).

## 2026-10-04 — Session 6: Phase 5 (parameter accounting)
**Done**
- `src/engine/params.ts` (pure): `accountParams({graph (flattened), groups, inference, defs, hp})` → `ParamReport`:
  - **Connected-model rule:** a part counts when it feeds a `logits` / `loss` part (reverse walk over the flattened
    edges). Everything else with weights is `unconnected` (`unconnected`, `unconnectedIds`). No Logits/Loss on the
    canvas → everything counts (`hasOutput: false`).
  - Categories `embedding | attention | ffn | norm | lm_head | other`: Embedding parts; parts inside an `mha` group;
    inside a `swiglu` group; RMSNorm; a Linear whose output goes straight into Logits; anything else.
  - `rows`: per-layer breakdown in topological order; row = top-level ancestor group (Transformer Block = layer)
    or the part itself (token_embeddings, ln_final, lm_head), each with its own per-category split.
  - `groups[id]`: highlight keys inside + connected / unconnected params (for glow and amber "not counted").
  - `formula` (`standardFormula(hp, L, actual)`): `V·d + L·(4d² + 3d·d_ff + 2d) + d + d·V` with substituted numbers;
    `L` = connected blocks; `matches` only if every category equals its term, else `diffs` per category.
  - `paramInsights(report)`: 1–2 factual lines (embedding + LM head share and the block count where blocks would
    dominate; FFN / attention ratio; norms share).
- Store: `inference.params` is derived next to shapes in `inferGraph` (same memo). New UI state: `analysisTab`
  (`'params'`), `openAnalysis(tab)`, `highlight: HighlightKey | null` + `setHighlight`. Esc (Canvas keydown) clears it.
- Analysis panel: header with tabs (`TABS` + `TAB_BODY` in `AnalysisPanel.tsx`), body `h-72`.
  `src/panels/analysis/ParamsTab.tsx`: big total + unconnected note, formula block (✓ matches / ≠ + diff list),
  stacked category bar + legend chips (click = highlight, again = clear), per-layer list with mini stacked bars
  (click = `useFocusNode`), "Unconnected" row (click = highlight them), insights.
- Canvas highlight (`src/canvas/highlight.ts`): matching parts glow in the category colour (ring grows at low zoom
  via the LOD bucket), others dim to 25 %; collapsed group cards containing matches glow, others dim; expanded frames
  without matches dim. A "Highlighting X · N params · Esc" chip shows over the canvas.
- Badges: part/group param badges turn amber with a "not counted" tooltip when unconnected; collapsed card says
  "(not counted)". Toolbar Params chip = connected total (+ amber `+X` for unconnected), click opens the tab.
  Drawer summary uses the connected total; drawer Size shows "x % of the model · Category" or a "not counted" note.
- `PARAM_CATEGORY_INFO` (label, colour, help) in `src/nodes/registry.ts`.
- Tests: `src/engine/params.test.ts` (13): default 16,468,480 + exact categories, rows order, formula string + match,
  group keys, insight text, wired 3rd block → 19,582,464 / L = 3, unwired block → unconnected, stray Linear (even
  fed from the model) doesn't change the total, extra connected Linear → Other + formula mismatch, hyperparam change,
  no-output canvas, 4-layer formula = 22,696,448. 86 tests pass; `pnpm build` passes.
- Browser-checked at 1440×900: numbers on the default graph, Attention / FFN / Embedding / Unconnected highlight
  (incl. collapsed attn and Block cards glowing), Esc, Block 2 row focus, d_model 768 / d_ff 2048 / V 32000 →
  63,311,616 (matches), two stray Linears → "+1.18M", palette block → amber "not counted" until wired, wired in →
  3 layers, L = 3, 70,391,040.

**Layout notes (for Phase 6)**
- Add the Memory tab: extend `AnalysisTab` in `useCanvasStore.ts`, add `{id: 'memory', label, meta}` to `TABS` and a
  body in `TAB_BODY` (`src/panels/AnalysisPanel.tsx`). The toolbar "Mem —" placeholder is still there.
- Reuse the connected rule: P for weights/grads/optimizer = `inference.params.total`; activation sums should iterate
  only parts with `connected` — `ParamReport.parts` only lists weighted parts, so for activations either export the
  `connected` set from `accountParams` (small change) or recompute the same reverse walk.
- Highlighting is generic over `HighlightKey`; a memory "heat" view could add its own keys or a separate state.

**Gotchas**
- Automated HTML5 drag-and-drop from the palette: `left_click_drag` works; a synthetic `DragEvent` with a
  `new DataTransfer()` dispatched on `.react-flow` also works (handy for scripted drops).
- `num_layers` (toolbar / hyperparams) still counts *all* blocks on the canvas, while the formula's `L` counts
  connected blocks — they differ while a new block isn't wired in (the formula note says "connected").
- Ref-based clicks in the browser pane log harmless React Flow `nodrag` errors (`view` is null on synthetic events).

**Next:** Phase 6 — memory estimation (`engine/memory` + tests, mode/dtype/checkpointing controls, Memory tab).
