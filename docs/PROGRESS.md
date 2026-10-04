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
