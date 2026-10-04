# LLM Canvas

Browser-only, Miro-style canvas for learning the Transformer/LLM architecture: place parts,
connect them, and see tensor shapes, parameter counts and memory estimates. Nothing is executed —
we only trace shapes and do arithmetic. Side project for learning: **keep it simple**.

## Read first (every session)
- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — what we're building (requirement IDs R1…R9).
- [docs/PLAN.md](docs/PLAN.md) — architecture, node catalog, formulas, phased checklist.
- [docs/PROGRESS.md](docs/PROGRESS.md) — session log; see the last entry for where we left off.

After every major piece of work (not just at session end): tick boxes in PLAN.md, append an
entry to PROGRESS.md (done / next / gotchas), and make a local git commit. If a decision changes
the plan, update PLAN.md rather than letting it drift.

## Hard rules
- **Never `git push`** or add remotes. Local commits only.
- **Only read files inside this project and the CS336 folder below.** Don't browse anywhere else on the machine.
- Work is delegated to subagents phase by phase; the main session coordinates and verifies.

## Reference model
CS336 Assignment 1 — `/Users/rahul/Documents/code/rahul/cs336/assignment1-basics/cs336_basics`.
The default canvas, shapes, param formulas and docs content must match **that** code
(pre-norm block, RMSNorm, RoPE, SwiGLU, no biases, no weight tying, AdamW) — not the original paper.
Default hyperparams come from its `train.py`, except the default graph has **2** Transformer
Blocks (user decision; `num_layers` is derived from the number of blocks on the canvas).

## Stack
Vite + React + TypeScript, React Flow (`@xyflow/react` v12), Zustand, Tailwind, Vitest, pnpm.

## Conventions
- `src/engine/` is pure TypeScript (no React imports) and is where shape inference, param counting
  and memory math live. It must stay unit-tested.
- One node type = one file in `src/nodes/` holding everything about it (ports, params schema,
  `infer`, `paramCount`, `savedForBackward`, docs). Register it in `src/nodes/registry.ts`.
- Every graph edit (nodes / edges / hyperparams) is a store action in `src/store/useCanvasStore.ts` that calls
  `remember(key?)` before `commit(...)` so it is undoable (see PLAN §2.4); pure UI state doesn't.
- Node params are either bound to a global hyperparam (`{bind: "d_model"}`) or overridden (`{value: n}`).
- Shapes carry both a size and a symbolic label (`{size: 512, label: "d_model"}`).
- Don't add libraries beyond the stack without a reason; prefer plain components.

## Commands
- `pnpm install` — install deps
- `pnpm dev` — dev server at http://localhost:5173 (also `.claude/launch.json` → "dev")
- `pnpm test` — run Vitest once (`pnpm test:watch` for watch mode); tests live next to code as `*.test.ts`
- `pnpm typecheck` — `tsc` (noEmit)
- `pnpm build` — typecheck + production build to `dist/`

Saved canvas lives in `localStorage["llm-canvas:v1"]` (document format version 3: hyperparams + group frames with `parentId` children);
clear it (or use Toolbar → Reset) to get the default graph. Older-format saves are ignored automatically.
