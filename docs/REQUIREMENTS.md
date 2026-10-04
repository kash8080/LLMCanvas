# LLM Canvas — Requirements

A browser-only, Miro/Figma-style canvas for learning the Transformer / LLM architecture.
Users place parts of a transformer on a canvas, connect them, configure them, and see how
tensor **shapes**, **parameter counts** and **memory** flow through the model. Nothing is
actually executed — we only trace shapes and do arithmetic.

Side project for learning, not production. Prefer simple over clever.

Reference model: CS336 Assignment 1 (`/Users/rahul/Documents/code/rahul/cs336/assignment1-basics/cs336_basics`),
**not** the original "Attention Is All You Need" paper.

Status legend used in PLAN.md: each requirement ID below is referenced from the phase that implements it.

---

## R1 — Canvas
- R1.1 Infinite canvas with pan and zoom in/out (mouse wheel, trackpad, buttons, "fit view").
- R1.2 Place parts by dragging from a palette onto the canvas.
- R1.3 Move, select, multi-select, delete, duplicate parts.
- R1.4 Connecting parts must be as easy as Miro: drag from an output port to an input port (or onto the node body).
- R1.5 Minimap for orientation.

## R2 — Model parts (node types)
- R2.1 A fixed catalog of part types (Embedding, RMSNorm, Linear, Attention, RoPE, SwiGLU, Add, …).
- R2.2 Each type has a fixed, hard-coded set of parameters; users can only edit those.
- R2.3 Parameters include sizes/dimensions (e.g. `d_model`, `in_features`, `num_heads`) and custom ones (e.g. `eps`, `theta`, causal mask on/off).
- R2.4 Some parameters must agree with the incoming tensor (e.g. Linear `in_features` == last dim of input). On mismatch the node is highlighted as an error with a readable message.
- R2.5 Inputs and outputs (ports) are visually distinct (side + color + shape).
- R2.6 Clicking a port (or hovering an edge) shows the tensor shape flowing through it, both symbolic (`B × T × d_model`) and concrete (`32 × 256 × 512`).

## R3 — Annotation elements
- R3.1 Text box and sticky note elements.
- R3.2 Configurable: background color, (text color / font size), resizable.

## R4 — Detail drawer
- R4.1 Clicking a part opens a closable right-side drawer.
- R4.2 Drawer sections are separate and clean: **Overview** (basic description), **Parameters** (editable), **Shapes** (in/out), **Size** (params + memory for this part), **Points to remember**, (and formula where useful).

## R5 — Grouping & semantic zoom
- R5.1 Parts can be grouped (e.g. a Transformer Block contains RMSNorm, Attention, Add, …).
- R5.2 Level of detail depends on zoom: zoomed out shows a single Transformer Block card; zooming in reveals its sub-parts (and further in, the internals of Attention / SwiGLU).
- R5.3 Manual expand/collapse override per group.

## R6 — Hyperparameters
- R6.1 Global model hyperparameters editable in one place: `vocab_size`, `context_length`, `d_model`, `num_layers`, `num_heads`, `d_ff`, `rope_theta`, `batch_size`, dtype.
- R6.2 Part parameters default to global hyperparameters, and can be overridden per part.

## R7 — Parameter accounting
- R7.1 Each part shows its parameter count.
- R7.2 Total parameter count of the model with the calculation shown (formula).
- R7.3 Breakdown of where parameters live (embedding vs attention vs FFN vs norms vs LM head), visual.

## R8 — Memory estimation
- R8.1 Estimated total memory for the selected hyperparameters and **mode**: Forward (inference), Forward+Backward, Training (incl. optimizer).
- R8.2 Breakdown: weights / gradients / optimizer state / activations, and *which parts* hold the activations (so the user can see where to optimise — e.g. attention probabilities, logits).
- R8.3 Memory knobs: dtype, batch size, context length, activation checkpointing (as in CS336 `checkpoint_blocks`).

## R9 — Defaults & persistence
- R9.1 On first load the canvas contains the CS336 TransformerLM (not an empty slate), using CS336 `train.py` defaults:
  `vocab_size=10000, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000, batch_size=32`.
- R9.2 Work is saved automatically in the browser; "Reset to CS336 default" is available.

## Non-goals
- No real model execution, no backend, no auth, no multi-user collaboration.
