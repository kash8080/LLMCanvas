// Core engine types (pure TS, no React). See docs/PLAN.md §2.1.

/** One tensor dimension: concrete size plus an optional symbolic label (e.g. {size: 512, label: 'd_model'}). */
export interface Dim {
  size: number
  label?: string
}

/** 'int64' for token ids / targets; 'float' uses the global dtype hyperparam (fp32 / bf16 / fp16). */
export type TensorDType = 'int64' | 'float'

/** A tensor shape. A scalar has `dims: []`. */
export interface Shape {
  dims: Dim[]
  dtype: TensorDType
}

/** Global hyperparameters (PLAN.md §3). `num_layers` is derived from the graph, not stored here. */
export interface Hyperparams {
  vocab_size: number
  context_length: number
  d_model: number
  num_heads: number
  d_ff: number
  rope_theta: number
  batch_size: number
  dtype: FloatDType
}

export type FloatDType = 'fp32' | 'bf16' | 'fp16'

/** Numeric hyperparameters a node param can be bound to. */
export type HyperparamKey = Exclude<keyof Hyperparams, 'dtype'>

/** A bind target: a hyperparam, or a value derived from hyperparams (d_head = d_model / num_heads). */
export type BindKey = HyperparamKey | 'd_head'

/** A node param is either bound to a global hyperparam (🔗) or a local override. */
export type ParamValue = { bind: BindKey } | { value: number | boolean | string }

export type ParamKind = 'int' | 'float' | 'bool' | 'enum'

export interface ParamSchema {
  key: string
  label: string
  kind: ParamKind
  /** Default binding or value for a freshly created node. */
  default: ParamValue
  help: string
  /** For kind 'enum'. */
  options?: string[]
  /** Minimum allowed local value (int/float). Bound values are validated in the hyperparams panel. */
  min?: number
}

/** A resolved param: the concrete value plus the symbol it came from (if bound). */
export interface ResolvedParam {
  value: number | boolean | string
  /** Symbolic label when bound (e.g. 'd_model', 'B'); undefined for local overrides. */
  label?: string
  bound: boolean
}

export type ResolvedParams = Record<string, ResolvedParam>

export interface PortDef {
  id: string
  label: string
}

export type Category = 'io' | 'embedding' | 'norm' | 'linear' | 'attention' | 'ffn' | 'elementwise' | 'loss'

export interface InferContext {
  /** Input shapes in the order of `NodeDef.inputs` (all present when `infer` is called). */
  inputs: Shape[]
  p: ResolvedParams
  hp: Hyperparams
}

export interface InferResult {
  /** Output shapes in the order of `NodeDef.outputs` (ignored when errors is non-empty). */
  outputs: Shape[]
  errors: string[]
}

export interface ParamTensor {
  name: string
  dims: Dim[]
}

export interface ParamCountResult {
  total: number
  tensors: ParamTensor[]
}

/** A tensor kept alive for the backward pass (PLAN.md §5). Phase 6 dedupes by source tensor. */
export interface SavedTensor {
  name: string
  /** 'input' / 'output' refer to the node's port `port`; 'internal' is an extra tensor (e.g. attention probs). */
  which: 'input' | 'output' | 'internal'
  port?: string
  shape: Shape
}

export interface SavedContext {
  inputs: Shape[]
  outputs: Shape[]
  p: ResolvedParams
  hp: Hyperparams
}

/** Where a part lives in the CS336 code: `cs336_basics/<file> · <symbol>`. */
export interface Cs336Ref {
  file: string
  symbol: string
}

/** Learning content shown in the detail drawer (PLAN.md §2.1, R4). Plain text; unicode math is fine. */
export interface NodeDocs {
  /** 2–4 plain-English sentences: what it does and why it is in the model. */
  overview: string
  /** Extra sentence for a well-known instance, keyed by its default title (e.g. `q_proj`, `ln_final`). */
  roles?: Record<string, string>
  /** Formula lines, rendered in monospace. */
  formula?: string[]
  /** How the parameter count is computed (groups; parts derive it from their weight tensors). */
  paramFormula?: string
  /** 3–6 key insights / common pitfalls. */
  pointsToRemember: string[]
  /** Longer help per param key (falls back to `ParamSchema.help`). */
  paramHelp?: Record<string, string>
  cs336Ref?: Cs336Ref
}

/** Everything about one part type (one file per type in src/nodes/). */
export interface NodeDef {
  type: string
  label: string
  category: Category
  inputs: PortDef[]
  outputs: PortDef[]
  params: ParamSchema[]
  infer: (ctx: InferContext) => InferResult
  paramCount: (p: ResolvedParams, hp: Hyperparams) => ParamCountResult
  savedForBackward: (ctx: SavedContext) => SavedTensor[]
  docs: NodeDocs
}

// ---- Graph model consumed by infer.ts (decoupled from React Flow) ----

export interface GraphNode {
  id: string
  type: string
  params: Record<string, ParamValue>
  /** Container (group) id. Only `flattenGroups` (engine/groups.ts) uses it, to find a group's proxies. */
  parentId?: string
}

export interface GraphEdge {
  id: string
  source: string
  sourceHandle: string | null
  target: string
  targetHandle: string | null
}

export interface GraphModel {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export type NodeStatus = 'ok' | 'error' | 'unknown'

export interface NodeResult {
  status: NodeStatus
  errors: string[]
  /** Per input port (def order); null = not connected or upstream unknown. */
  inputShapes: (Shape | null)[]
  /** Per output port (def order); null when status != 'ok'. */
  outputShapes: (Shape | null)[]
  resolved: ResolvedParams
  paramCount: ParamCountResult
}

export interface InferenceResult {
  nodes: Record<string, NodeResult>
  /** Shape flowing along each edge (keyed by edge id); null when unknown. */
  edges: Record<string, Shape | null>
}
