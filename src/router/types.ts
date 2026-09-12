/**
 * Router types. These are the shared vocabulary between feature extraction,
 * scoring, policy, the episode log and the training datasets, so they change
 * only with an explicit version bump (see FEATURE_VECTOR_VERSION).
 */

export type Tier = 'local-tiny' | 'local' | 'cloud-cheap' | 'cloud-strong';

export const TIERS: Tier[] = ['local-tiny', 'local', 'cloud-cheap', 'cloud-strong'];

export function tierRank(t: Tier): number {
  return TIERS.indexOf(t);
}

export function isLocalTier(t: Tier): boolean {
  return t === 'local' || t === 'local-tiny';
}

/**
 * Task taxonomy. Chosen so that each class has a defensible prior about whether
 * a small local model can do it, and so that classes are recognisable from the
 * task text plus any supplied code without a model call.
 */
export type TaskClass =
  | 'format' // pure formatting/lint/whitespace
  | 'rename' // mechanical identifier rename
  | 'prompt-edit' // editing a prompt string / template
  | 'docs' // docstrings, comments, README prose
  | 'explain' // explain code, no mutation
  | 'local-edit' // bounded edit inside one function/block
  | 'bugfix-local' // fix a localized bug with a known symptom
  | 'add-validation' // guard clauses, bounds checks, input validation
  | 'write-tests' // add/extend tests for existing behaviour
  | 'config' // dependency/build/CI config tweak
  | 'refactor-multi' // refactor spanning files or modules
  | 'feature-new' // implement new behaviour
  | 'perf' // performance work requiring measurement
  | 'debug-unknown' // diagnose with no repro or unknown cause
  | 'concurrency' // async/threading/race conditions
  | 'security' // auth, crypto, injection, secrets
  | 'migration' // framework/language/schema migration
  | 'architecture' // design, API shape, module boundaries
  | 'algorithm' // non-trivial algorithmic work
  | 'unknown';

export interface TaskFeatures {
  /** ---- size ---- */
  taskChars: number;
  estInputTokens: number;
  /** Output tokens are estimated from the requested change size, not guessed blindly. */
  estOutputTokens: number;
  fileCount: number;
  changedLines: number;
  /** ---- structure of the code in scope ---- */
  loopCount: number;
  funcCount: number;
  maxNesting: number;
  branchCount: number;
  /** ---- risk / difficulty flags ---- */
  hasAsync: boolean;
  hasConcurrency: boolean;
  hasTypes: boolean;
  hasErrorHandling: boolean;
  hasTestsInScope: boolean;
  hasStackTrace: boolean;
  hasReproSteps: boolean;
  hasExplicitAcceptanceCriteria: boolean;
  hasExternalApiMention: boolean;
  hasPerfLanguage: boolean;
  hasSecurityLanguage: boolean;
  hasMigrationLanguage: boolean;
  /** ---- phrasing signals ---- */
  isQuestion: boolean;
  isExplainOnly: boolean;
  constraintCount: number;
  /** 0..1 — how much the wording leaves open. Higher = more guessing required. */
  ambiguity: number;
  /** 0..1 — how tightly the change is scoped to specific symbols/lines. */
  locality: number;
  mentionsSpecificSymbol: boolean;
  /** ---- classification ---- */
  taskClass: TaskClass;
  /** 0..1 prior difficulty implied by the class alone. */
  classDifficulty: number;
  languages: string[];
  /** Human-readable observations, surfaced in `proto route --explain`. */
  signals: string[];
}

/**
 * Bump when the numeric vector changes meaning. Stored in the weights file and
 * in every episode so that datasets from different versions are never mixed.
 */
export const FEATURE_VECTOR_VERSION = 1;

/**
 * The exact ordered feature names for the learned scorer. Adding a feature
 * requires bumping FEATURE_VECTOR_VERSION, because old weights would otherwise
 * be silently applied to a shifted vector.
 */
export const FEATURE_NAMES = [
  'bias',
  'log_input_tokens',
  'log_output_tokens',
  'log_file_count',
  'log_changed_lines',
  'log_loop_count',
  'log_func_count',
  'log_max_nesting',
  'log_branch_count',
  'has_async',
  'has_concurrency',
  'has_types',
  'has_error_handling',
  'has_tests_in_scope',
  'has_stack_trace',
  'has_repro_steps',
  'has_acceptance_criteria',
  'has_external_api',
  'has_perf_language',
  'has_security_language',
  'has_migration_language',
  'is_question',
  'is_explain_only',
  'log_constraint_count',
  'ambiguity',
  'locality',
  'mentions_symbol',
  'class_difficulty',
  'class_is_local_edit',
  'class_is_bugfix_local',
  'class_is_validation',
  'class_is_rename_or_format',
  'class_is_tests',
  'class_is_docs_or_explain',
  'class_is_refactor_multi',
  'class_is_feature_new',
  'class_is_perf',
  'class_is_debug_unknown',
  'class_is_concurrency',
  'class_is_security',
  'class_is_migration',
  'class_is_architecture',
  'class_is_algorithm',
] as const;

export type FeatureName = (typeof FEATURE_NAMES)[number];

/** Input supplied alongside the task text. */
export interface TaskContext {
  /** Free-form task description from the user. */
  task: string;
  /** Files in scope: path -> content. */
  files?: Array<{ path: string; content: string }>;
  /** An existing diff/patch the user wants reviewed or fixed. */
  diff?: string;
  /** Working directory the task refers to. */
  workspace?: string;
  /** SIGINT-style constraints the user typed explicitly. */
  constraints?: string[];
}

export interface ExpectedCost {
  localCostUsd: number;
  cloudCostUsd: number;
  localLatencyMs: number;
  cloudLatencyMs: number;
}

/**
 * The environment the decision was made in.
 *
 * Stored on every decision (and therefore every episode) because it is what
 * makes offline counterfactual replay sound: re-running the policy under a
 * *different* config is only meaningful if we know which tiers were even
 * eligible at the time. Without this, `proto replay` would happily "discover"
 * that it should have used a cloud model that had no API key.
 */
export interface RouterEnvironmentSnapshot {
  localEnabled: boolean;
  localAvailable: boolean;
  localAvailabilityNote?: string;
  localModelLoaded: boolean;
  localContextWindow: number;
  localTokensPerSec: number;
  tinyModelAvailable: boolean;
  cloudAvailable: boolean;
  cloudUnavailableReason?: string;
  verifierAvailable: boolean;
  cloudBudgetRemainingUsd: number;
}

export interface RouteDecision {
  tier: Tier;
  /** One-line explanation suitable for a status line. */
  reason: string;
  /** Full list of contributing reasons, in priority order. */
  reasons: string[];
  pLocalSuccess: number;
  difficulty: number;
  taskClass: TaskClass;
  expected: ExpectedCost;
  /** True when this decision was made by the exploration policy, not by utility. */
  exploration: boolean;
  /** Conditions that made a tier ineligible. */
  vetoes: string[];
  /** True when no verification is available, so the quality floor was raised. */
  unverified: boolean;
  features: TaskFeatures;
  vector: number[];
  vectorVersion: number;
  /** Which scorer produced pLocalSuccess. */
  scorer: 'heuristic' | 'learned' | 'hybrid';
  /**
   * Set when trained weights exist but could not be used (most commonly after a
   * FEATURE_VECTOR_VERSION bump). Silently falling back to the heuristic would
   * look like a routing-quality regression with no visible cause, so the failure
   * is carried on the decision and printed by `proto route --explain`.
   */
  scorerError?: string;
  /** True when the chosen tier is a fallback forced by an unavailable tier. */
  forced: boolean;
  /** Snapshot of tier eligibility at decision time (needed for replay). */
  env: RouterEnvironmentSnapshot;
}
