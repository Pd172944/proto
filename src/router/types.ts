/**
 * Router types: the shared vocabulary between feature extraction, scoring and
 * the policy. Routing is a pure function of the task and the environment, which
 * is what lets `proto route` explain any decision without calling a model.
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
 * Carried on every decision because the reason a tier was rejected is usually
 * about the environment, not the task: `proto route --explain` can only say
 * "local was unavailable" if the decision remembers that it was.
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
  /** Conditions that made a tier ineligible. */
  vetoes: string[];
  /** True when no verification is available, so the quality floor was raised. */
  unverified: boolean;
  features: TaskFeatures;
  /** True when the chosen tier is a fallback forced by an unavailable tier. */
  forced: boolean;
  /** Snapshot of tier eligibility at decision time, for `--explain`. */
  env: RouterEnvironmentSnapshot;
}
