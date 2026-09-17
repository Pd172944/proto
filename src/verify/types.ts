/**
 * Verification types.
 *
 * The candidate format is a JSON envelope rather than a raw unified diff. That
 * is a deliberate trade-off for a local-model-first harness:
 *
 *  - Small models produce malformed unified diffs constantly (wrong hunk
 *    headers, wrong line counts, fabricated context). A `find`/`replace` pair
 *    fails loudly and locally when the anchor is absent, which the verifier can
 *    detect without guessing.
 *  - `find`/`replace` is exactly checkable: "does this anchor appear exactly
 *    once in the file?" is a fact, whereas "is this diff correct?" requires
 *    applying it and hoping.
 *  - A full-file `content` variant is allowed for new files or whole-file
 *    rewrites, and is far easier for a weak model to produce correctly than a
 *    diff.
 *
 * The cost is that large refactors need many anchors. That is acceptable: those
 * tasks are routed to the cloud anyway.
 */

export type Severity = 'error' | 'warning' | 'info';

export interface CheckResult {
  id: string;
  ok: boolean;
  severity: Severity;
  detail: string;
  /** Extra lines worth showing the user (e.g. the offending pattern match). */
  evidence?: string;
}

export interface CandidateEdit {
  file: string;
  /** Exact text to locate. Mutually exclusive with `content`. */
  find?: string;
  /** Replacement text for `find`. */
  replace?: string;
  /** Full new file content. */
  content?: string;
}

export interface CandidateOutput {
  summary: string;
  /** Present for edit tasks. */
  edits: CandidateEdit[];
  /** Present for explain-only tasks. */
  answer?: string;
  explanation?: string;
  risk?: 'low' | 'medium' | 'high';
  /** Anything the model itself flagged as uncertain, when it said so. */
  uncertainty?: string;
  /** Raw text before parsing, kept so a failure can be explained. */
  raw?: string;
}

/** A file after a candidate is applied in memory (never on disk during verify). */
export interface AppliedFile {
  path: string;
  before: string | null;
  after: string;
  /** 0 for a new file. */
  changedLines: number;
}

export interface VerificationReport {
  /** True when nothing with severity `error` was found. */
  passed: boolean;
  /** 0..1 quality score. Reported to the user; nothing else consumes it. */
  score: number;
  checks: CheckResult[];
  /** Human-readable reasons the candidate failed. */
  blockers: string[];
  candidate: CandidateOutput | null;
  parseError?: string;
  applied: AppliedFile[];
  /** Output of the test command, when it ran. */
  testOutput?: string;
  durationMs: number;
}

export function isEditTask(candidate: CandidateOutput | null, features: { isExplainOnly: boolean }): boolean {
  if (features.isExplainOnly) return false;
  if (!candidate) return false;
  if (candidate.answer && (!candidate.edits || candidate.edits.length === 0)) return false;
  return (candidate.edits?.length ?? 0) > 0;
}
