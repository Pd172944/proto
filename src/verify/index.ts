/**
 * The verifier: turn a model response into a pass/fail verdict plus a score.
 *
 * This module is the load-bearing part of "route easy work to a cheap local
 * model". A router without a verifier is just a way to be wrong more cheaply.
 * The verifier is what lets the harness *know* a local attempt failed and
 * escalate, and it is also the primary reward signal for the RL pipeline — so
 * its scoring must be stable, explainable and independent of which model
 * produced the candidate.
 *
 * Order of checks is intentional: cheap structural facts first (did it parse,
 * does the anchor exist, does it exceed the size cap), then language parsing,
 * then heuristics, then optionally the project's own tests. The first hard
 * failure is usually enough to escalate, and earlier exits save time.
 */

import { resolve, sep } from 'node:path';

import {
  addedLines,
  applyEdits,
  diffLineCount,
  parseCandidate,
  patternChecks,
  validateEditPath,
} from './patch.ts';
import { asCheck, syntaxCheck } from './syntax.ts';
import type { AppliedFile, CheckResult, VerificationReport } from './types.ts';
import type { ProtoConfig } from '../config/schema.ts';
import type { TaskContext, TaskFeatures } from '../router/types.ts';
import { languageOfPath } from '../router/features.ts';
import { exec } from '../util/proc.ts';
import { readTextOrNull } from '../util/fsx.ts';

export interface VerifyInput {
  /** Raw model output. */
  text: string;
  ctx: TaskContext;
  cfg: ProtoConfig;
  dataDir: string;
  features: TaskFeatures;
  /** Skip the configured test command even if enabled (used by `--no-tests`). */
  skipTests?: boolean;
}

export async function verifyCandidate(input: VerifyInput): Promise<VerificationReport> {
  const started = Date.now();
  const checks: CheckResult[] = [];
  const { ctx, cfg, features } = input;

  if (!cfg.verify.enabled) {
    // Verification disabled: we cannot claim the candidate is correct, so we
    // report a neutral, explicitly-unverified pass. `unverified` raises the
    // routing quality floor, so this path is rare by design.
    const parsed = parseCandidate(input.text, { expectedPaths: ctx.files?.map((f) => f.path) });
    return {
      passed: true,
      score: 0.5,
      checks: [{ id: 'verify-disabled', ok: true, severity: 'info', detail: 'verification is disabled in config' }],
      blockers: [],
      candidate: parsed.candidate,
      applied: [],
      durationMs: Date.now() - started,
    };
  }

  const parsed = parseCandidate(input.text, { expectedPaths: ctx.files?.map((f) => f.path) });
  if (!parsed.candidate) {
    return {
      passed: false,
      score: 0,
      checks: [{ id: 'parse', ok: false, severity: 'error', detail: parsed.parseError ?? 'unparseable response' }],
      blockers: [parsed.parseError ?? 'unparseable response'],
      candidate: null,
      ...(parsed.parseError ? { parseError: parsed.parseError } : {}),
      applied: [],
      durationMs: Date.now() - started,
    };
  }
  const candidate = parsed.candidate;

  checks.push({
    id: 'parse',
    ok: true,
    severity: 'info',
    detail:
      parsed.shape === 'json'
        ? `parsed structured output with ${candidate.edits.length} edit(s)`
        : `model did not use the JSON envelope; salvaged as "${parsed.shape}"`,
  });
  if (parsed.shape !== 'json') {
    checks.push({
      id: 'envelope-drift',
      ok: false,
      severity: 'warning',
      detail: `expected a JSON envelope, got ${parsed.shape}; this is a reliability signal for the local model`,
    });
  }

  // --- explain-only tasks ---------------------------------------------------
  if (features.isExplainOnly && candidate.edits.length === 0) {
    const answer = (candidate.answer ?? candidate.explanation ?? '').trim();
    if (answer.length < 40) {
      checks.push({
        id: 'answer-too-short',
        ok: false,
        severity: 'error',
        detail: `explanation is only ${answer.length} characters; too short to be a real answer`,
      });
    } else {
      checks.push({ id: 'answer', ok: true, severity: 'info', detail: `explanation answer (${answer.length} chars)` });
    }
    if (/\b(I can(?:'|no)t|I cannot|I'm not able|I am not able|as an AI|I don't have access|unable to assist)\b/i.test(answer)) {
      checks.push({ id: 'refusal', ok: false, severity: 'error', detail: 'the model refused the task' });
    }
    return finalize(checks, candidate, [], undefined, started);
  }

  // --- edit tasks -----------------------------------------------------------
  if (candidate.edits.length === 0) {
    checks.push({
      id: 'no-edits',
      ok: false,
      severity: 'error',
      detail: 'the task requires code changes but the candidate contains none',
    });
    return finalize(checks, candidate, [], undefined, started);
  }

  const known = collectKnownFiles(ctx, candidate.edits.map((e) => e.file));
  const { applied, errors } = applyEdits(candidate, known);
  checks.push(...errors);

  // Size cap: a "small fix" that rewrites 900 lines is not the requested change.
  const totalChanged = applied.reduce((a, f) => a + f.changedLines, 0);
  if (totalChanged > cfg.verify.maxPatchLines) {
    checks.push({
      id: 'size-cap',
      ok: false,
      severity: 'error',
      detail: `patch changes ~${totalChanged} lines, above the ${cfg.verify.maxPatchLines}-line cap for an automated edit`,
    });
  } else if (applied.length > 0) {
    checks.push({ id: 'size-cap', ok: true, severity: 'info', detail: `patch changes ~${totalChanged} line(s)` });
  }

  // Deletion sanity: refuse candidates that only remove code.
  for (const file of applied) {
    if (file.before === null) continue;
    const beforeLines = file.before.split('\n').length;
    const afterLines = file.after.split('\n').length;
    if (beforeLines > 20 && afterLines > 0 && afterLines < beforeLines * 0.5) {
      checks.push({
        id: `mass-deletion:${file.path}`,
        ok: false,
        severity: 'error',
        detail: `${file.path}: removes ${beforeLines - afterLines} of ${beforeLines} lines; refusing a change this destructive`,
      });
    }
  }

  // --- syntax ---------------------------------------------------------------
  for (const file of applied) {
    const lang = languageOfPath(file.path);
    if (!lang) continue;
    const result = await syntaxCheck(file.after, lang, `${input.dataDir}/tmp`, 10_000);
    checks.push(asCheck(result, `syntax:${file.path}`, file.path));
  }

  // --- patterns -------------------------------------------------------------
  checks.push(
    ...patternChecks(applied, {
      rejectNewTodos: cfg.verify.rejectNewTodos,
      forbiddenPatterns: cfg.verify.forbiddenPatterns,
    }),
  );

  // --- tests ----------------------------------------------------------------
  let testOutput: string | undefined;
  if (!input.skipTests && cfg.verify.runTests && cfg.verify.testCommand && applied.length > 0) {
    const result = await runTests(cfg.verify.testCommand, ctx.workspace, cfg.verify.testTimeoutMs);
    testOutput = result.output;
    checks.push({
      id: 'tests',
      ok: result.ok,
      severity: result.ok ? 'info' : 'error',
      detail: result.ok
        ? `test command passed in ${result.durationMs}ms`
        : `test command failed (exit ${result.code ?? 'signal'})`,
      ...(result.ok ? {} : { evidence: result.output.split('\n').slice(-12).join('\n').slice(0, 800) }),
    });
  } else if (cfg.verify.runTests && !cfg.verify.testCommand) {
    checks.push({
      id: 'tests',
      ok: true,
      severity: 'info',
      detail: 'verify.runTests is on but no verify.testCommand is configured; skipped',
    });
  }

  return finalize(checks, candidate, applied, testOutput, started);
}

function finalize(
  checks: CheckResult[],
  candidate: VerificationReport['candidate'],
  applied: AppliedFile[],
  testOutput: string | undefined,
  started: number,
): VerificationReport {
  const errors = checks.filter((c) => !c.ok && c.severity === 'error');
  const warnings = checks.filter((c) => !c.ok && c.severity === 'warning');
  let score = 1 - 0.4 * errors.length - 0.08 * warnings.length;
  score = Math.max(0, Math.min(1, score));
  const passed = errors.length === 0;
  // A failing candidate must never look "almost good" to the reward function.
  if (!passed) score = Math.min(score, 0.3);
  return {
    passed,
    score: Math.round(score * 1000) / 1000,
    checks,
    blockers: errors.map((e) => e.detail),
    candidate,
    applied,
    ...(testOutput ? { testOutput } : {}),
    durationMs: Date.now() - started,
  };
}

/* ------------------------------------------------------------------ */
/* File resolution                                                     */
/* ------------------------------------------------------------------ */

/**
 * Build the in-memory view of every file an edit refers to.
 * Prefers files supplied in the task context, then falls back to reading from
 * the workspace — bounded to inside the workspace root.
 */
export function collectKnownFiles(ctx: TaskContext, paths: string[]): Map<string, string> {
  const map = new Map<string, string>();
  for (const f of ctx.files ?? []) map.set(normalizeRel(f.path), f.content);
  for (const p of paths) {
    const rel = normalizeRel(p);
    if (map.has(rel)) continue;
    const content = readWorkspaceFile(ctx.workspace, rel);
    if (content !== null) map.set(rel, content);
  }
  return map;
}

function normalizeRel(p: string): string {
  return p.replace(/\\/g, '/').replace(/^\.\//, '');
}

export function readWorkspaceFile(workspace: string | undefined, rel: string): string | null {
  if (!workspace) return null;
  if (!validateEditPath(rel).ok) return null;
  const root = resolve(workspace);
  const full = resolve(root, rel);
  if (full !== root && !full.startsWith(root + sep)) return null;
  return readTextOrNull(full);
}

/* ------------------------------------------------------------------ */
/* Test runner                                                         */
/* ------------------------------------------------------------------ */

export interface TestRunResult {
  ok: boolean;
  code: number | null;
  output: string;
  durationMs: number;
}

/**
 * Run the project's tests with the lowest reasonable priority.
 *
 * Note the deliberate choice to run the command through the shell: test commands
 * are user-authored strings like `npm test -- --run`. The command comes from the
 * user's own config file, not from a model, so this is not an injection surface
 * — model output never becomes part of this string.
 */
export async function runTests(command: string, cwd: string | undefined, timeoutMs: number): Promise<TestRunResult> {
  const res = await exec('/bin/sh', ['-c', command], {
    ...(cwd ? { cwd } : {}),
    timeoutMs,
    lowPriority: true,
    maxOutputBytes: 256 * 1024,
  });
  const output = `${res.stdout}\n${res.stderr}`.trim();
  return {
    ok: res.code === 0 && !res.timedOut,
    code: res.code,
    output: res.timedOut ? `[timed out after ${timeoutMs}ms]\n${output}` : output,
    durationMs: res.durationMs,
  };
}

export * from './types.ts';
export { parseCandidate, applyEdits, validateEditPath, diffLineCount, addedLines, patternChecks };
export { syntaxCheck, balanceCheck } from './syntax.ts';
