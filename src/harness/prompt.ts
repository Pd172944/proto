/**
 * Prompt construction.
 *
 * Two properties matter more than prompt eloquence here:
 *
 *  1. **The output envelope must be trivially checkable.** We ask for a strict
 *     JSON object with `find`/`replace` anchors, because that is what the
 *     verifier can validate without guessing. Prose is not verifiable.
 *  2. **The system preamble must be byte-stable** so that providers with prompt
 *     caching (Anthropic, OpenAI) actually hit the cache on repeat calls. That
 *     is why nothing run-specific is allowed into the system message — no
 *     timestamps, no task text, no file names.
 *
 * `PROMPT_VERSION` is stored on every attempt, so a reported result can always
 * be traced back to the instructions that produced it.
 */

import type { TaskContext, TaskFeatures } from '../router/types.ts';
import type { Message } from '../providers/types.ts';
import { truncateMiddle } from '../util/text.ts';

export const PROMPT_VERSION = 'v1';

const EDIT_SYSTEM = `You are the coding tier of a two-model harness. You receive a task and the exact files it refers to.

Reply with ONE JSON object and nothing else. No markdown fences, no commentary.

Schema:
{
  "summary": "one sentence describing the change",
  "edits": [
    { "file": "relative/path.ext", "find": "exact existing text", "replace": "new text" }
  ],
  "risk": "low" | "medium" | "high",
  "uncertainty": "optional: what you are unsure about"
}

Rules for "find"/"replace" edits:
- "find" must be copied character-for-character from the provided file, including indentation and line breaks.
- "find" must occur EXACTLY ONCE in that file. Include surrounding lines to disambiguate if needed.
- Change only what the task requires. Never reformat unrelated code.
- Do not add TODO comments, debug prints, or commented-out code.
- If you need to create a new file, or replace a whole file, use "content" instead of "find"/"replace".

If the task cannot be done with the information given, return:
{ "summary": "cannot complete", "edits": [], "uncertainty": "what is missing" }`;

const EXPLAIN_SYSTEM = `You are the coding tier of a two-model harness. You receive a question about code and must answer it.

Reply with ONE JSON object and nothing else. No markdown fences, no commentary.

Schema:
{ "summary": "one sentence", "answer": "your full answer, markdown allowed inside the string" }

Be concrete and reference specific lines or symbols from the provided code. If the information given is insufficient, say exactly what is missing rather than guessing.`;

const REPAIR_SYSTEM = `You are the coding tier of a two-model harness. Your previous answer was automatically rejected by a verifier.

Reply with ONE corrected JSON object using the same schema as before. Fix every listed problem. Do not restate the previous answer, and do not relax the task requirements to make the check pass.`;

export function systemPromptFor(features: TaskFeatures): { content: string; version: string } {
  const base = features.isExplainOnly ? EXPLAIN_SYSTEM : EDIT_SYSTEM;
  return { content: base, version: PROMPT_VERSION };
}

export function repairSystemPrompt(): string {
  return REPAIR_SYSTEM;
}

export interface BuildUserPromptInput {
  ctx: TaskContext;
  features: TaskFeatures;
  /** Character budget for file contents; keeps a small model inside its window. */
  maxContextChars: number;
}

export function buildUserPrompt(input: BuildUserPromptInput): string {
  const { ctx, features } = input;
  const parts: string[] = [];

  parts.push(`TASK:\n${ctx.task.trim()}`);

  if (ctx.constraints?.length) {
    parts.push(`CONSTRAINTS (must hold):\n${ctx.constraints.map((c) => `- ${c}`).join('\n')}`);
  }

  if (ctx.diff) {
    parts.push(`CURRENT DIFF:\n\`\`\`diff\n${truncateMiddle(ctx.diff, Math.floor(input.maxContextChars * 0.4))}\n\`\`\``);
  }

  const files = ctx.files ?? [];
  if (files.length > 0) {
    const perFile = Math.max(1500, Math.floor(input.maxContextChars / files.length));
    const rendered = files.map(
      (f) => `FILE: ${f.path}\n\`\`\`\n${truncateMiddle(f.content, perFile)}\n\`\`\``,
    );
    parts.push(`FILES IN SCOPE (${files.length}):\n${rendered.join('\n\n')}`);
  } else if (!features.isExplainOnly) {
    parts.push(
      'NO FILES PROVIDED. If the change targets a specific file, use a relative path that matches the project layout.',
    );
  }

  return parts.join('\n\n');
}

export function buildRepairPrompt(input: {
  previousText: string;
  blockers: string[];
  failedChecks: Array<{ detail: string; evidence?: string }>;
  maxPreviousChars?: number;
}): string {
  const lines: string[] = [];
  lines.push('The verifier rejected your previous answer.');
  lines.push('');
  lines.push('PROBLEMS FOUND:');
  for (const b of input.blockers) lines.push(`- ${b}`);
  for (const c of input.failedChecks) {
    lines.push(`- ${c.detail}${c.evidence ? `\n  evidence: ${c.evidence.split('\n').join('\n  ')}` : ''}`);
  }
  lines.push('');
  lines.push('YOUR PREVIOUS ANSWER:');
  lines.push(truncateMiddle(input.previousText, input.maxPreviousChars ?? 4000));
  lines.push('');
  lines.push('Return a corrected JSON object now.');
  return lines.join('\n');
}

export function buildMessages(input: {
  system: string;
  user: string;
  /** Mark the system message as a prompt-cache boundary when supported. */
  cacheSystem?: boolean;
}): Message[] {
  const system: Message = { role: 'system', content: input.system };
  if (input.cacheSystem) system.cacheBreakpoint = true;
  return [system, { role: 'user', content: input.user }];
}
