/**
 * The plan/execute split: a strong model decides, a fast model types.
 *
 * The idea is that the two halves of a coding turn have very different requirements.
 * Deciding *what* to change needs judgement — reading the task, finding the right
 * function, knowing that renaming a symbol means every call site. Producing the exact
 * characters of the edit does not: it is mechanical, it is bounded by a file you can
 * paste into the prompt, and it is where most of the output tokens go.
 *
 * So the planner runs once, on the strong model, over the whole task. The executor
 * runs once per edit, on the local model, over one file and one instruction. The
 * executor's prompts are small and its job is narrow, which is exactly the regime a
 * small local model is good at — and because the steps are independent they run
 * concurrently, which is the only way a slow local model beats a fast remote one.
 *
 * **This is deliberately not a general agent.** There is no tool loop here, no
 * exploration mid-plan, and no adaptivity: the planner sees the files it is given and
 * commits to a plan. That is a real limitation, and it is the trade for a cheap
 * pipeline that is easy to reason about and to verify. `proto code` remains the
 * general path for work that needs to look around.
 *
 * Every stage is verified. The assembled result goes through exactly the same
 * verifier as a single-model answer, so a split run cannot be *less* checked than a
 * normal one — and a failed verification is fed back to the planner for one repair
 * round rather than being reported as a mysterious failure.
 */

import type { ChatRequest, Provider } from '../providers/types.ts';
import type { ProtoConfig } from '../config/schema.ts';
import type { TaskContext } from '../router/types.ts';

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

export interface PlanStep {
  /** File the step edits, or creates. */
  file: string;
  /** What to change, in plain language. No code — see the planner prompt. */
  intent: string;
}

export interface Plan {
  summary: string;
  steps: PlanStep[];
  /** Set when the task is a question rather than an edit. */
  answer?: string;
}

export interface SplitEdit {
  find?: string;
  replace?: string;
  content?: string;
}

export interface ExecutedStep {
  step: PlanStep;
  /** Everything the executor produced for this step. Empty means it failed. */
  edits: SplitEdit[];
  /** Populated when the step failed, with the reason. */
  error?: string;
  /** Wall-clock for this step, which is where the parallelism shows up. */
  durationMs: number;
  /** What the executor spent on this step. The local model's share of the work. */
  inputTokens: number;
  outputTokens: number;
}

export type SplitEvent =
  | { type: 'planned'; plan: Plan; plannerMs: number }
  | { type: 'execute-start'; file: string; intent: string }
  | { type: 'execute-end'; file: string; ok: boolean; durationMs: number; edits?: number; error?: string }
  | { type: 'repair'; round: number; blockers: string[] };

export interface SplitOptions {
  cfg: ProtoConfig;
  dataDir: string;
  ctx: TaskContext;
  /** The strong model: decides what to do. */
  planner: Provider;
  /** The fast model: produces the edits. */
  executor: Provider;
  /**
   * How many executor calls may be in flight at once. Defaults to 1.
   *
   * **Serial is the right default, which is counter-intuitive and was measured.**
   * Local decode is memory-bandwidth bound, so concurrent requests do not add
   * throughput — they split the same bandwidth and make every request slower. On a
   * 9B over three files:
   *
   *     concurrency 1   4.9s / 9.2s / 14.1s   -> 14.1s wall
   *     concurrency 4  21.6s / 26.0s / 30.7s  -> 30.7s wall
   *
   * Over twice as slow, with the same work. Raise this only for an executor that
   * genuinely batches on its side and has bandwidth to spare — vLLM with continuous
   * batching is the real case, where several requests are scheduled into one forward
   * pass and concurrency is close to free.
   */
  concurrency?: number;
  /** Repair rounds after a failed verification. */
  maxRepair?: number;
  onEvent?: (event: SplitEvent) => void;
}

export interface SplitResult {
  plan: Plan;
  steps: ExecutedStep[];
  /** The assembled candidate text, in the standard envelope the parser expects. */
  candidateText: string;
  plannerMs: number;
  executorMs: number;
  /** Executor tokens in total, which is the bill the local model picked up. */
  executorInputTokens: number;
  executorOutputTokens: number;
  warnings: string[];
}

/* ------------------------------------------------------------------ */
/* Prompts                                                             */
/* ------------------------------------------------------------------ */

/**
 * The planner is asked for *intentions*, not code.
 *
 * Deliberately: if it emitted the code too, the executor would have nothing to do and
 * the split would be pointless. Keeping the planner at the level of "rename
 * calc_total to compute_total in this module" is also what keeps its output short,
 * which is the expensive part of a cloud call.
 */
const PLANNER_SYSTEM = `You plan edits to a codebase. You do not write the edits themselves.

Reply with a single JSON object and nothing else:

{
  "summary": "one line describing the change",
  "steps": [ { "file": "relative/path.ext", "intent": "what to change in this file, in one or two sentences" } ],
  "answer": "only when the task is a question rather than a change"
}

Rules:
- "intent" must describe the change precisely enough that someone with only that file open could carry it out. Name the function, the variable, the condition.
- "intent" must NOT contain the replacement code. That is someone else's job.
- One step per file. If a file needs two unrelated changes, that is still one step; describe both.
- Only include files that actually need to change. Do not list files you merely read.
- If the task is a question, set "answer" and leave "steps" empty.`;

/**
 * The executor sees one file and one instruction, and returns every edit that
 * instruction implies.
 *
 * `edits` is a list because an instruction routinely covers more than one change —
 * "fix the divisor in mean() and the return in clamp()" is one coherent intent and two
 * separate edits. An earlier version asked for a single `find`/`replace`, and the
 * model dutifully returned the first change and silently dropped the second. The
 * verifier could not catch it: the edit it did receive was perfectly well formed.
 */
const EXECUTOR_SYSTEM = `You apply a described change to one file and reply with a single JSON object.

{
  "file": "the path you were given",
  "edits": [
    { "find": "exact existing text, copied character-for-character from the file", "replace": "the text to put in its place" }
  ]
}

Or, when creating a file or replacing it wholesale:
{ "file": "path", "content": "the entire new file contents" }

Rules:
- Emit ONE entry in "edits" per distinct change the instruction asks for. If the
  instruction mentions two changes, there must be two entries. Do not stop after the first.
- "find" must be copied exactly from the file, including indentation and line breaks.
- Each "find" must occur EXACTLY ONCE. If it does not, include more surrounding lines until it does.
- Make only the changes you were asked for. Do not reformat, reorder or "improve" anything else.
- Only the final return value matters; extra commentary does not.
- Reply with the JSON object and nothing else.`;

function fileBlock(files: Array<{ path: string; content: string }>, budgetChars: number): string {
  const out: string[] = [];
  let used = 0;
  for (const f of files) {
    const header = `\n--- ${f.path} ---\n`;
    const room = budgetChars - used - header.length;
    if (room <= 200) break;
    const body = f.content.length > room ? `${f.content.slice(0, room)}\n…[truncated]` : f.content;
    out.push(header + body);
    used += header.length + body.length;
  }
  return out.join('');
}

/* ------------------------------------------------------------------ */
/* Parsing                                                             */
/* ------------------------------------------------------------------ */

/** Pull the first JSON object out of a reply that may be wrapped in prose or a fence. */
export function extractJson(text: string): unknown {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
  const body = fenced?.[1]?.trim() ?? trimmed;
  const start = body.indexOf('{');
  const end = body.lastIndexOf('}');
  if (start === -1 || end === -1 || end < start) return null;
  try {
    return JSON.parse(body.slice(start, end + 1));
  } catch {
    return null;
  }
}

export function parsePlan(text: string): Plan | null {
  const raw = extractJson(text) as Record<string, unknown> | null;
  if (raw === null || typeof raw !== 'object') return null;
  const summary = typeof raw['summary'] === 'string' ? raw['summary'] : '';
  const stepsRaw = Array.isArray(raw['steps']) ? (raw['steps'] as unknown[]) : [];
  const steps: PlanStep[] = [];
  for (const item of stepsRaw) {
    const s = (item ?? {}) as Record<string, unknown>;
    const file = typeof s['file'] === 'string' ? s['file'].trim() : '';
    const intent = typeof s['intent'] === 'string' ? s['intent'].trim() : '';
    if (file !== '' && intent !== '') steps.push({ file, intent });
  }
  const answer = typeof raw['answer'] === 'string' && raw['answer'].trim() !== '' ? raw['answer'] : undefined;
  if (steps.length === 0 && answer === undefined) return null;
  return { summary, steps, ...(answer === undefined ? {} : { answer }) };
}

/**
 * Read the executor's reply into a list of edits.
 *
 * `edits` is the documented shape. A bare `find`/`replace` at the top level is still
 * accepted, because it costs three lines and models do drift to the simpler shape —
 * silently discarding a perfectly good single edit would be a worse trade than the
 * small amount of tolerance here.
 */
export function parseEdits(text: string): SplitEdit[] {
  const raw = extractJson(text) as Record<string, unknown> | null;
  if (raw === null) return [];

  if (typeof raw['content'] === 'string' && raw['content'] !== '') {
    return [{ content: raw['content'] }];
  }

  const out: SplitEdit[] = [];
  const list = Array.isArray(raw['edits']) ? (raw['edits'] as unknown[]) : [];
  for (const item of list) {
    const e = (item ?? {}) as Record<string, unknown>;
    if (typeof e['content'] === 'string' && e['content'] !== '') {
      out.push({ content: e['content'] });
      continue;
    }
    if (typeof e['find'] === 'string' && typeof e['replace'] === 'string') {
      out.push({ find: e['find'], replace: e['replace'] });
    }
  }
  if (out.length === 0 && typeof raw['find'] === 'string' && typeof raw['replace'] === 'string') {
    out.push({ find: raw['find'], replace: raw['replace'] });
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Pipeline                                                            */
/* ------------------------------------------------------------------ */

/** Run `tasks` with at most `limit` in flight, preserving input order in the result. */
async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const worker = async (): Promise<void> => {
    for (;;) {
      const i = next++;
      if (i >= items.length) return;
      out[i] = await fn(items[i] as T, i);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, worker));
  return out;
}

export interface PlanRequest {
  cfg: ProtoConfig;
  ctx: TaskContext;
  planner: Provider;
  /** Extra instruction appended on a repair round. */
  repair?: { blockers: string[] };
}

/** Ask the planner for a plan. One call, no tools. */
export async function planTask(input: PlanRequest): Promise<{ plan: Plan | null; text: string; ms: number }> {
  const { cfg, ctx, planner } = input;
  const budget = Math.max(4000, Math.floor(cfg.local.contextWindow * 2.5));
  const files = ctx.files ?? [];

  const context =
    `Task:\n${ctx.task}\n` +
    (ctx.constraints?.length ? `\nConstraints:\n${ctx.constraints.map((c) => `- ${c}`).join('\n')}\n` : '') +
    (ctx.diff ? `\nExisting diff:\n${ctx.diff.slice(0, 4000)}\n` : '') +
    `\nFiles in scope:${fileBlock(files, budget)}`;

  const repairNote =
    input.repair === undefined
      ? ''
      : `\n\nA previous plan was tried and verification rejected it:\n` +
        input.repair.blockers.map((b) => `- ${b}`).join('\n') +
        `\nProduce a corrected plan that fixes exactly these problems.\n`;

  const req: ChatRequest = {
    messages: [
      { role: 'system', content: PLANNER_SYSTEM },
      { role: 'user', content: context + repairNote },
    ],
    maxTokens: cfg.cloud.maxOutputTokens,
    temperature: 0.1,
    thinking: cfg.cloud.thinking,
  };

  const started = Date.now();
  const response = await planner.chat(req);
  return { plan: parsePlan(response.text), text: response.text, ms: Date.now() - started };
}

/** Ask the executor to carry out one step. One file, one instruction, one edit. */
export async function executeStep(input: {
  cfg: ProtoConfig;
  executor: Provider;
  step: PlanStep;
  files: Array<{ path: string; content: string }>;
  /**
   * Why a previous attempt at this step was rejected.
   *
   * Retrying on the executor rather than the planner is the right loop: an ambiguous
   * anchor is a local mistake with a local fix, and re-planning the whole task to
   * correct one `find` string wastes the expensive model on the cheap model's error.
   */
  feedback?: string[];
}): Promise<{ edits: SplitEdit[]; error?: string; text: string; inputTokens: number; outputTokens: number }> {
  const { cfg, executor, step, files } = input;
  const target = files.find((f) => f.path === step.file);

  const budget = Math.max(2000, Math.floor(cfg.local.contextWindow * 1.8));
  const body =
    target === undefined
      ? `The file ${step.file} does not exist yet. Create it.`
      : `--- ${step.file} ---\n${target.content.slice(0, budget)}`;

  const attemptNote =
    input.feedback === undefined || input.feedback.length === 0
      ? ''
      : `\n\nA previous attempt was REJECTED for these reasons:\n` +
        input.feedback.map((f) => `- ${f}`).join('\n') +
        `\nFix exactly these problems. For an ambiguous anchor, include more surrounding lines so the text occurs once.\n`;

  const req: ChatRequest = {
    messages: [
      { role: 'system', content: EXECUTOR_SYSTEM },
      { role: 'user', content: `File: ${step.file}\n\nChange to make: ${step.intent}\n${attemptNote}\n${body}` },
    ],
    maxTokens: Math.min(cfg.local.maxOutputTokens, 2048),
    temperature: 0,
    thinking: false,
  };

  const response = await executor.chat(req);
  const edits = parseEdits(response.text);
  return {
    edits,
    ...(edits.length === 0 ? { error: 'executor returned no usable find/replace or content' } : {}),
    text: response.text,
    inputTokens: response.usage.inputTokens,
    outputTokens: response.usage.outputTokens,
  };
}

/**
 * Turn executed steps into the same envelope a single model would have produced, so
 * that everything downstream — parsing, verification, the diff renderer — is shared
 * rather than duplicated for this path.
 */
export function assembleCandidate(plan: Plan, steps: ExecutedStep[]): string {
  const edits: Array<Record<string, unknown>> = [];
  for (const s of steps) {
    for (const e of s.edits) {
      if (e.content !== undefined) edits.push({ file: s.step.file, content: e.content });
      else edits.push({ file: s.step.file, find: e.find, replace: e.replace });
    }
  }

  const payload: Record<string, unknown> = {
    summary: plan.summary,
    edits,
    ...(plan.answer === undefined ? {} : { answer: plan.answer }),
  };
  return JSON.stringify(payload, null, 2);
}

/**
 * Run the whole pipeline for one repair round.
 *
 * Exposed separately from the loop below so the caller can decide how many rounds to
 * attempt; this function never loops.
 */
export async function runSplitOnce(
  opts: SplitOptions,
  repair?: { blockers: string[] },
  executorFeedback?: Map<string, string[]>,
  /** Reuse a plan instead of asking for a new one. */
  reusePlan?: Plan,
): Promise<SplitResult> {
  const onEvent = opts.onEvent ?? ((): void => {});
  const warnings: string[] = [];
  const files = opts.ctx.files ?? [];

  /*
   * A retry caused by a bad *edit* must not re-plan.
   *
   * The plan was not the problem — an ambiguous anchor was — and the planner is the
   * cloud model, which is the expensive half of this pipeline. Re-planning to fix the
   * cheap model's mistake doubled the cloud cost of a retry for no benefit, which the
   * first measurement made obvious: two planner calls at ~2s each for one unchanged plan.
   */
  const planned =
    reusePlan !== undefined
      ? { plan: reusePlan, text: '', ms: 0 }
      : await planTask({
          cfg: opts.cfg,
          ctx: opts.ctx,
          planner: opts.planner,
          ...(repair === undefined ? {} : { repair }),
        });

  if (planned.plan === null) {
    warnings.push('the planner did not return a usable plan');
    return {
      plan: { summary: '', steps: [] },
      steps: [],
      candidateText: planned.text,
      plannerMs: planned.ms,
      executorMs: 0,
      executorInputTokens: 0,
      executorOutputTokens: 0,
      warnings,
    };
  }

  onEvent({ type: 'planned', plan: planned.plan, plannerMs: planned.ms });

  // A question needs no execution: the planner's answer is the whole reply.
  if (planned.plan.steps.length === 0) {
    return {
      plan: planned.plan,
      steps: [],
      candidateText: JSON.stringify({ summary: planned.plan.summary, edits: [], answer: planned.plan.answer ?? '' }, null, 2),
      plannerMs: planned.ms,
      executorMs: 0,
      executorInputTokens: 0,
      executorOutputTokens: 0,
      warnings,
    };
  }

  const concurrency = Math.max(1, opts.concurrency ?? 1);
  const executorStarted = Date.now();

  const steps = await mapLimit(planned.plan.steps, concurrency, async (step): Promise<ExecutedStep> => {
    const started = Date.now();
    onEvent({ type: 'execute-start', file: step.file, intent: step.intent });
    try {
      const fb = executorFeedback?.get(step.file);
      const out = await executeStep({
        cfg: opts.cfg,
        executor: opts.executor,
        step,
        files,
        ...(fb === undefined ? {} : { feedback: fb }),
      });
      const durationMs = Date.now() - started;
      onEvent({
        type: 'execute-end',
        file: step.file,
        ok: out.edits.length > 0,
        edits: out.edits.length,
        durationMs,
        ...(out.error === undefined ? {} : { error: out.error }),
      });
      return {
        step,
        edits: out.edits,
        durationMs,
        inputTokens: out.inputTokens,
        outputTokens: out.outputTokens,
        ...(out.error === undefined ? {} : { error: out.error }),
      };
    } catch (err) {
      const durationMs = Date.now() - started;
      const error = err instanceof Error ? err.message : String(err);
      onEvent({ type: 'execute-end', file: step.file, ok: false, durationMs, error });
      return { step, edits: [], error, durationMs, inputTokens: 0, outputTokens: 0 };
    }
  });

  const executorMs = Date.now() - executorStarted;
  const failed = steps.filter((s) => s.edits.length === 0);
  if (failed.length > 0) {
    warnings.push(
      `${failed.length} of ${steps.length} step(s) produced no edit: ${failed.map((f) => f.step.file).join(', ')}`,
    );
  }

  return {
    plan: planned.plan,
    steps,
    candidateText: assembleCandidate(planned.plan, steps),
    plannerMs: planned.ms,
    executorMs,
    executorInputTokens: steps.reduce((a, s) => a + s.inputTokens, 0),
    executorOutputTokens: steps.reduce((a, s) => a + s.outputTokens, 0),
    warnings,
  };
}
