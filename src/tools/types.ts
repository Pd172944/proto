/**
 * Tool contract for the agent harness.
 *
 * Design notes, drawn from what the mature coding agents get right:
 *
 *  - **Risk is declared per tool, not per call site.** A read is always safe; a
 *    write or a shell command is not. Declaring it on the tool means the approval
 *    layer cannot be bypassed by adding a new call path later.
 *  - **Every tool returns a structured result**, not a string. The transcript
 *    renderer, the cost accounting and the (future) verification layer all need
 *    the metadata; flattening to text at the boundary loses it forever.
 *  - **Tools are pure functions of (args, context).** No hidden global state, so a
 *    tool can be unit-tested without a session or a terminal.
 *  - **Output is truncated at the tool, not at the model.** A 200k-character file
 *    read must not silently blow the context window; the tool reports what it
 *    dropped so the model can ask for a narrower range.
 */

import { isAbsolute, relative, resolve, sep } from 'node:path';

export type Risk = 'read' | 'write' | 'exec';

export interface ApprovalRequest {
  kind: Risk;
  /** One-line summary, e.g. `Write src/app.ts`. */
  title: string;
  /** Human-readable detail: the diff, the command, the paths. */
  detail: string;
  /** Extra context that helps a user decide, e.g. "outside the workspace". */
  warning?: string;
}

export type ApprovalDecision = 'allow-once' | 'allow-always' | 'deny';

export interface ToolContext {
  /** Absolute path of the project root the agent may touch. */
  workspace: string;
  /** Ask the user. Non-interactive runs must answer without prompting. */
  approve(request: ApprovalRequest): Promise<ApprovalDecision>;
  /** Cooperative cancellation from Ctrl-C. */
  signal?: AbortSignal;
  /** False for `proto agent --print` style one-shot runs. */
  interactive: boolean;
  /** Hard cap on characters any single tool may return. */
  maxOutputChars: number;
}

export interface ToolResult {
  ok: boolean;
  /** One-line label for the transcript, e.g. `Read src/app.ts (128 lines)`. */
  title: string;
  /** Model-facing content. */
  output: string;
  /** Structured extras for the renderer and for accounting. */
  meta?: {
    /** Files this call touched, for a change summary. */
    files?: string[];
    /** Lines added/removed, when the tool knows. */
    added?: number;
    removed?: number;
    /** True when the output was truncated. */
    truncated?: boolean;
    /** Wall-clock duration. */
    durationMs?: number;
    [key: string]: unknown;
  };
}

export interface Tool {
  /** Snake-case, stable: the model sees this name. */
  name: string;
  /** Shown to the model. Must say *when* to use it, not just what it does. */
  description: string;
  /** JSON Schema for the arguments. */
  parameters: Record<string, unknown>;
  risk: Risk;
  run(args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult>;
}

/** Registry lookup plus schema export for the provider call. */
export class ToolRegistry {
  private readonly tools = new Map<string, Tool>();

  register(tool: Tool): this {
    if (this.tools.has(tool.name)) throw new Error(`duplicate tool: ${tool.name}`);
    this.tools.set(tool.name, tool);
    return this;
  }

  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  list(): Tool[] {
    return [...this.tools.values()];
  }

  /** Provider-facing tool list. */
  specs(): Array<{ name: string; description: string; parameters: Record<string, unknown> }> {
    return this.list().map((t) => ({ name: t.name, description: t.description, parameters: t.parameters }));
  }

  /** Tools grouped by risk, used by `/tools` and by the dry-run banner. */
  byRisk(): Record<Risk, string[]> {
    const out: Record<Risk, string[]> = { read: [], write: [], exec: [] };
    for (const t of this.list()) out[t.risk].push(t.name);
    return out;
  }
}

/** Truncate tool output, keeping the head and tail and saying what was dropped. */
export function capOutput(text: string, maxChars: number): { text: string; truncated: boolean } {
  if (text.length <= maxChars) return { text, truncated: false };
  const head = Math.floor(maxChars * 0.7);
  const tail = Math.max(0, maxChars - head - 120);
  const dropped = text.length - head - tail;
  return {
    text: `${text.slice(0, head)}\n\n… [${dropped} characters truncated; narrow the request or read a line range] …\n\n${text.slice(text.length - tail)}`,
    truncated: true,
  };
}

/** Reject absolute paths and traversal; returns a workspace-relative path. */
export function resolveInsideWorkspace(
  workspace: string,
  p: string,
): { ok: true; abs: string; rel: string } | { ok: false; error: string } {
  const root = resolve(workspace);
  const abs = isAbsolute(p) ? resolve(p) : resolve(root, p);
  if (abs !== root && !abs.startsWith(root + sep)) {
    return { ok: false, error: `path escapes the workspace: ${p}` };
  }
  return { ok: true, abs, rel: relative(root, abs) || '.' };
}
