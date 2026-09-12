/**
 * The agent loop.
 *
 *   user turn → model → tool calls → results → model → … → final answer
 *
 * Why this shape, and not something more elaborate:
 *
 *  - **Linear, single-threaded, one conversation.** Cognition's public post-mortem
 *    on multi-agent systems argues that splitting one task across parallel agents
 *    fragments the context and produces conflicting decisions. Everything here
 *    shares one message history, so a decision made in step 2 is visible in step 9.
 *  - **Every step is bounded.** A maximum step count and a wall-clock deadline stop
 *    a model that loops forever from costing the user real money. Being killed by
 *    the budget is a normal outcome, not a crash.
 *  - **Tool errors are data.** A failed tool returns `ok: false` and the model sees
 *    the reason. Throwing would end the turn and lose the whole context of what it
 *    was doing.
 *  - **The transcript is an event stream.** The loop does not render anything; it
 *    emits events. That keeps the same loop usable from the TUI, from a one-shot
 *    `--print` run, and from tests, without a terminal attached.
 *
 * Note on streaming: when the provider supports `chatStream`, text is forwarded as
 * it arrives so the user sees progress rather than a stall. The loop falls back to
 * `chat()` otherwise, and the *resulting* `ChatResponse` is the same either way, so
 * streaming never changes behaviour or cost accounting.
 */

import type { ChatRequest, ChatResponse, Message, Provider, ToolCall } from '../providers/types.ts';
import { ProviderError } from '../providers/types.ts';
import type { Tool, ToolContext, ToolRegistry, ToolResult } from '../tools/types.ts';
import { buildSystemPrompt, gatherProjectContext, looksLikeVerification, turnReminder } from './prompt.ts';
import type { ProjectContext, TurnState } from './prompt.ts';
import { HARNESS_VERSION } from '../version.ts';

/* ------------------------------------------------------------------ */
/* Events                                                              */
/* ------------------------------------------------------------------ */

export type AgentEvent =
  | { type: 'context'; project: ProjectContext; systemPrompt: string }
  | { type: 'model-start'; step: number }
  | { type: 'text-delta'; text: string }
  | { type: 'text'; text: string }
  | { type: 'reasoning'; text: string }
  | { type: 'tool-start'; toolCall: ToolCall; tool: Tool | null }
  | { type: 'tool-end'; toolCall: ToolCall; result: ToolResult }
  | { type: 'approval'; title: string; decision: string }
  | { type: 'reminder'; text: string }
  | { type: 'notice'; level: 'info' | 'warn' | 'error'; text: string }
  | { type: 'usage'; usage: { inputTokens: number; outputTokens: number; cachedInputTokens?: number }; costUsd: number }
  | { type: 'done'; reason: 'complete' | 'max-steps' | 'deadline' | 'aborted' | 'error'; steps: number; text: string };

export interface AgentResult {
  text: string;
  steps: number;
  reason: 'complete' | 'max-steps' | 'deadline' | 'aborted' | 'error';
  messages: Message[];
  usage: { inputTokens: number; outputTokens: number; cachedInputTokens: number };
  costUsd: number;
  editedFiles: string[];
  commands: string[];
  ranVerification: boolean;
}

/* ------------------------------------------------------------------ */
/* Options                                                            */
/* ------------------------------------------------------------------ */

export interface AgentOptions {
  provider: Provider;
  tools: ToolRegistry;
  workspace: string;
  /** Conversation so far, excluding the system prompt (which is prepended here). */
  history: Message[];
  /** The new user message for this turn. */
  input: string;
  maxSteps?: number;
  /** Wall-clock budget for the whole turn. */
  deadlineMs?: number;
  maxOutputTokens?: number;
  temperature?: number;
  /** Called for every event; the caller renders. */
  onEvent?: (event: AgentEvent) => void;
  /** Approval callback for risky tools. */
  approve: (req: Parameters<ToolContext['approve']>[0]) => Promise<'allow-once' | 'allow-always' | 'deny'>;
  signal?: AbortSignal;
  interactive?: boolean;
  maxToolOutputChars?: number;
  /** Reuse a previously gathered context (cheaper, and stable within a session). */
  project?: ProjectContext;
  /** Skip the verification reminder (used by tests). */
  skipReminders?: boolean;
}

/* ------------------------------------------------------------------ */
/* The loop                                                           */
/* ------------------------------------------------------------------ */

export async function runAgentTurn(opts: AgentOptions): Promise<AgentResult> {
  const onEvent = opts.onEvent ?? ((): void => {});
  const maxSteps = opts.maxSteps ?? 40;
  const deadline = Date.now() + (opts.deadlineMs ?? 10 * 60_000);
  const interactive = opts.interactive ?? true;
  const maxToolOutputChars = opts.maxToolOutputChars ?? 24_000;

  const project = opts.project ?? (await gatherProjectContext(opts.workspace));
  const systemPrompt = buildSystemPrompt(project, { tools: opts.tools.list().map((t) => t.name) });
  onEvent({ type: 'context', project, systemPrompt });

  const messages: Message[] = [
    { role: 'system', content: systemPrompt, cacheBreakpoint: true },
    ...opts.history,
    { role: 'user', content: opts.input },
  ];

  const state: TurnState = { edited: [], commands: [], ranVerification: false, idleTurns: 0 };
  const usage = { inputTokens: 0, outputTokens: 0, cachedInputTokens: 0 };
  let costUsd = 0;
  let steps = 0;
  let finalText = '';
  let reason: AgentResult['reason'] = 'complete';

  for (let step = 1; step <= maxSteps; step++) {
    steps = step;

    if (opts.signal?.aborted) {
      reason = 'aborted';
      break;
    }
    if (Date.now() > deadline) {
      reason = 'deadline';
      onEvent({ type: 'notice', level: 'warn', text: `Stopped: the ${Math.round((opts.deadlineMs ?? 600_000) / 1000)}s budget for this turn ran out.` });
      break;
    }

    // A reminder is appended as a user message *before* the next model call, so it
    // lands where the model is actually deciding what to do next.
    if (!opts.skipReminders) {
      const reminder = turnReminder(state);
      if (reminder) {
        messages.push({ role: 'user', content: `<reminder>${reminder}</reminder>` });
        onEvent({ type: 'reminder', text: reminder });
      }
    }

    onEvent({ type: 'model-start', step });

    const request: ChatRequest = {
      messages,
      tools: opts.tools.specs(),
      maxTokens: opts.maxOutputTokens ?? 8192,
      temperature: opts.temperature ?? 0.2,
      ...(opts.signal ? { signal: opts.signal } : {}),
      meta: { agentStep: step },
    };

    let response: ChatResponse;
    try {
      response = await callModel(opts.provider, request, onEvent);
    } catch (err) {
      const message = err instanceof ProviderError ? `${err.message}${err.hint ? ` (${err.hint})` : ''}` : String(err);
      onEvent({ type: 'notice', level: 'error', text: message });
      reason = 'error';
      finalText = finalText || `Request failed: ${message}`;
      break;
    }

    usage.inputTokens += response.usage.inputTokens;
    usage.outputTokens += response.usage.outputTokens;
    usage.cachedInputTokens += response.usage.cachedInputTokens ?? 0;
    costUsd += response.costUsd;
    onEvent({
      type: 'usage',
      usage: {
        inputTokens: response.usage.inputTokens,
        outputTokens: response.usage.outputTokens,
        ...(response.usage.cachedInputTokens !== undefined ? { cachedInputTokens: response.usage.cachedInputTokens } : {}),
      },
      costUsd: response.costUsd,
    });

    if (response.text.trim()) {
      finalText = response.text;
      onEvent({ type: 'text', text: response.text });
    }

    if (response.toolCalls.length === 0) {
      // No tool calls means the model considers the turn finished. The final
      // assistant message must still be appended: without it the transcript ends on
      // a tool result, so the next user turn has no record of what the agent just
      // said and will happily contradict itself.
      messages.push({ role: 'assistant', content: response.text || '' });
      state.idleTurns = finalText.trim() ? 0 : state.idleTurns + 1;
      if (response.finishReason === 'error') {
        reason = 'error';
      }
      break;
    }

    state.idleTurns = 0;

    // Record the assistant turn including its tool calls, so the model sees its own
    // decisions on the next step. Text is preserved verbatim.
    messages.push({
      role: 'assistant',
      content: response.text || '',
      toolCalls: response.toolCalls,
    });

    for (const toolCall of response.toolCalls) {
      const tool = opts.tools.get(toolCall.name);
      onEvent({ type: 'tool-start', toolCall, tool: tool ?? null });

      if (!tool) {
        const available = opts.tools.list().map((t) => t.name).join(', ');
        const result: ToolResult = {
          ok: false,
          title: `${toolCall.name} (unknown tool)`,
          output: `No tool named "${toolCall.name}". Available tools: ${available}.`,
        };
        messages.push(toolResultMessage(toolCall, result));
        onEvent({ type: 'tool-end', toolCall, result });
        continue;
      }

      const ctx: ToolContext = {
        workspace: opts.workspace,
        interactive,
        maxOutputChars: maxToolOutputChars,
        approve: async (req) => {
          const decision = await opts.approve(req);
          onEvent({ type: 'approval', title: req.title, decision });
          return decision;
        },
        ...(opts.signal ? { signal: opts.signal } : {}),
      };

      let result: ToolResult;
      try {
        result = await tool.run(asObject(toolCall.args), ctx);
      } catch (err) {
        result = {
          ok: false,
          title: `${toolCall.name} threw`,
          output: `The tool raised an error: ${err instanceof Error ? err.message : String(err)}`,
        };
      }

      if (result.ok && (tool.name === 'edit_file' || tool.name === 'write_file')) {
        for (const f of result.meta?.files ?? []) if (!state.edited.includes(f)) state.edited.push(f);
      }
      if (tool.name === 'run_command') {
        const command = typeof toolCall.args === 'object' && toolCall.args !== null
          ? String((toolCall.args as Record<string, unknown>)['command'] ?? '')
          : '';
        if (command) state.commands.push(command);
        if (looksLikeVerification(command)) state.ranVerification = true;
      }

      messages.push(toolResultMessage(toolCall, result));
      onEvent({ type: 'tool-end', toolCall, result });
    }
  }

  if (steps >= maxSteps) {
    reason = 'max-steps';
    onEvent({ type: 'notice', level: 'warn', text: `Stopped after ${maxSteps} steps without finishing.` });
  }

  onEvent({ type: 'done', reason, steps, text: finalText });

  return {
    text: finalText,
    steps,
    reason,
    messages,
    usage,
    costUsd,
    editedFiles: state.edited,
    commands: state.commands,
    ranVerification: state.ranVerification,
  };
}

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

async function callModel(
  provider: Provider,
  request: ChatRequest,
  onEvent: (event: AgentEvent) => void,
): Promise<ChatResponse> {
  if (typeof provider.chatStream === 'function') {
    let streamed = '';
    return await provider.chatStream(request, (event) => {
      if (event.type === 'text-delta') {
        streamed += event.text;
        onEvent({ type: 'text-delta', text: event.text });
      }
    });
  }
  return await provider.chat(request);
}

function toolResultMessage(toolCall: ToolCall, result: ToolResult): Message {
  const header = result.ok ? `[ok] ${result.title}` : `[failed] ${result.title}`;
  return {
    role: 'tool',
    name: toolCall.name,
    toolCallId: toolCall.id,
    content: `${header}\n\n${result.output}`,
  };
}

function asObject(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {};
}

/** Rough token estimate, used by the TUI when a provider reports nothing. */
export function estimateMessagesTokens(messages: Message[]): number {
  let chars = 0;
  for (const m of messages) chars += m.content.length;
  return Math.max(1, Math.round(chars / 3.6));
}

export { HARNESS_VERSION };
