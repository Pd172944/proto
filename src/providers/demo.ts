/**
 * Demo provider — a scripted "model" that drives the real agent loop.
 *
 * Its only purpose is to let someone see the entire harness work with **zero
 * setup**: no API key, no downloaded model, no network. That matters because the
 * alternative way to evaluate a tool like this is to configure three things and
 * spend money before finding out whether the interface is any good.
 *
 * It is honest about what it is: the decisions are canned, but everything else is
 * real. The tool calls go through the real registry, the real approval prompts, the
 * real workspace, the real renderer and the real accounting. So `proto code --demo`
 * is a genuine end-to-end self-test of the harness — and of whether the local
 * environment is set up correctly — rather than a mock-up.
 *
 * It is deliberately **read-only**: it lists, searches, reads and runs one
 * harmless status command. A demo that edited your files would be a trap.
 */

import { computeCost } from './types.ts';
import type { ChatRequest, ChatResponse, Health, Provider, ProviderCapabilities, ToolCall } from './types.ts';
import { estimateTokens } from '../util/text.ts';

interface DemoStep {
  say?: string;
  call?: { name: string; args: Record<string, unknown> };
}

const CAPABILITIES: ProviderCapabilities = {
  tools: true,
  jsonSchema: true,
  streaming: false,
  promptCaching: false,
  contextWindow: 32_000,
  maxOutputTokens: 4096,
};

export class DemoProvider implements Provider {
  readonly id = 'demo';
  readonly label = 'Demo (scripted, read-only)';
  readonly kind = 'cloud' as const;
  readonly model = 'demo-harness';
  readonly capabilities = CAPABILITIES;

  private step = 0;
  private sawToolResult = false;

  async chat(req: ChatRequest): Promise<ChatResponse> {
    const last = req.messages[req.messages.length - 1];
    const lastTool = last?.role === 'tool' ? last.content : null;
    const usage = {
      inputTokens: estimateTokens(req.messages.map((m) => m.content).join('\n')),
      outputTokens: 40,
    };

    // The plan depends on what the tools actually returned, so the demo exercises
    // the real loop rather than replaying a fixed transcript.
    const plan = this.nextStep(lastTool);
    if (!plan.call) {
      return {
        text: plan.say ?? 'Demo finished.',
        toolCalls: [],
        usage,
        finishReason: 'stop',
        model: this.model,
        providerId: this.id,
        latencyMs: 5,
        costUsd: 0,
      };
    }

    const toolCalls: ToolCall[] = [{ id: `demo_${this.step}`, name: plan.call.name, args: plan.call.args }];
    return {
      text: '',
      toolCalls,
      usage,
      finishReason: 'tool_calls',
      model: this.model,
      providerId: this.id,
      latencyMs: 5,
      costUsd: computeCost(usage, { in: 0, out: 0 }),
    };
  }

  private nextStep(lastTool: string | null): DemoStep {
    if (lastTool === null) {
      this.step = 1;
      return {
        call: { name: 'list_files', args: { depth: 1 } },
      };
    }

    // After each tool result, decide the next move. This makes the demo adapt to
    // whatever project it is run in.
    this.step++;
    switch (this.step) {
      case 2: {
        const file = pickSourceFile(lastTool);
        if (file) return { call: { name: 'read_file', args: { path: file, limit: 60 } } };
        return { call: { name: 'search', args: { pattern: 'TODO' } } };
      }
      case 3:
        return { call: { name: 'run_command', args: { command: 'git status --short || echo "not a git repository"' } } };
      default:
        return {
          say: [
            'That was the demo provider: the decisions were canned, but everything else was real — the same tool',
            'registry, approval flow, workspace, renderer and cost accounting used by a live session.',
            '',
            'Nothing was modified. To use a real model:',
            '  1. export ANTHROPIC_API_KEY=...   (or OPENROUTER_API_KEY / OPENAI_API_KEY)',
            '  2. proto doctor --probe-cloud',
            '  3. proto code                      (or `proto code --local` for a local model)',
            '',
            'To try a local model with no key: `ollama pull ornith-1.5:9b && proto code --local`.',
          ].join('\n'),
        };
    }
  }

  async health(): Promise<Health> {
    return { ok: true, detail: 'demo provider always available (read-only, scripted)' };
  }
}

/** Pick the first plausible source file out of a `list_files` transcript. */
function pickSourceFile(listing: string): string | null {
  const candidates: string[] = [];
  for (const rawLine of listing.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('▸')) continue;
    // list_files renders "  ▸ dir/" for directories and "  name" for files.
    const match = line.match(/^[▸\s]*([A-Za-z0-9._-]+\.[A-Za-z0-9]+)$/);
    if (match?.[1]) candidates.push(match[1]);
  }
  const preferred = candidates.find((c) => /\.(ts|tsx|js|jsx|py|go|rs|rb|java|kt|swift|c|h|cpp|cs)$/i.test(c));
  return preferred ?? candidates[0] ?? null;
}
