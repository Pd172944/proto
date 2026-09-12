/**
 * Deterministic mock provider.
 *
 * Used for three real purposes (not just tests):
 *  1. `proto eval --live` needs a zero-cost stand-in to exercise the full
 *     pipeline in CI without spending money or requiring a GPU.
 *  2. Tests must never touch the network or the user's local runtime.
 *  3. `proto run --provider mock` lets a user see the whole routing/verification
 *     path before configuring anything at all.
 *
 * The mock can be told to fail on purpose, which is how we exercise escalation.
 */

import { computeCost } from './types.ts';
import type { ChatRequest, ChatResponse, Health, Provider, ProviderCapabilities, ProviderKind } from './types.ts';
import { estimateTokens } from '../util/text.ts';

export interface MockOptions {
  id?: string;
  kind?: ProviderKind;
  model?: string;
  /** 0..1 — probability that this provider returns unusable output. */
  failureRate?: number;
  /** Fixed latency to simulate. */
  latencyMs?: number;
  /** Canned response text. `{{prompt}}` is replaced with the last user message. */
  response?: string;
  capabilities?: Partial<ProviderCapabilities>;
  price?: { in: number; out: number };
}

const DEFAULT_RESPONSE = JSON.stringify(
  {
    summary: 'mock candidate',
    edits: [
      {
        file: 'example.py',
        find: 'for i in range(len(items)):',
        replace: 'for item in items:',
      },
    ],
    risk: 'low',
  },
  null,
  2,
);

/**
 * Build a plausible, *verifiable* candidate from the prompt itself.
 *
 * A mock that returns a fixed string is useless for exercising the pipeline: it
 * fails verification for reasons that have nothing to do with the code under
 * test, so every demo and test ends up in the escalation path. Instead we pull
 * the first file body out of the prompt and append a comment in that language —
 * syntactically valid, different from the original, and targeted at a real path.
 * That makes `proto run --mock` a faithful dry run of the whole loop.
 */
function synthesizeCandidate(prompt: string): string {
  const fileMatch = prompt.match(/FILE: (\S+)\n```[A-Za-z0-9+#]*\n([\s\S]*?)\n```/);
  if (!fileMatch) return DEFAULT_RESPONSE;
  const path = fileMatch[1] as string;
  const body = fileMatch[2] as string;
  const comment = commentFor(path);
  if (comment === null) return DEFAULT_RESPONSE; // unknown language: let it fail honestly
  const edited = `${body}\n${comment} proto mock: deterministic edit for pipeline testing\n`;
  return JSON.stringify(
    { summary: `mock edit of ${path}`, edits: [{ file: path, content: edited }], risk: 'low' },
    null,
    2,
  );
}

function commentFor(path: string): string | null {
  if (/\.(py|sh|bash|zsh|rb|yml|yaml|toml|r)$/i.test(path)) return '#';
  if (/\.(ts|tsx|js|jsx|mjs|cjs|go|rs|java|kt|c|cc|cpp|h|hpp|cs|swift|php)$/i.test(path)) return '//';
  if (/\.(sql)$/i.test(path)) return '--';
  return null;
}

/** Deterministic PRNG so the same call index always yields the same outcome. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class MockProvider implements Provider {
  readonly id: string;
  readonly label: string;
  readonly kind: ProviderKind;
  readonly model: string;
  readonly capabilities: ProviderCapabilities;
  private readonly failureRate: number;
  private readonly latencyMs: number;
  private readonly response: string;
  /** True when the caller supplied no canned response, so we synthesize per-prompt. */
  private readonly synthesize: boolean;
  private readonly price: { in: number; out: number };
  private calls = 0;
  /** Every request this instance has seen; handy for assertions. */
  readonly history: ChatRequest[] = [];

  constructor(opts: MockOptions = {}) {
    this.id = opts.id ?? 'mock';
    this.label = `Mock (${this.id})`;
    this.kind = opts.kind ?? 'local';
    this.model = opts.model ?? 'mock-model';
    this.failureRate = opts.failureRate ?? 0;
    this.latencyMs = opts.latencyMs ?? 5;
    this.response = opts.response ?? DEFAULT_RESPONSE;
    this.synthesize = opts.response === undefined;
    this.price = opts.price ?? { in: 0, out: 0 };
    this.capabilities = {
      tools: false,
      jsonSchema: true,
      streaming: false,
      promptCaching: false,
      contextWindow: 8192,
      maxOutputTokens: 2048,
      ...(opts.capabilities ?? {}),
    };
  }

  async chat(req: ChatRequest): Promise<ChatResponse> {
    this.calls++;
    const rand = mulberry32(this.calls * 7919);
    await new Promise((r) => setTimeout(r, this.latencyMs));
    const lastUser = [...req.messages].reverse().find((m) => m.role === 'user');

    if (this.failureRate > 0 && rand() < this.failureRate) {
      const broken = 'Sure! Here is a broken snippet:\n```python\nfor i in range(len(items)\n```';
      const usage = {
        inputTokens: estimateTokens(lastUser?.content ?? ''),
        outputTokens: estimateTokens(broken),
      };
      this.history.push(req);
      return {
        text: broken,
        toolCalls: [],
        usage,
        finishReason: 'stop',
        model: this.model,
        providerId: this.id,
        latencyMs: this.latencyMs,
        costUsd: computeCost(usage, this.price),
      };
    }

    const text = this.response.replace('{{prompt}}', lastUser?.content ?? '');
    // When no canned response was supplied, synthesize one from the prompt so the
    // candidate targets a real file and can actually pass verification.
    const finalText = this.synthesize ? synthesizeCandidate(lastUser?.content ?? '') : text;
    const usage = {
      inputTokens: estimateTokens(req.messages.map((m) => m.content).join('\n')),
      outputTokens: estimateTokens(finalText),
    };
    this.history.push(req);
    return {
      text: finalText,
      toolCalls: [],
      usage,
      finishReason: 'stop',
      model: this.model,
      providerId: this.id,
      latencyMs: this.latencyMs,
      costUsd: computeCost(usage, this.price),
    };
  }

  async health(): Promise<Health> {
    return { ok: true, detail: `mock provider "${this.id}" always healthy`, latencyMs: 0 };
  }
}

/** A mock that always fails verification — drives the escalation path in tests. */
export function alwaysFailingMock(id = 'mock-bad'): MockProvider {
  return new MockProvider({ id, failureRate: 1 });
}
