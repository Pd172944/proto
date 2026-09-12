/**
 * Ollama provider (native `/api/chat`).
 *
 * We talk to the native API rather than Ollama's OpenAI-compatible shim because
 * two knobs matter directly to this project's goals:
 *
 *  - `keep_alive`: how long the model stays resident in RAM. A coding harness
 *    that leaves a 7B model pinned forever would violate "as little load on the
 *    user's computer as possible", so the harness sends `keep_alive: 0` after a
 *    batch of work (`unload()`) and a short keep-alive during it.
 *  - `options.num_ctx` / `num_predict`: explicit KV-cache and output bounds.
 *    Ollama's defaults are context-length dependent and can balloon memory.
 *
 * It also reports `load_duration`/`prompt_eval_count`, which gives the router a
 * real measured cold-start signal instead of a guess.
 */

import { emptyResponse, ProviderError, computeCost } from './types.ts';
import type {
  ChatRequest,
  ChatResponse,
  FinishReason,
  Health,
  Provider,
  ProviderCapabilities,
  ToolCall,
  Usage,
} from './types.ts';
import { HttpError, joinUrl, request } from '../util/http.ts';
import { estimateTokens } from '../util/text.ts';

export interface OllamaOptions {
  id: string;
  baseUrl: string;
  model: string;
  keepAliveSec: number;
  timeoutMs: number;
  contextWindow: number;
  maxOutputTokens: number;
  temperature: number;
  /** Free local compute, but electricity is not free; a nominal rate keeps budgets honest. */
  pricePerMillionTokens?: number;
}

interface OllamaTags {
  models?: Array<{ name?: string; model?: string; size?: number; details?: { parameter_size?: string } }>;
}

export class OllamaProvider implements Provider {
  readonly id: string;
  readonly label = 'Ollama (local)';
  readonly kind = 'local' as const;
  readonly model: string;
  readonly baseUrl: string;
  readonly capabilities: ProviderCapabilities;
  private readonly keepAliveSec: number;
  private readonly timeoutMs: number;
  private readonly contextWindow: number;
  private readonly maxOutputTokens: number;
  private readonly temperature: number;
  private readonly pricePerMillionTokens: number;

  constructor(opts: OllamaOptions) {
    this.id = opts.id;
    this.baseUrl = opts.baseUrl;
    this.model = opts.model;
    this.keepAliveSec = opts.keepAliveSec;
    this.timeoutMs = opts.timeoutMs;
    this.contextWindow = opts.contextWindow;
    this.maxOutputTokens = opts.maxOutputTokens;
    this.temperature = opts.temperature;
    this.pricePerMillionTokens = opts.pricePerMillionTokens ?? 0;
    this.capabilities = {
      tools: true,
      jsonSchema: false,
      streaming: false,
      promptCaching: false,
      contextWindow: opts.contextWindow,
      maxOutputTokens: opts.maxOutputTokens,
    };
  }

  async chat(req: ChatRequest): Promise<ChatResponse> {
    const body: Record<string, unknown> = {
      model: this.model,
      messages: req.messages.map((m) => ({
        role: m.role,
        content: m.content,
        ...(m.role === 'tool' ? { tool_name: m.name } : {}),
      })),
      stream: false,
      keep_alive: `${Math.max(0, this.keepAliveSec)}s`,
      options: {
        temperature: req.temperature ?? this.temperature,
        num_predict: req.maxTokens ?? this.maxOutputTokens,
        num_ctx: this.contextWindow,
        ...(req.stop?.length ? { stop: req.stop } : {}),
      },
    };
    if (req.jsonSchema) body['format'] = req.jsonSchema;
    if (req.tools?.length) {
      body['tools'] = req.tools.map((t) => ({
        type: 'function',
        function: { name: t.name, description: t.description, parameters: t.parameters },
      }));
    }

    const started = Date.now();
    let res;
    try {
      res = await request({
        url: joinUrl(this.baseUrl, 'api/chat'),
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
        timeoutMs: this.timeoutMs,
        signal: req.signal,
        retries: 0, // a local server either answers or is not running; retrying wastes the user's time
        label: this.id,
      });
    } catch (err) {
      throw this.transportError(err);
    }
    const latencyMs = Date.now() - started;

    if (!res.ok) {
      const detail = res.text.slice(0, 300);
      const missingModel = /not found|no such model|pull/i.test(detail);
      throw new ProviderError(this.id, `Ollama HTTP ${res.status}: ${detail}`, {
        status: res.status,
        retryable: res.status >= 500,
        hint: missingModel
          ? `Model "${this.model}" is not downloaded. Run: ollama pull ${this.model}`
          : undefined,
      });
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(res.text);
    } catch {
      return emptyResponse(this.id, this.model, latencyMs, 'Ollama returned invalid JSON');
    }
    return this.normalize(parsed, latencyMs, req);
  }

  private normalize(parsed: unknown, latencyMs: number, req: ChatRequest): ChatResponse {
    const root = (parsed ?? {}) as Record<string, unknown>;
    const message = (root['message'] ?? {}) as Record<string, unknown>;
    const text = typeof message['content'] === 'string' ? message['content'] : '';

    const toolCalls: ToolCall[] = [];
    if (Array.isArray(message['tool_calls'])) {
      for (const [i, raw] of (message['tool_calls'] as unknown[]).entries()) {
        const tc = (raw ?? {}) as Record<string, unknown>;
        const fn = (tc['function'] ?? {}) as Record<string, unknown>;
        const name = typeof fn['name'] === 'string' ? fn['name'] : '';
        if (!name) continue;
        let args: unknown = fn['arguments'];
        if (typeof args === 'string') {
          try {
            args = JSON.parse(args);
          } catch {
            /* keep raw */
          }
        }
        toolCalls.push({ id: `ollama_${i}`, name, args });
      }
    }

    const promptTokens = numOr(root['prompt_eval_count'], estimateTokens(req.messages.map((m) => m.content).join('\n')));
    const outputTokens = numOr(root['eval_count'], estimateTokens(text));
    const usage: Usage = { inputTokens: promptTokens, outputTokens };
    const loadDurationNs = typeof root['load_duration'] === 'number' ? (root['load_duration'] as number) : 0;
    if (loadDurationNs > 0) usage.estimated = false;

    const finish: FinishReason = root['done_reason'] === 'length' ? 'length' : toolCalls.length ? 'tool_calls' : 'stop';
    if (!text && toolCalls.length === 0) {
      return { ...emptyResponse(this.id, this.model, latencyMs, 'Ollama returned an empty message'), usage };
    }

    return {
      text,
      toolCalls,
      usage,
      finishReason: finish,
      model: typeof root['model'] === 'string' ? (root['model'] as string) : this.model,
      providerId: this.id,
      latencyMs,
      costUsd: computeCost(usage, { in: this.pricePerMillionTokens, out: this.pricePerMillionTokens }),
    };
  }

  /** Best-effort, cheap model list. Used by `doctor` and `models list`. */
  async list(): Promise<string[]> {
    try {
      const res = await request({
        url: joinUrl(this.baseUrl, 'api/tags'),
        method: 'GET',
        timeoutMs: 5000,
        retries: 0,
        label: `${this.id}-tags`,
      });
      if (!res.ok) return [];
      const parsed = JSON.parse(res.text) as OllamaTags;
      return (parsed.models ?? [])
        .map((m) => m.name ?? m.model ?? '')
        .filter(Boolean)
        .sort();
    } catch {
      return [];
    }
  }

  async health(): Promise<Health> {
    const started = Date.now();
    try {
      const models = await this.list();
      const latencyMs = Date.now() - started;
      if (models.length === 0) {
        return {
          ok: false,
          detail: `Ollama is not responding at ${this.baseUrl}`,
          latencyMs,
          unreachable: true,
          hint: 'Start it with `ollama serve` (or the Ollama desktop app). See docs/local-models.md.',
        };
      }
      const present = models.some((m) => m === this.model || m.startsWith(`${this.model}:`));
      return {
        ok: present,
        detail: present
          ? `Ollama up, "${this.model}" is downloaded`
          : `Ollama up, but "${this.model}" is not downloaded`,
        latencyMs,
        models,
        hint: present ? undefined : `Run: ollama pull ${this.model}`,
      };
    } catch (err) {
      return {
        ok: false,
        detail: err instanceof Error ? err.message : String(err),
        latencyMs: Date.now() - started,
        unreachable: true,
        hint: 'Start it with `ollama serve`. See docs/local-models.md.',
      };
    }
  }

  /** Ask Ollama to evict the model from RAM immediately. */
  async unload(): Promise<boolean> {
    try {
      const res = await request({
        url: joinUrl(this.baseUrl, 'api/generate'),
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: this.model, keep_alive: 0 }),
        timeoutMs: 10_000,
        retries: 0,
        label: `${this.id}-unload`,
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  private transportError(err: unknown): ProviderError {
    if (err instanceof HttpError) {
      return new ProviderError(this.id, err.message, { status: err.status, retryable: err.retryable });
    }
    const msg = err instanceof Error ? err.message : String(err);
    return new ProviderError(this.id, msg, {
      retryable: true,
      hint: `Could not reach Ollama at ${this.baseUrl}. Start it with \`ollama serve\`.`,
    });
  }
}

function numOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}
