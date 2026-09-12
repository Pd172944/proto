/**
 * HTTP with retries, timeouts and provider-agnostic error classification.
 *
 * Every network call in the harness funnels through `request()`:
 *  - hard timeout via AbortController (a hung local model must not wedge the
 *    harness forever — this is the single most common failure mode for local
 *    inference servers),
 *  - retry with exponential backoff + jitter, honouring `Retry-After`,
 *  - classification of retryable (429/5xx/network) vs terminal (4xx) failures,
 *  - optional per-attempt logging (never logs bodies, which may contain code).
 */

export interface HttpRequest {
  url: string;
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  headers?: Record<string, string>;
  body?: string;
  timeoutMs?: number;
  retries?: number;
  signal?: AbortSignal;
  /** Label used in log lines, e.g. the provider id. */
  label?: string;
}

export interface HttpResponse {
  status: number;
  ok: boolean;
  headers: Record<string, string>;
  text: string;
  latencyMs: number;
  attempts: number;
}

export class HttpError extends Error {
  readonly status: number;
  readonly retryable: boolean;
  readonly body: string;
  constructor(message: string, status: number, retryable: boolean, body = '') {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.retryable = retryable;
    this.body = body;
  }
}

const RETRYABLE_STATUS = new Set([408, 409, 425, 429, 500, 502, 503, 504, 522, 524]);

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    if (signal) {
      const onAbort = (): void => {
        clearTimeout(t);
        reject(new Error('aborted'));
      };
      if (signal.aborted) onAbort();
      else signal.addEventListener('abort', onAbort, { once: true });
    }
  });
}

function parseRetryAfter(headers: Headers): number | null {
  const raw = headers.get('retry-after');
  if (!raw) return null;
  const secs = Number(raw);
  if (Number.isFinite(secs)) return Math.min(secs * 1000, 30_000);
  const date = Date.parse(raw);
  if (Number.isFinite(date)) return Math.max(0, Math.min(date - Date.now(), 30_000));
  return null;
}

export async function request(req: HttpRequest): Promise<HttpResponse> {
  const retries = req.retries ?? 2;
  const timeoutMs = req.timeoutMs ?? 60_000;
  let attempt = 0;
  let lastErr: unknown;

  while (attempt <= retries) {
    attempt++;
    const started = Date.now();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error('timeout')), timeoutMs);
    const onOuterAbort = (): void => controller.abort(new Error('aborted'));
    if (req.signal) {
      if (req.signal.aborted) controller.abort(new Error('aborted'));
      else req.signal.addEventListener('abort', onOuterAbort, { once: true });
    }

    try {
      const res = await fetch(req.url, {
        method: req.method ?? 'POST',
        headers: req.headers,
        body: req.body,
        signal: controller.signal,
      });
      const text = await res.text();
      const headers: Record<string, string> = {};
      res.headers.forEach((v, k) => {
        headers[k.toLowerCase()] = v;
      });
      const latencyMs = Date.now() - started;

      if (!res.ok && RETRYABLE_STATUS.has(res.status) && attempt <= retries) {
        const wait = parseRetryAfter(res.headers) ?? backoff(attempt);
        lastErr = new HttpError(
          `HTTP ${res.status} from ${req.label ?? req.url}`,
          res.status,
          true,
          text.slice(0, 500),
        );
        await sleep(wait, req.signal);
        continue;
      }

      return { status: res.status, ok: res.ok, headers, text, latencyMs, attempts: attempt };
    } catch (err) {
      lastErr = err;
      const aborted = req.signal?.aborted === true;
      if (aborted) throw new Error('request aborted');
      if (attempt > retries) break;
      await sleep(backoff(attempt), req.signal);
    } finally {
      clearTimeout(timer);
      if (req.signal) req.signal.removeEventListener('abort', onOuterAbort);
    }
  }

  const detail = lastErr instanceof Error ? lastErr.message : String(lastErr);
  throw new HttpError(
    `request failed after ${attempt} attempt(s): ${detail}`,
    0,
    true,
    lastErr instanceof HttpError ? lastErr.body : '',
  );
}

function backoff(attempt: number): number {
  const base = Math.min(500 * 2 ** (attempt - 1), 8000);
  return base + Math.floor(Math.random() * 250);
}

export function joinUrl(base: string, path: string): string {
  return `${base.replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`;
}
