/**
 * Transport-failure diagnosis.
 *
 * Two problems this solves, both of which cost a user real time:
 *
 *  1. **`fetch failed` says nothing.** Node's fetch wraps the real reason — `ENOTFOUND`,
 *     `ECONNRESET`, a TLS error — in `err.cause`, and the top-level message is always
 *     the useless string `fetch failed`. Without walking the cause chain, a DNS failure,
 *     a VPN problem, an expired certificate and a dead local server are indistinguishable.
 *
 *  2. **A single hint cannot fit both tiers.** "Could not reach … Is the server running?"
 *     is exactly right for a local runtime you forgot to start, and actively misleading
 *     for `api.anthropic.com`, where the answer is a network, VPN, DNS or proxy problem.
 *     A user seeing "is the server running?" about Anthropic's API reasonably concludes
 *     the harness is broken rather than their connection.
 *
 * Pure functions over an error object, so the classification is unit-testable without a
 * network.
 */

import { ProviderError } from './types.ts';

export type TransportKind = 'dns' | 'tls' | 'timeout' | 'refused' | 'reset' | 'unknown';

export interface TransportDiagnosis {
  /** The provider-facing message, with the buried cause appended. */
  message: string;
  /** Actionable next step, tailored to the tier and the failure class. */
  hint: string;
  kind: TransportKind;
  /** The most specific code found in the cause chain, if any. */
  code?: string;
  /** Whether retrying could plausibly help. Unknown failures are treated as retryable. */
  retryable: boolean;
}

const CODES: Record<TransportKind, Set<string>> = {
  dns: new Set(['ENOTFOUND', 'EAI_AGAIN', 'EAI_NODATA']),
  tls: new Set([
    'UNABLE_TO_VERIFY_LEAF_SIGNATURE',
    'SELF_SIGNED_CERT_IN_CHAIN',
    'DEPTH_ZERO_SELF_SIGNED_CERT',
    'CERT_HAS_EXPIRED',
    'ERR_TLS_CERT_ALTNAME_INVALID',
    'UNABLE_TO_GET_ISSUER_CERT_LOCALLY',
  ]),
  timeout: new Set(['ETIMEDOUT', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT', 'ABORT_ERR']),
  refused: new Set(['ECONNREFUSED']),
  reset: new Set(['ECONNRESET', 'EPIPE', 'UND_ERR_SOCKET', 'ERR_STREAM_PREMATURE_CLOSE']),
  unknown: new Set(),
};

/**
 * Walk `err.cause` and collect the code/message at each level, deepest last.
 * Depth-limited because a cause chain can be circular in badly behaved libraries.
 */
export function causeChain(err: unknown): Array<{ code?: string; message: string }> {
  const out: Array<{ code?: string; message: string }> = [];
  let current: unknown = err;
  for (let depth = 0; depth < 6 && current !== null && current !== undefined; depth++) {
    const record = current as { code?: unknown; message?: unknown; cause?: unknown };
    const entry: { code?: string; message: string } = {
      message: typeof record.message === 'string' ? record.message : String(current),
    };
    if (typeof record.code === 'string') entry.code = record.code;
    out.push(entry);
    current = record.cause;
  }
  return out;
}

function classify(chain: Array<{ code?: string; message: string }>): { kind: TransportKind; code?: string } {
  // Deepest-first: the most specific cause is at the bottom of the chain.
  for (const entry of [...chain].reverse()) {
    if (!entry.code) continue;
    for (const kind of ['tls', 'dns', 'timeout', 'refused', 'reset'] as TransportKind[]) {
      if (CODES[kind].has(entry.code)) return { kind, code: entry.code };
    }
  }
  // Fall back to message sniffing for runtimes that do not set a code.
  const text = chain.map((c) => c.message).join(' ');
  if (/certificate|self.signed|tls|ssl/i.test(text)) return { kind: 'tls' };
  if (/getaddrinfo|dns/i.test(text)) return { kind: 'dns' };
  if (/timed? ?out|timeout/i.test(text)) return { kind: 'timeout' };
  if (/abort/i.test(text)) return { kind: 'timeout' };
  return { kind: 'unknown' };
}

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

export interface DiagnoseInput {
  providerId: string;
  baseUrl: string;
  kind: 'cloud' | 'local';
  timeoutMs?: number;
  status?: number;
}

export function diagnoseTransportError(err: unknown, input: DiagnoseInput): TransportDiagnosis {
  const chain = causeChain(err);
  const { kind, code } = classify(chain);
  const host = hostOf(input.baseUrl);
  const top = chain[0]?.message ?? 'request failed';
  // The buried cause is the whole point: surface it rather than the word "fetch failed".
  const deepest = chain[chain.length - 1];
  const detail =
    deepest && deepest !== chain[0] && deepest.message !== top
      ? `${deepest.code ? `${deepest.code}: ` : ''}${deepest.message}`
      : undefined;
  const message = detail ? `${top} (${detail})` : top;

  let hint: string;
  if (input.kind === 'local') {
    hint =
      kind === 'refused'
        ? `Nothing is listening on ${input.baseUrl}. Start your local runtime (see docs/local-models.md).`
        : kind === 'timeout'
          ? `${input.baseUrl} did not respond in time. The model may still be loading; try again, or raise local.requestTimeoutMs.`
          : `Could not reach your local runtime at ${input.baseUrl}. Is it running? See docs/local-models.md.`;
  } else {
    switch (kind) {
      case 'dns':
        hint = `DNS lookup failed for ${host}. Check your network or VPN — this is a connectivity problem, not an API key problem.`;
        break;
      case 'tls':
        hint = `TLS verification failed for ${host}. A corporate proxy or TLS-inspecting firewall is the usual cause; point NODE_EXTRA_CA_CERTS at your proxy's CA bundle.`;
        break;
      case 'timeout':
        hint = `${host} did not respond within ${input.timeoutMs ? `${Math.round(input.timeoutMs / 1000)}s` : 'the timeout'}. Check your network, VPN or proxy.`;
        break;
      case 'refused':
      case 'reset':
      case 'unknown':
      default:
        hint =
          `Could not reach ${host}. Check your network connection, VPN or proxy settings — ` +
          `the request never reached the API.`;
        break;
    }
  }

  return {
    message,
    hint,
    kind,
    ...(code ? { code } : {}),
    // A refused connection will not fix itself; DNS, resets and timeouts often do.
    retryable: kind !== 'refused' && input.status !== 401 && input.status !== 403,
  };
}

/** Build a ProviderError from a transport failure, with a tier-appropriate hint. */
export function transportProviderError(err: unknown, input: DiagnoseInput): ProviderError {
  const diagnosis = diagnoseTransportError(err, input);
  return new ProviderError(input.providerId, diagnosis.message, {
    retryable: diagnosis.retryable,
    hint: diagnosis.hint,
  });
}
