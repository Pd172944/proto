/**
 * Transport-failure diagnosis tests.
 *
 * The regression these guard is specific and was reported by a real user: a transient
 * DNS failure during a turn produced
 *
 *   ✗ fetch failed (Could not reach https://api.anthropic.com/v1. Is the server running?)
 *
 * which invites the conclusion that the *harness* is broken, when the actual cause was
 * `ENOTFOUND` on the user's network — hidden inside `err.cause` and discarded.
 *
 * So the central assertion is negative: a cloud connectivity failure must never suggest
 * that a server needs starting.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { causeChain, diagnoseTransportError, transportProviderError } from '../src/providers/errors.ts';

function transportError(code: string, message: string, extra: Record<string, unknown> = {}): Error {
  // Shaped exactly like Node's fetch failure: a useless top-level message and the real
  // reason one level down.
  return Object.assign(new Error('fetch failed'), {
    cause: Object.assign(new Error(message), { code, ...extra }),
  });
}

const CLOUD = { providerId: 'anthropic', baseUrl: 'https://api.anthropic.com/v1', kind: 'cloud' as const, timeoutMs: 180_000 };
const LOCAL = { providerId: 'local', baseUrl: 'http://127.0.0.1:11434', kind: 'local' as const, timeoutMs: 60_000 };

describe('causeChain', () => {
  it('walks nested causes deepest-last', () => {
    const err = transportError('ENOTFOUND', 'getaddrinfo ENOTFOUND api.anthropic.com');
    const chain = causeChain(err);
    assert.equal(chain[0]?.message, 'fetch failed');
    assert.equal(chain[1]?.code, 'ENOTFOUND');
    assert.match(chain[1]?.message ?? '', /getaddrinfo/);
  });

  it('survives a circular cause chain', () => {
    const a = new Error('a') as Error & { cause?: unknown };
    const b = new Error('b') as Error & { cause?: unknown };
    a.cause = b;
    b.cause = a;
    const chain = causeChain(a);
    assert.ok(chain.length <= 6, 'must be depth-limited rather than looping forever');
    assert.ok(chain.length >= 2);
  });

  it('handles a non-Error throw', () => {
    assert.equal(causeChain('just a string')[0]?.message, 'just a string');
    assert.equal(causeChain(null).length, 0);
  });
});

describe('cloud transport failures', () => {
  it('explains a DNS failure and never blames a local server', () => {
    const d = diagnoseTransportError(transportError('ENOTFOUND', 'getaddrinfo ENOTFOUND api.anthropic.com'), CLOUD);
    assert.equal(d.kind, 'dns');
    assert.match(d.message, /ENOTFOUND/, 'the buried cause must be surfaced');
    assert.match(d.hint, /network or VPN/i);
    // The regression guard: this exact advice was wrong and actively misleading.
    assert.doesNotMatch(d.hint, /server running/i);
    assert.doesNotMatch(d.hint, /start your local runtime/i);
  });

  it('points at a proxy for TLS verification failures', () => {
    const d = diagnoseTransportError(
      transportError('UNABLE_TO_VERIFY_LEAF_SIGNATURE', 'unable to verify the first certificate'),
      CLOUD,
    );
    assert.equal(d.kind, 'tls');
    assert.match(d.hint, /proxy|firewall/i);
    assert.match(d.hint, /NODE_EXTRA_CA_CERTS/);
  });

  it('reports the timeout it actually waited', () => {
    const d = diagnoseTransportError(transportError('UND_ERR_CONNECT_TIMEOUT', 'Connect Timeout Error'), CLOUD);
    assert.equal(d.kind, 'timeout');
    assert.match(d.hint, /180s/);
  });

  it('treats a dropped connection as a connectivity problem', () => {
    const d = diagnoseTransportError(transportError('ECONNRESET', 'socket hang up'), CLOUD);
    assert.equal(d.kind, 'reset');
    assert.match(d.hint, /network|VPN|proxy/i);
    assert.equal(d.retryable, true, 'a reset is worth retrying');
  });

  it('gives a usable hint even for an unrecognised code', () => {
    const d = diagnoseTransportError(transportError('ESOMETHINGNEW', 'weird'), CLOUD);
    assert.equal(d.kind, 'unknown');
    assert.match(d.hint, /api\.anthropic\.com/);
    assert.doesNotMatch(d.hint, /server running/i);
  });
});

describe('local transport failures', () => {
  it('says to start the local runtime when nothing is listening', () => {
    const d = diagnoseTransportError(transportError('ECONNREFUSED', 'connect ECONNREFUSED 127.0.0.1:11434'), LOCAL);
    assert.equal(d.kind, 'refused');
    assert.match(d.hint, /Start your local runtime/);
    assert.match(d.hint, /127\.0\.0\.1:11434/);
    assert.equal(d.retryable, false, 'a refused connection will not fix itself');
  });

  it('mentions model loading on a local timeout', () => {
    const d = diagnoseTransportError(transportError('ETIMEDOUT', 'timed out'), LOCAL);
    assert.equal(d.kind, 'timeout');
    assert.match(d.hint, /loading|try again|local\.requestTimeoutMs/i);
  });
});

describe('classification edge cases', () => {
  it('prefers the deepest, most specific code', () => {
    // A generic wrapper hiding a TLS error must classify as TLS, not unknown.
    const wrapped = Object.assign(new Error('fetch failed'), {
      cause: Object.assign(new Error('outer'), {
        code: 'UND_ERR_SOCKET',
        cause: Object.assign(new Error('bad cert'), { code: 'CERT_HAS_EXPIRED' }),
      }),
    });
    assert.equal(diagnoseTransportError(wrapped, CLOUD).kind, 'tls');
  });

  it('falls back to message sniffing when no code is set', () => {
    const err = Object.assign(new Error('fetch failed'), { cause: new Error('getaddrinfo ENOTFOUND example.com') });
    assert.equal(diagnoseTransportError(err, CLOUD).kind, 'dns');
  });

  it('does not treat an auth failure as retryable connectivity', () => {
    const d = diagnoseTransportError(transportError('ECONNRESET', 'reset'), { ...CLOUD, status: 401 });
    assert.equal(d.retryable, false);
  });

  it('builds a ProviderError with the diagnosis attached', () => {
    const err = transportProviderError(transportError('ENOTFOUND', 'getaddrinfo ENOTFOUND api.anthropic.com'), CLOUD);
    assert.equal(err.name, 'ProviderError');
    assert.equal(err.providerId, 'anthropic');
    assert.match(err.message, /ENOTFOUND/);
    assert.ok(err.hint && /network or VPN/i.test(err.hint));
    assert.equal(err.retryable, true);
  });

  it('handles a baseUrl that is not a parseable URL', () => {
    const d = diagnoseTransportError(transportError('ENOTFOUND', 'nope'), { ...CLOUD, baseUrl: 'not a url' });
    assert.match(d.hint, /not a url/);
  });
});
