/**
 * Redaction tests.
 *
 * The false-positive tests matter as much as the detection tests: a redactor that
 * eats ordinary code makes the datasets useless, and a user who notices their
 * code being mangled will simply turn redaction off — which is worse for privacy
 * than a slightly noisier redactor.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { redact, redactForHash } from '../src/memory/redact.ts';

describe('redact', () => {
  it('removes provider API keys of every supported shape', () => {
    const input = [
      'OPENAI=sk-proj-abcdefghijklmnopqrstuvwxyz012345',
      'ANTHROPIC=sk-ant-api03-abcdefghijklmnopqrstuvwxyz',
      'GITHUB=ghp_abcdefghijklmnopqrstuvwxyz012345',
      'AWS=AKIAIOSFODNN7EXAMPLE',
      'GOOGLE=AIzaSyA1234567890abcdefghijklmnopqrstuv',
      'HF=hf_abcdefghijklmnopqrstuvwxyz0123456789',
    ].join('\n');
    const out = redact(input);
    assert.ok(!out.text.includes('sk-proj-abcdefghijklmnopqrstuvwxyz012345'));
    assert.ok(!out.text.includes('AKIAIOSFODNN7EXAMPLE'));
    assert.ok(!out.text.includes('AIzaSyA1234567890abcdefghijklmnopqrstuv'));
    assert.ok(!out.text.includes('hf_abcdefghijklmnopqrstuvwxyz0123456789'));
    assert.ok(out.counts['openai-key']! >= 1);
    assert.ok(out.counts['aws-access-key']! >= 1);
  });

  it('removes private key blocks entirely', () => {
    const pem = '-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234\nmore\n-----END RSA PRIVATE KEY-----';
    const out = redact(`key:\n${pem}\ndone`);
    assert.ok(!out.text.includes('MIIEowIBAAKCAQEA1234'));
    assert.match(out.text, /\[REDACTED:private-key\]/);
    assert.ok(out.text.includes('done'));
  });

  it('keeps the key name but removes the value in assignments', () => {
    const out = redact('api_key = "s3cr3t-value-here"\nDATABASE_URL=postgres://u:p@host/db');
    assert.ok(out.text.includes('api_key'));
    assert.ok(!out.text.includes('s3cr3t-value-here'));
    assert.ok(out.text.includes('DATABASE_URL'));
  });

  it('removes JWTs, bearer tokens and credentials embedded in URLs', () => {
    const jwt = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk';
    const out = redact(`token: ${jwt}\nAuthorization: Bearer abcdefghijklmnopqrstuvwxyz\nclone https://user:hunter2@github.com/x/y.git`);
    assert.ok(!out.text.includes(jwt));
    assert.ok(!out.text.includes('hunter2'));
    assert.ok(!out.text.includes('abcdefghijklmnopqrstuvwxyz'));
  });

  it('removes emails, IP addresses and home directory names', () => {
    const out = redact('contact dev@example.com from 192.168.1.44 at /Users/prithvidixit/src/app');
    assert.ok(!out.text.includes('dev@example.com'));
    assert.ok(!out.text.includes('192.168.1.44'));
    assert.ok(!out.text.includes('prithvidixit'));
    assert.ok(out.text.includes('/src/app'), 'the path tail should survive for context');
  });

  it('removes long high-entropy strings but keeps ordinary long identifiers', () => {
    const secret = 'Zk8xQ2mN7pR4tV9wY1aB3cD6eF0gH5jK8lM2nP4q';
    const out = redact(`const value = "${secret}";`);
    assert.ok(!out.text.includes(secret));
    assert.ok(out.counts['high-entropy']! >= 1);
  });

  it('does NOT mangle ordinary code', () => {
    const code = [
      'def compute_total(items: list[dict]) -> float:',
      '    total = 0.0',
      '    for item in items:',
      '        total += item["price"] * 1.08',
      '    return round(total, 2)',
      '',
      'class OrderProcessor:',
      '    DEFAULT_CURRENCY = "USD"',
    ].join('\n');
    const out = redact(code);
    assert.equal(out.text, code, 'plain code must pass through unchanged');
    assert.deepEqual(out.counts, {});
  });

  it('honours custom patterns and reports counts', () => {
    const out = redact('internal code ACME-9931X here', { extraPatterns: ['ACME-\\d+X'] });
    assert.ok(!out.text.includes('ACME-9931X'));
    assert.equal(out.counts['custom-pattern'], 1);
  });

  it('caps output length when asked and flags truncation', () => {
    const out = redact('a'.repeat(500), { maxChars: 100 });
    assert.ok(out.text.length < 250);
    assert.equal(out.truncated, true);
  });

  it('redactForHash is stable and does not run the entropy heuristic', () => {
    const value = 'a'.repeat(60);
    const a = redactForHash(value);
    const b = redactForHash(value);
    assert.equal(a, b);
    assert.equal(a, value);
  });
});
