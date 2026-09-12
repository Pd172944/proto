/**
 * Redaction.
 *
 * This runs *before* anything is written to disk. That ordering is important:
 * a design where redaction happens at upload time means the plaintext secrets
 * are sitting in `var/episodes/*.jsonl` on the user's disk, and any bug in the
 * upload path leaks them. Redacting at rest means the worst case is a less
 * useful training record, not a leaked key.
 *
 * The rules are layered deliberately:
 *   1. Structural secrets (private keys, JWTs, `KEY=value` assignments) where
 *      the shape alone is conclusive.
 *   2. Known credential formats (provider prefixes like `sk-`, `ghp_`, `AKIA`).
 *   3. High-entropy strings, which catch tokens from services we have never
 *      heard of. This is the noisiest rule and the one that most often removes
 *      legitimate content (base64 fixtures, content hashes), so it is reported
 *      separately in the redaction audit trail.
 *   4. Personal identifiers (emails, IPs, home directories) — not secrets, but
 *      they are the connective tissue of de-anonymisation.
 *
 * We never attempt "smart" partial redaction such as keeping the first four
 * characters of a key. Partial secrets are still secrets, and a prefix plus a
 * leaked service narrows an attacker's search enormously.
 */

import { shannonEntropy } from '../util/text.ts';

export interface RedactionRule {
  name: string;
  re: RegExp;
  replace: string | ((match: string, ...groups: string[]) => string);
}

const PLACEHOLDER = (kind: string): string => `[REDACTED:${kind}]`;

/** Keep the key name, remove the value: `API_KEY=abc` -> `API_KEY=[REDACTED:...]`. */
const SECRET_ASSIGNMENT =
  /((?:[A-Za-z0-9_.-]*(?:api[_-]?key|secret|token|password|passwd|pwd|credential|private[_-]?key|access[_-]?key|auth)[A-Za-z0-9_.-]*)\s*[:=]\s*)(["']?)([^\s"',;]{6,})\2/gi;

export const REDACTION_RULES: RedactionRule[] = [
  {
    name: 'private-key-block',
    re: /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
    replace: PLACEHOLDER('private-key'),
  },
  {
    name: 'openai-key',
    re: /\bsk-(?:ant-|proj-|svcacct-)?[A-Za-z0-9_-]{16,}\b/g,
    replace: PLACEHOLDER('api-key'),
  },
  { name: 'anthropic-key', re: /\bsk-ant-[A-Za-z0-9_-]{20,}\b/g, replace: PLACEHOLDER('api-key') },
  { name: 'github-token', re: /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b/g, replace: PLACEHOLDER('github-token') },
  { name: 'github-pat', re: /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g, replace: PLACEHOLDER('github-token') },
  { name: 'gitlab-token', re: /\bglpat-[A-Za-z0-9_-]{16,}\b/g, replace: PLACEHOLDER('gitlab-token') },
  { name: 'aws-access-key', re: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g, replace: PLACEHOLDER('aws-key') },
  { name: 'google-api-key', re: /\bAIza[0-9A-Za-z_-]{35}\b/g, replace: PLACEHOLDER('google-key') },
  { name: 'slack-token', re: /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/g, replace: PLACEHOLDER('slack-token') },
  { name: 'huggingface-token', re: /\bhf_[A-Za-z0-9]{20,}\b/g, replace: PLACEHOLDER('hf-token') },
  { name: 'stripe-key', re: /\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b/g, replace: PLACEHOLDER('stripe-key') },
  {
    name: 'jwt',
    re: /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g,
    replace: PLACEHOLDER('jwt'),
  },
  {
    name: 'authorization-header',
    re: /\b(Authorization\s*:\s*)(Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=-]{8,}/gi,
    replace: (_m, p1, p2) => `${p1}${p2} ${PLACEHOLDER('bearer')}`,
  },
  {
    name: 'url-credentials',
    re: /\b([a-z][a-z0-9+.-]*:\/\/)([^/\s:@]{1,64}):([^/\s@]{1,128})@/gi,
    replace: (_m, scheme) => `${scheme}${PLACEHOLDER('url-credentials')}@`,
  },
  {
    name: 'secret-assignment',
    re: SECRET_ASSIGNMENT,
    replace: (_m, prefix, quote) => `${prefix}${quote}${PLACEHOLDER('secret-value')}${quote}`,
  },
  {
    name: 'dotenv-known-secret',
    // Targeted at well-known credential-bearing env names rather than every
    // SCREAMING_CASE assignment, which would shred ordinary config code.
    re: /^((?:AWS|AZURE|GCP|GOOGLE|OPENAI|ANTHROPIC|STRIPE|TWILIO|SENDGRID|SLACK|GITHUB|GITLAB|HF|HUGGINGFACE|DATABASE_URL|DB|REDIS|MONGO|SMTP|JWT|SESSION|OAUTH|CLIENT)[A-Z0-9_]*)\s*=\s*(.{8,})$/gm,
    replace: (_m, name) => `${name}=${PLACEHOLDER('env-value')}`,
  },
  {
    name: 'email',
    re: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
    replace: PLACEHOLDER('email'),
  },
  {
    // Only RFC1918/loopback-adjacent and public IPv4; ports are preserved.
    name: 'ipv4',
    re: /\b(?:\d{1,3}\.){3}\d{1,3}\b/g,
    replace: PLACEHOLDER('ip'),
  },
  {
    name: 'home-path',
    re: /(?:\/Users\/|\/home\/)([A-Za-z0-9._-]+)(\/|$)/g,
    replace: (_m, _user, trail) => `[REDACTED:home]${trail}`,
  },
  {
    name: 'macos-udid',
    re: /\b[0-9A-F]{8}-[0-9A-F]{16}\b/g,
    replace: PLACEHOLDER('device-id'),
  },
];

/** Entropy-based detection: long, dense, non-natural-language tokens. */
const ENTROPY_CANDIDATE = /\b[A-Za-z0-9+/=_-]{40,}\b/g;

export interface RedactResult {
  text: string;
  counts: Record<string, number>;
  charsRemoved: number;
  /** Set when the input was truncated by a cap rather than by redaction. */
  truncated?: boolean;
}

export interface RedactOptions {
  /** Extra user-supplied regexes, applied after the built-ins. */
  extraPatterns?: string[];
  /** Run the entropy heuristic. On by default; documented as the noisiest rule. */
  entropyHeuristic?: boolean;
  /** Never return more than this many characters. */
  maxChars?: number;
}

export function redact(input: string, opts: RedactOptions = {}): RedactResult {
  const counts: Record<string, number> = {};
  let text = input;
  const originalLength = input.length;

  const bump = (name: string, n = 1): void => {
    counts[name] = (counts[name] ?? 0) + n;
  };

  for (const rule of REDACTION_RULES) {
    // Rules are declared with the `g` flag; reset lastIndex defensively because
    // module-level RegExp objects are shared across calls.
    rule.re.lastIndex = 0;
    text = text.replace(rule.re, (...args: unknown[]) => {
      bump(rule.name);
      if (typeof rule.replace === 'string') return rule.replace;
      const match = String(args[0]);
      // String.replace always appends (offset, wholeString) after the capture
      // groups, so dropping the last two entries yields exactly the groups.
      const groups = args.slice(1, -2).map((g) => (g === undefined ? '' : String(g)));
      return rule.replace(match, ...groups);
    });
  }

  if (opts.extraPatterns?.length) {
    for (const source of opts.extraPatterns) {
      let re: RegExp;
      try {
        re = new RegExp(source, 'g');
      } catch {
        continue; // invalid user regex; surfaced by config validation instead
      }
      text = text.replace(re, () => {
        bump('custom-pattern');
        return PLACEHOLDER('custom');
      });
    }
  }

  if (opts.entropyHeuristic !== false) {
    ENTROPY_CANDIDATE.lastIndex = 0;
    text = text.replace(ENTROPY_CANDIDATE, (match) => {
      // Natural-language words with hyphens/underscores can be long but have low
      // entropy; require both high entropy and mixed character classes.
      const classes = [/[a-z]/, /[A-Z]/, /\d/].filter((r) => r.test(match)).length;
      const entropy = shannonEntropy(match);
      if (entropy >= 3.6 && classes >= 2) {
        bump('high-entropy');
        return PLACEHOLDER('high-entropy');
      }
      return match;
    });
  }

  let truncated = false;
  const maxChars = opts.maxChars;
  if (maxChars !== undefined && text.length > maxChars) {
    text = text.slice(0, maxChars) + `\n…[truncated ${text.length - maxChars} chars by memory cap]`;
    truncated = true;
  }

  return {
    text,
    counts,
    charsRemoved: Math.max(0, originalLength - text.length),
    ...(truncated ? { truncated } : {}),
  };
}

/** Convenience for metadata fields where only a fingerprint is needed. */
export function redactForHash(input: string): string {
  return redact(input, { entropyHeuristic: false }).text;
}

export function totalRedactions(counts: Record<string, number>): number {
  return Object.values(counts).reduce((a, b) => a + b, 0);
}
