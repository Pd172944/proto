/**
 * Text helpers: token estimation, hashing, code-fence extraction, entropy.
 *
 * No tokenizer dependency is used on purpose. We only ever need *approximate*
 * token counts for (a) context-window guards and (b) cost estimates, and a
 * heuristic is both cheaper and more robust than shipping a BPE table for every
 * model the user might pick. Estimates are labelled as estimates everywhere
 * they surface to the user.
 */

import { createHash } from 'node:crypto';

/**
 * Approximate token count.
 *
 * Heuristic calibrated against cl100k-family tokenizers on mixed prose+code:
 *  - code is denser in punctuation/long identifiers, so it tokenizes worse
 *    (~3.2 chars/token) than prose (~4.0 chars/token).
 *  - whitespace runs collapse to roughly one token per 4 spaces.
 * The error band is roughly +/-15%, which is acceptable for routing decisions.
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  const chars = text.length;
  if (chars < 16) return Math.max(1, Math.ceil(chars / 3));
  const codeLike = countMatches(text, /[{}()[\];=<>]|=>|\bdef\b|\bfunc\b|\breturn\b/g);
  const codeRatio = Math.min(1, codeLike / Math.max(1, chars / 40));
  const charsPerToken = 4.0 - 0.8 * codeRatio;
  return Math.max(1, Math.round(chars / charsPerToken));
}

export function countMatches(text: string, re: RegExp): number {
  const m = text.match(re);
  return m ? m.length : 0;
}

export function sha256(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

export function sha256Short(text: string, len = 16): string {
  return sha256(text).slice(0, len);
}

/**
 * Shannon entropy in bits/char. Used by the redactor to flag probable secrets
 * (API keys, tokens, base64 blobs) that pattern matching alone would miss.
 */
export function shannonEntropy(text: string): number {
  if (!text) return 0;
  const counts = new Map<string, number>();
  for (const ch of text) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  let h = 0;
  for (const c of counts.values()) {
    const p = c / text.length;
    h -= p * Math.log2(p);
  }
  return h;
}

export function truncate(text: string, max: number, note = '…[truncated]'): string {
  if (text.length <= max) return text;
  return text.slice(0, Math.max(0, max - note.length)) + note;
}

/** Keep head and tail; middle content is usually the least informative. */
export function truncateMiddle(text: string, max: number): string {
  if (text.length <= max) return text;
  const head = Math.floor(max * 0.6);
  const tail = Math.max(0, max - head - 20);
  return `${text.slice(0, head)}\n…[${text.length - head - tail} chars elided]…\n${text.slice(text.length - tail)}`;
}

export interface CodeBlock {
  lang: string;
  code: string;
  /** File path if the fence was annotated, e.g. ```ts title=src/a.ts */
  path?: string;
}

const FENCE_RE = /```([^\n`]*)\n([\s\S]*?)```/g;

/**
 * Extract fenced code blocks. Models are told to emit a specific JSON envelope,
 * but they frequently fall back to markdown fences, so the verifier can still
 * salvage a usable candidate.
 */
export function extractCodeBlocks(text: string): CodeBlock[] {
  const blocks: CodeBlock[] = [];
  FENCE_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = FENCE_RE.exec(text)) !== null) {
    const info = (m[1] ?? '').trim();
    const code = m[2] ?? '';
    const lang = (info.split(/[\s:]/)[0] ?? '').toLowerCase();
    const pathMatch = info.match(/(?:\btitle=|\bpath=|\bfile=)(\S+)/);
    const block: CodeBlock = { lang, code };
    if (pathMatch?.[1]) block.path = pathMatch[1].replace(/^["']|["']$/g, '');
    blocks.push(block);
  }
  return blocks;
}

/** Extract the first balanced JSON object/array found in a text blob. */
export function extractJson(text: string): unknown {
  const trimmed = text.trim();
  const candidates: string[] = [];
  const fence = trimmed.match(/```(?:json)?\s*\n([\s\S]*?)```/);
  if (fence?.[1]) candidates.push(fence[1]);
  candidates.push(trimmed);
  const firstBrace = trimmed.indexOf('{');
  const firstBracket = trimmed.indexOf('[');
  const start =
    firstBrace === -1 ? firstBracket : firstBracket === -1 ? firstBrace : Math.min(firstBrace, firstBracket);
  if (start >= 0) candidates.push(balancedSlice(trimmed, start));

  for (const c of candidates) {
    if (!c) continue;
    try {
      return JSON.parse(c);
    } catch {
      /* try next */
    }
  }
  return undefined;
}

function balancedSlice(text: string, start: number): string {
  const open = text[start];
  const close = open === '{' ? '}' : ']';
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i] as string;
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === open) depth++;
    else if (ch === close) {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return text.slice(start);
}

/** Single-line summary for logs and tables. */
export function oneLine(text: string, max = 120): string {
  const s = text.replace(/\s+/g, ' ').trim();
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
}

export function indent(text: string, prefix: string): string {
  return text
    .split('\n')
    .map((l) => prefix + l)
    .join('\n');
}

/** Deterministic JSON — stable key order, so hashes are reproducible. */
export function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value) ?? 'null';
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(obj[k])}`).join(',')}}`;
}
