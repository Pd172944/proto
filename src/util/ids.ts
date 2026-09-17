/**
 * Time-sortable, collision-resistant identifiers.
 *
 * We use a ULID-shaped scheme (48-bit timestamp + 80 bits of randomness,
 * Crockford base32) so that ids sort lexicographically by creation time. This
 * matters because session files are written and read back in id order
 * and we want deterministic, cheap ordering without reading timestamps.
 */

import { createHash, randomBytes } from 'node:crypto';

const CROCKFORD = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';

let lastTime = -1;
let lastRandom: number[] = [];

function encodeTime(ms: number, length: number): string {
  let out = '';
  let t = ms;
  for (let i = length - 1; i >= 0; i--) {
    out = (CROCKFORD[t % 32] as string) + out;
    t = Math.floor(t / 32);
  }
  return out;
}

function randomChars(count: number): number[] {
  const bytes = randomBytes(count);
  const out: number[] = [];
  for (let i = 0; i < count; i++) out.push((bytes[i] as number) % 32);
  return out;
}

/** Monotonic ULID: strictly increasing within a single process. */
export function ulid(now: number = Date.now()): string {
  if (now === lastTime) {
    // Increment previous randomness to guarantee monotonicity.
    let i = lastRandom.length - 1;
    while (i >= 0) {
      const v = (lastRandom[i] as number) + 1;
      if (v < 32) {
        lastRandom[i] = v;
        break;
      }
      lastRandom[i] = 0;
      i--;
    }
    if (i < 0) lastRandom = randomChars(16);
  } else {
    lastTime = now;
    lastRandom = randomChars(16);
  }
  return encodeTime(now, 10) + lastRandom.map((v) => CROCKFORD[v] as string).join('');
}

/** Short, human-friendly id used for jobs, attempts and run directories. */
export function shortId(prefix = ''): string {
  const r = randomBytes(4).toString('hex');
  const t = Date.now().toString(36).slice(-4);
  return `${prefix}${prefix ? '-' : ''}${t}${r}`;
}

/** Stable content-addressed id, for dedup (for example of edit candidates). */
export function contentId(content: string): string {
  return createHash('sha256').update(content, 'utf8').digest('hex').slice(0, 32);
}

/** Random hex string, used for rotating anonymisation salts. */
export function randomHex(bytes = 16): string {
  return randomBytes(bytes).toString('hex');
}
