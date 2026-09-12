/**
 * Clock helpers.
 *
 * Why this module exists: the daily cloud budget and the daily training budget
 * are *the user's* daily budgets, so their accounting day must be the user's
 * local day. Keying them off `toISOString().slice(0, 10)` (UTC) makes the budget
 * reset at, say, 17:00 local for someone in UTC-7 — which would let a user spend
 * twice the intended amount in a single working day and then lock them out
 * mid-afternoon.
 *
 * Episode *shard* filenames deliberately still use ISO/UTC dates: those exist for
 * sortable storage, not for daily accounting, and mixing the two concerns is how
 * the bug crept in originally.
 */

/** `YYYY-MM-DD` in local time. Safe to compare lexicographically. */
export function localDayKey(date: Date = new Date()): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

/** Local hour, 0-23. Used by the training window gate. */
export function localHour(date: Date = new Date()): number {
  return date.getHours();
}

/** Subtract `days` from `key` (a local day key), returned in the same form. */
export function shiftDayKey(key: string, days: number): string {
  const [y, m, d] = key.split('-').map(Number) as [number, number, number];
  const date = new Date(y, (m ?? 1) - 1, d ?? 1);
  date.setDate(date.getDate() + days);
  return localDayKey(date);
}
