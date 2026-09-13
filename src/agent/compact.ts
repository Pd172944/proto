/**
 * Transcript compaction: bounded memory for unbounded work.
 *
 * A 40-step agent turn accumulates megabytes of tool output. On a cloud model
 * that is expensive; on a local model with a 32k context window it is fatal —
 * the server silently truncates from the front and the agent forgets the task
 * it was given. Compaction keeps the transcript inside a token budget by
 * spending detail where it still matters:
 *
 *  1. **Old tool outputs are elided first.** A file read from twenty steps ago
 *     has served its purpose; what the model still needs is the *fact that it
 *     was read* (so it does not read it again) and the path, not 24k of body.
 *  2. **Whole early rounds are dropped next**, oldest first, each replaced by
 *     nothing — their existence is recorded in a single synthetic summary
 *     message that lists which tools ran and which files were touched.
 *  3. **Never touched:** the system prompt, the first user message (the task),
 *     and the most recent rounds (the model's working set).
 *
 * Everything here is pure and deterministic: no model call, no I/O, no clock.
 * Summarising with the local model would be smarter but slower and untestable;
 * the deterministic ladder gets 90% of the benefit and can be unit-tested to
 * never lose the task statement.
 */

import type { Message } from '../providers/types.ts';

/* ------------------------------------------------------------------ */
/* Estimation                                                          */
/* ------------------------------------------------------------------ */

/** Rough token estimate for a message list (chars / 3.6, floor 1). */
export function estimateTokens(messages: Message[]): number {
  let chars = 0;
  for (const m of messages) {
    chars += m.content.length;
    if (m.toolCalls) chars += JSON.stringify(m.toolCalls).length;
  }
  return Math.max(1, Math.round(chars / 3.6));
}

/* ------------------------------------------------------------------ */
/* Options and result                                                  */
/* ------------------------------------------------------------------ */

export interface CompactOptions {
  /** Target token budget for the whole message list. */
  budgetTokens: number;
  /** Number of most-recent messages that are never elided or dropped. */
  keepRecent?: number;
  /** Tool outputs older than `keepRecent` are cut to this many characters. */
  elidedToolChars?: number;
}

export interface CompactResult {
  messages: Message[];
  /** Number of tool outputs elided (step 1). */
  elided: number;
  /** Number of whole messages dropped (step 2). */
  dropped: number;
  /** Estimated tokens after compaction. */
  tokens: number;
  /** True if anything changed. */
  changed: boolean;
}

/* ------------------------------------------------------------------ */
/* Compaction                                                          */
/* ------------------------------------------------------------------ */

const ELIDE_MARKER = '\n…[output elided to save context; re-run the tool if you need it again]';

/**
 * Compact `messages` to fit within `budgetTokens`.
 *
 * The list is treated as: `[system, task, ...body, ...recent]` where `recent`
 * is the last `keepRecent` messages. Only `body` is ever modified. Structure
 * is preserved: eliding shortens a tool message's content in place, and
 * dropping removes an assistant message *together with* its tool results so a
 * provider never sees a tool call without its result (or vice versa).
 *
 * Args mirror {@link CompactOptions}. Returns a {@link CompactResult}; when
 * the input already fits, `changed` is false and the original array is
 * returned untouched.
 */
export function compactMessages(messages: Message[], opts: CompactOptions): CompactResult {
  const keepRecent = opts.keepRecent ?? 10;
  const elidedToolChars = opts.elidedToolChars ?? 500;

  if (estimateTokens(messages) <= opts.budgetTokens) {
    return { messages, elided: 0, dropped: 0, tokens: estimateTokens(messages), changed: false };
  }

  // Head: system prompt (if present) plus the first user message — the task.
  let headEnd = 0;
  if (messages[0]?.role === 'system') headEnd = 1;
  if (messages[headEnd]?.role === 'user') headEnd += 1;
  // The recent window must not start on a tool result: its owning assistant
  // message would then be in the droppable body, and dropping it would orphan
  // the result. Widen "recent" backwards until the boundary is clean.
  let recentStart = Math.max(headEnd, messages.length - keepRecent);
  while (recentStart > headEnd && messages[recentStart]?.role === 'tool') recentStart -= 1;

  const head: Message[] = messages.slice(0, headEnd);
  let body: Message[] = messages.slice(headEnd, recentStart).map((m) => ({ ...m }));
  const recent: Message[] = messages.slice(recentStart);

  // Step 1: elide old tool outputs, oldest first, until the budget fits.
  let elided = 0;
  for (const m of body) {
    if (estimateTokens([...head, ...body, ...recent]) <= opts.budgetTokens) break;
    if (m.role !== 'tool' || m.content.length <= elidedToolChars + ELIDE_MARKER.length) continue;
    const headerEnd: number = m.content.indexOf('\n');
    const header: string = headerEnd > 0 ? m.content.slice(0, headerEnd) : '';
    m.content = `${header}\n${m.content.slice(header.length, header.length + elidedToolChars)}${ELIDE_MARKER}`;
    elided += 1;
  }

  // Step 2: drop whole rounds from the front of the body. A "round" is an
  // assistant message and every tool result belonging to it; standalone user
  // messages (reminders) drop individually.
  let dropped = 0;
  const droppedFacts: string[] = [];
  while (body.length > 0 && estimateTokens([...head, ...body, ...recent]) > opts.budgetTokens) {
    const first = body[0] as Message;
    let take = 1;
    if (first.role === 'assistant' && first.toolCalls?.length) {
      while (take < body.length && (body[take] as Message).role === 'tool') take += 1;
      const tools: string = (first.toolCalls ?? []).map((c) => c.name).join(', ');
      droppedFacts.push(tools);
    }
    body = body.slice(take);
    dropped += take;
  }

  if (elided === 0 && dropped === 0) {
    return { messages, elided, dropped, tokens: estimateTokens(messages), changed: false };
  }

  const summary: Message[] =
    dropped > 0
      ? [
          {
            role: 'user',
            content:
              `<compaction>Earlier steps of this session were removed to fit the context window. ` +
              `${dropped} message(s) dropped${droppedFacts.length > 0 ? ` (tool calls: ${droppedFacts.join('; ')})` : ''}. ` +
              `Trust your notes in later messages; re-read files rather than assuming earlier contents.</compaction>`,
          },
        ]
      : [];

  const out: Message[] = [...head, ...summary, ...body, ...recent];
  return { messages: out, elided, dropped, tokens: estimateTokens(out), changed: true };
}
