/**
 * Compaction: the transcript must fit the budget without ever losing the task.
 *
 * The properties pinned here are the ones whose silent regression would be
 * worst: the system prompt and first user message survive every compaction,
 * tool-call pairing is never broken, recent messages are untouched, and a
 * transcript that already fits is returned identically (no gratuitous churn).
 */

import assert from 'node:assert/strict';
import { describe, test } from 'node:test';

import { compactMessages, estimateTokens } from '../src/agent/compact.ts';
import type { Message } from '../src/providers/types.ts';

function toolRound(id: string, outputChars: number): Message[] {
  return [
    {
      role: 'assistant',
      content: '',
      toolCalls: [{ id, name: 'read_file', args: { path: `src/${id}.py` } }],
    },
    { role: 'tool', name: 'read_file', toolCallId: id, content: `[ok] read src/${id}.py\n\n${'x'.repeat(outputChars)}` },
  ];
}

function transcript(rounds: number, outputChars = 6000): Message[] {
  const messages: Message[] = [
    { role: 'system', content: 'You are a coding agent.' },
    { role: 'user', content: 'Fix the flux capacitor in engine.py' },
  ];
  for (let i = 0; i < rounds; i++) messages.push(...toolRound(`t${i}`, outputChars));
  messages.push({ role: 'assistant', content: 'Done.' });
  return messages;
}

describe('compactMessages', () => {
  test('returns the input untouched when it already fits', () => {
    const messages = transcript(2, 100);
    const result = compactMessages(messages, { budgetTokens: 100_000 });
    assert.equal(result.changed, false);
    assert.equal(result.messages, messages);
  });

  test('elides old tool outputs first and gets under budget', () => {
    const messages = transcript(20);
    const before = estimateTokens(messages);
    const result = compactMessages(messages, { budgetTokens: Math.round(before / 2) });
    assert.equal(result.changed, true);
    assert.ok(result.elided > 0);
    assert.ok(result.tokens <= Math.round(before / 2));
  });

  test('never loses the system prompt or the task message', () => {
    const messages = transcript(30);
    const result = compactMessages(messages, { budgetTokens: 3_000 });
    const out = result.messages;
    assert.equal(out[0]?.role, 'system');
    const task = out.find((m) => m.role === 'user' && m.content.includes('flux capacitor'));
    assert.ok(task, 'the original task statement must survive compaction');
  });

  test('keeps the most recent messages verbatim', () => {
    const messages = transcript(30);
    const lastTen = messages.slice(-10);
    const result = compactMessages(messages, { budgetTokens: 5_000, keepRecent: 10 });
    assert.deepEqual(result.messages.slice(-10), lastTen);
  });

  test('never leaves a tool result without its assistant tool call', () => {
    const messages = transcript(30);
    const result = compactMessages(messages, { budgetTokens: 3_000 });
    const out = result.messages;
    const callIds = new Set(out.flatMap((m) => (m.toolCalls ?? []).map((c) => c.id)));
    for (const m of out) {
      if (m.role === 'tool' && m.toolCallId) {
        assert.ok(callIds.has(m.toolCallId), `orphaned tool result ${m.toolCallId}`);
      }
    }
  });

  test('announces dropped history with a synthetic user message', () => {
    const messages = transcript(30);
    const result = compactMessages(messages, { budgetTokens: 3_000 });
    if (result.dropped > 0) {
      const note = result.messages.find((m) => m.content.includes('<compaction>'));
      assert.ok(note, 'a compaction note must be present when messages are dropped');
    }
  });

  test('elided outputs keep their header line so the model knows what ran', () => {
    const messages = transcript(20);
    const before = estimateTokens(messages);
    const result = compactMessages(messages, { budgetTokens: Math.round(before / 2) });
    const elidedMsg = result.messages.find((m) => m.role === 'tool' && m.content.includes('elided'));
    assert.ok(elidedMsg);
    assert.match(elidedMsg.content.split('\n')[0] ?? '', /\[ok\] read src\//);
  });
});
