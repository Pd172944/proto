/**
 * Tests for the plan/execute split.
 *
 * The interesting cases here are the ones that bit in practice. The pipeline was first
 * written to expect one edit per step, and the model — following a plan that named two
 * changes — returned the first and silently dropped the second. Verification could not
 * catch it, because the edit it did receive was perfectly well formed. That is the
 * failure this file exists to prevent from coming back.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { assembleCandidate, extractJson, parseEdits, parsePlan, executeStep, planTask } from '../src/harness/split.ts';
import type { ExecutedStep, PlanStep } from '../src/harness/split.ts';
import { MockProvider } from '../src/providers/mock.ts';
import { testConfig } from './helpers.ts';

/* ------------------------------------------------------------------ */
/* Parsing                                                             */
/* ------------------------------------------------------------------ */

describe('extractJson', () => {
  it('reads a bare object', () => {
    assert.deepEqual(extractJson('{"a":1}'), { a: 1 });
  });

  it('reads an object inside a fenced block', () => {
    assert.deepEqual(extractJson('```json\n{"a":1}\n```'), { a: 1 });
  });

  it('reads an object surrounded by prose', () => {
    assert.deepEqual(extractJson('Sure! Here you go:\n{"a":1}\nHope that helps.'), { a: 1 });
  });

  it('returns null rather than throwing on junk', () => {
    assert.equal(extractJson('not json at all'), null);
    assert.equal(extractJson(''), null);
    assert.equal(extractJson('{broken'), null);
  });
});

describe('parsePlan', () => {
  it('reads a plan with steps', () => {
    const plan = parsePlan('{"summary":"s","steps":[{"file":"a.py","intent":"rename x"}]}');
    assert.equal(plan?.summary, 's');
    assert.deepEqual(plan?.steps, [{ file: 'a.py', intent: 'rename x' }]);
  });

  it('reads an answer-only plan, which is how a question comes back', () => {
    const plan = parsePlan('{"summary":"s","steps":[],"answer":"42"}');
    assert.equal(plan?.answer, '42');
    assert.deepEqual(plan?.steps, []);
  });

  it('drops malformed steps instead of failing the whole plan', () => {
    const plan = parsePlan('{"summary":"s","steps":[{"file":"a.py","intent":"ok"},{"file":"","intent":"no"},{"nope":1}]}');
    assert.deepEqual(plan?.steps, [{ file: 'a.py', intent: 'ok' }]);
  });

  it('rejects a plan with neither steps nor an answer', () => {
    assert.equal(parsePlan('{"summary":"only a summary"}'), null);
    assert.equal(parsePlan('not json'), null);
  });
});

describe('parseEdits', () => {
  it('reads the documented multi-edit shape', () => {
    const edits = parseEdits('{"file":"a.py","edits":[{"find":"a","replace":"b"},{"find":"c","replace":"d"}]}');
    assert.equal(edits.length, 2);
    assert.deepEqual(edits[0], { find: 'a', replace: 'b' });
    assert.deepEqual(edits[1], { find: 'c', replace: 'd' });
  });

  it('reads a whole-file replacement', () => {
    assert.deepEqual(parseEdits('{"file":"a.py","content":"x = 1"}'), [{ content: 'x = 1' }]);
  });

  it('still accepts a bare find/replace, because models drift to the simpler shape', () => {
    // Tolerance here is deliberate: discarding a good single edit over shape would be
    // a worse failure than accepting an undocumented one.
    assert.deepEqual(parseEdits('{"file":"a.py","find":"a","replace":"b"}'), [{ find: 'a', replace: 'b' }]);
  });

  it('returns an empty list rather than throwing when nothing usable is present', () => {
    assert.deepEqual(parseEdits('{"file":"a.py"}'), []);
    assert.deepEqual(parseEdits('garbage'), []);
    // A find with no replace is not an edit.
    assert.deepEqual(parseEdits('{"file":"a.py","find":"a"}'), []);
  });

  it('keeps every edit when the step covers several changes', () => {
    // The regression: two changes were asked for, one was returned, and the loss was
    // invisible downstream.
    const edits = parseEdits(
      '{"file":"stats.py","edits":[{"find":"len(values) - 1","replace":"len(values)"},{"find":"return low","replace":"return value"}]}',
    );
    assert.equal(edits.length, 2, 'both changes must survive parsing');
  });
});

/* ------------------------------------------------------------------ */
/* Assembly                                                            */
/* ------------------------------------------------------------------ */

function step(file: string, edits: Array<{ find?: string; replace?: string; content?: string }>): ExecutedStep {
  return {
    step: { file, intent: 'whatever' } satisfies PlanStep,
    edits,
    durationMs: 1,
    inputTokens: 10,
    outputTokens: 5,
  };
}

describe('assembleCandidate', () => {
  it('emits one envelope entry per edit, not per step', () => {
    const text = assembleCandidate({ summary: 's', steps: [] }, [
      step('a.py', [
        { find: 'x', replace: 'y' },
        { find: 'p', replace: 'q' },
      ]),
    ]);
    const parsed = JSON.parse(text) as { edits: unknown[] };
    assert.equal(parsed.edits.length, 2, 'a step with two edits must produce two entries');
  });

  it('spans several files, in step order', () => {
    const text = assembleCandidate({ summary: 's', steps: [] }, [
      step('a.py', [{ find: 'x', replace: 'y' }]),
      step('b.py', [{ find: 'p', replace: 'q' }]),
    ]);
    const parsed = JSON.parse(text) as { edits: Array<{ file: string }> };
    assert.deepEqual(parsed.edits.map((e) => e.file), ['a.py', 'b.py']);
  });

  it('passes a whole-file replacement through as content', () => {
    const text = assembleCandidate({ summary: 's', steps: [] }, [step('new.py', [{ content: 'x = 1' }])]);
    const parsed = JSON.parse(text) as { edits: Array<{ content?: string }> };
    assert.equal(parsed.edits[0]?.content, 'x = 1');
  });

  it('drops steps that produced nothing instead of emitting an empty edit', () => {
    const text = assembleCandidate({ summary: 's', steps: [] }, [step('a.py', []), step('b.py', [{ find: 'p', replace: 'q' }])]);
    const parsed = JSON.parse(text) as { edits: unknown[] };
    assert.equal(parsed.edits.length, 1);
  });

  it('carries an answer through for a question', () => {
    const text = assembleCandidate({ summary: 's', steps: [], answer: 'because' }, []);
    const parsed = JSON.parse(text) as { answer: string; edits: unknown[] };
    assert.equal(parsed.answer, 'because');
    assert.deepEqual(parsed.edits, []);
  });
});

/* ------------------------------------------------------------------ */
/* Provider interaction                                                */
/* ------------------------------------------------------------------ */

describe('planTask', () => {
  it('parses a plan out of a fenced reply', async () => {
    const provider = new MockProvider({
      id: 'planner',
      kind: 'cloud',
      response: '```json\n{"summary":"fix it","steps":[{"file":"a.py","intent":"change x"}]}\n```',
    });
    const cfg = testConfig('/tmp/split-test', {});
    const out = await planTask({ cfg, ctx: { task: 'do a thing' }, planner: provider });
    assert.equal(out.plan?.summary, 'fix it');
    assert.equal(out.plan?.steps.length, 1);
  });

  it('reports a null plan when the model ignores the format', async () => {
    const provider = new MockProvider({ id: 'planner', kind: 'cloud', response: 'I shall fix it forthwith.' });
    const cfg = testConfig('/tmp/split-test', {});
    const out = await planTask({ cfg, ctx: { task: 'do a thing' }, planner: provider });
    assert.equal(out.plan, null, 'an unparseable plan must be reported, not guessed at');
  });
});

describe('executeStep', () => {
  const cfg = testConfig('/tmp/split-test', {});

  it('returns every edit the executor emits', async () => {
    const provider = new MockProvider({
      id: 'executor',
      kind: 'local',
      response: '{"file":"a.py","edits":[{"find":"a","replace":"b"},{"find":"c","replace":"d"}]}',
    });
    const out = await executeStep({
      cfg,
      executor: provider,
      step: { file: 'a.py', intent: 'two changes' },
      files: [{ path: 'a.py', content: 'a\nc\n' }],
    });
    assert.equal(out.edits.length, 2);
    assert.equal(out.error, undefined);
  });

  it('flags a step that produced no edit, so the caller can count the loss', async () => {
    const provider = new MockProvider({ id: 'executor', kind: 'local', response: '{"file":"a.py"}' });
    const out = await executeStep({
      cfg,
      executor: provider,
      step: { file: 'a.py', intent: 'do something' },
      files: [{ path: 'a.py', content: 'a\n' }],
    });
    assert.deepEqual(out.edits, []);
    assert.match(out.error ?? '', /no usable/);
  });

  it('includes the rejection reason when retrying a step', async () => {
    const provider = new MockProvider({ id: 'executor', kind: 'local', response: '{"file":"a.py","edits":[{"find":"a","replace":"b"}]}' });
    await executeStep({
      cfg,
      executor: provider,
      step: { file: 'a.py', intent: 'change a' },
      files: [{ path: 'a.py', content: 'a\n' }],
      feedback: ['anchor text appears 2 times'],
    });
    // The feedback has to reach the model, or a retry is just a re-roll.
    const sent = provider.history.at(-1)?.messages.map((m) => m.content).join('\n') ?? '';
    assert.match(sent, /anchor text appears 2 times/);
    assert.match(sent, /REJECTED/);
  });
});
