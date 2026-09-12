/**
 * Verifier tests.
 *
 * The verifier is the component that lets a weak model be used safely, so these
 * tests focus on the failure modes that matter: does it refuse to apply a patch
 * whose anchor is absent or ambiguous, does it catch real syntax errors, and does
 * it refuse destructive or suspicious changes.
 */

import { strict as assert } from 'node:assert';
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { parseCandidate, applyEdits, validateEditPath, diffLineCount, addedLines, patternChecks } from '../src/verify/patch.ts';
import { balanceCheck, syntaxCheck } from '../src/verify/syntax.ts';
import { verifyCandidate } from '../src/verify/index.ts';
import { extractFeatures } from '../src/router/features.ts';
import { hasBinary } from '../src/util/proc.ts';
import { tempDir, testConfig } from './helpers.ts';

const HAS_PYTHON = await hasBinary('python3');

const GOOD_PY = `def total(items):
    return sum(item["price"] for item in items)
`;

describe('candidate parsing', () => {
  it('parses the requested JSON envelope', () => {
    const parsed = parseCandidate(
      JSON.stringify({ summary: 'fix', edits: [{ file: 'a.py', find: 'x', replace: 'y' }], risk: 'low' }),
    );
    assert.equal(parsed.shape, 'json');
    assert.equal(parsed.candidate?.edits.length, 1);
    assert.equal(parsed.candidate?.edits[0]?.file, 'a.py');
  });

  it('accepts aliases the models actually emit', () => {
    const parsed = parseCandidate(
      JSON.stringify({ description: 'd', changes: [{ path: 'a.py', old: 'x', new: 'y' }] }),
    );
    assert.equal(parsed.candidate?.edits[0]?.find, 'x');
    assert.equal(parsed.candidate?.edits[0]?.replace, 'y');
  });

  it('salvages a single fenced code block when the envelope is missing', () => {
    const parsed = parseCandidate('Here you go:\n```python\nprint(1)\n```', { expectedPaths: ['a.py'] });
    assert.equal(parsed.shape, 'fence');
    assert.equal(parsed.candidate?.edits[0]?.content, 'print(1)\n');
  });

  it('treats leftover prose as an answer, not as edits', () => {
    const parsed = parseCandidate('The function is memoised, so repeated calls are O(1).');
    assert.equal(parsed.shape, 'prose');
    assert.equal(parsed.candidate?.edits.length, 0);
    assert.ok((parsed.candidate?.answer ?? '').length > 10);
  });

  it('rejects an envelope with no usable edits', () => {
    const parsed = parseCandidate(JSON.stringify({ summary: 'nothing', edits: [] }));
    assert.ok(parsed.parseError);
  });
});

describe('edit application', () => {
  it('applies a unique anchor', () => {
    const known = new Map([['a.py', GOOD_PY]]);
    const { applied, errors } = applyEdits(
      { summary: 's', edits: [{ file: 'a.py', find: 'def total(items):', replace: 'def total(all_items):' }] },
      known,
    );
    assert.equal(errors.length, 0);
    assert.match(applied[0]?.after ?? '', /def total\(all_items\)/);
  });

  it('refuses an absent anchor instead of guessing', () => {
    const known = new Map([['a.py', GOOD_PY]]);
    const { applied, errors } = applyEdits(
      { summary: 's', edits: [{ file: 'a.py', find: 'def nonexistent():', replace: 'x' }] },
      known,
    );
    assert.equal(applied.length, 0);
    assert.match(errors[0]?.detail ?? '', /anchor text not found/);
  });

  it('refuses an ambiguous anchor', () => {
    const known = new Map([['a.py', 'x = 1\ny = 2\nx = 1\n']]);
    const { applied, errors } = applyEdits(
      { summary: 's', edits: [{ file: 'a.py', find: 'x = 1', replace: 'x = 2' }] },
      known,
    );
    assert.equal(applied.length, 0);
    assert.match(errors[0]?.detail ?? '', /appears 2 times/);
  });

  it('detects a no-op edit', () => {
    const known = new Map([['a.py', GOOD_PY]]);
    const { errors } = applyEdits(
      { summary: 's', edits: [{ file: 'a.py', find: 'return', replace: 'return' }] },
      known,
    );
    assert.match(errors[0]?.detail ?? '', /no-op/);
  });

  it('refuses paths outside the workspace and secret files', () => {
    assert.equal(validateEditPath('/etc/passwd').ok, false);
    assert.equal(validateEditPath('../../etc/passwd').ok, false);
    assert.equal(validateEditPath('node_modules/x/y.js').ok, false);
    assert.equal(validateEditPath('.git/config').ok, false);
    assert.equal(validateEditPath('.env').ok, false);
    assert.equal(validateEditPath('secrets.json').ok, false);
    assert.equal(validateEditPath('src/app/main.py').ok, true);
  });

  it('counts changed lines without an O(n*m) diff', () => {
    assert.equal(diffLineCount('a\nb\nc\n', 'a\nB\nc\n'), 2);
    assert.ok(diffLineCount('a\n', 'a\nb\nc\nd\n') >= 3);
    assert.deepEqual(addedLines('a\nb\n', 'a\nc\n'), ['c']);
  });
});

describe('syntax checking', () => {
  it('uses a real Python parser when available', async (t) => {
    if (!HAS_PYTHON) return t.skip('python3 not installed');
    const dir = tempDir();
    const ok = await syntaxCheck(GOOD_PY, 'python', dir);
    assert.equal(ok.ok, true);
    assert.equal(ok.method, 'parser');

    const bad = await syntaxCheck('def f(:\n  pass\n', 'python', dir);
    assert.equal(bad.ok, false);
    assert.equal(bad.method, 'parser');
  });

  it('detects JavaScript syntax errors with node', async () => {
    const dir = tempDir();
    const ok = await syntaxCheck('export const a = [1, 2].map((x) => x * 2);\n', 'javascript', dir);
    assert.equal(ok.ok, true);
    const bad = await syntaxCheck('const a = [1, 2;\n', 'javascript', dir);
    assert.equal(bad.ok, false);
  });

  it('validates JSON and reports a helpful detail', async () => {
    const dir = tempDir();
    assert.equal((await syntaxCheck('{"a":1}', 'json', dir)).ok, true);
    assert.equal((await syntaxCheck('{"a":}', 'json', dir)).ok, false);
  });

  it('balances delimiters with nesting awareness and comment handling', () => {
    assert.equal(balanceCheck('f(a[0], {b: 1})').ok, true);
    assert.equal(balanceCheck('f(a[0)').ok, false);
    assert.equal(balanceCheck('# ( unbalanced in a comment').ok, true);
    assert.equal(balanceCheck('x = "(not a real paren"').ok, true);
    assert.equal(balanceCheck('x = "unterminated').ok, false);
  });
});

describe('pattern checks', () => {
  const applied = (before: string, after: string): ReturnType<typeof applyEdits>['applied'] => [
    { path: 'a.py', before, after, changedLines: 1 },
  ];

  it('flags a newly introduced TODO as an error when configured to reject them', () => {
    const results = patternChecks(applied('x = 1\n', 'x = 1\n# TODO: fix this\n'), {
      rejectNewTodos: true,
      forbiddenPatterns: [],
    });
    assert.equal(results.length, 1);
    assert.equal(results[0]?.severity, 'error');
  });

  it('flags hardcoded credentials and dangerous primitives as errors', () => {
    const secret = patternChecks(applied('', 'api_key = "abcdef1234567890"\n'), { rejectNewTodos: false, forbiddenPatterns: [] });
    assert.ok(secret.some((r) => r.id.includes('hardcoded-secret')));
    const shell = patternChecks(applied('', 'import os\nos.system("ls")\n'), { rejectNewTodos: false, forbiddenPatterns: [] });
    assert.ok(shell.some((r) => r.id.includes('dangerous-shell')));
  });

  it('does not flag pre-existing problems that the patch did not touch', () => {
    const before = '# TODO: legacy\nx = 1\n';
    const after = '# TODO: legacy\nx = 1\ny = 2\n';
    const results = patternChecks(applied(before, after), { rejectNewTodos: true, forbiddenPatterns: [] });
    assert.equal(results.length, 0, 'only added lines may be judged');
  });

  it('honours user-configured forbidden patterns', () => {
    const results = patternChecks(applied('', 'send_to_s3(data)\n'), {
      rejectNewTodos: false,
      forbiddenPatterns: ['send_to_s3'],
    });
    assert.equal(results.length, 1);
    assert.equal(results[0]?.severity, 'error');
  });
});

describe('verifyCandidate end to end', () => {
  const dir = tempDir();
  const cfg = testConfig(dir);
  const ctx = {
    task: 'Fix the off-by-one loop.',
    files: [{ path: 'prices.py', content: 'for i in range(len(items) + 1):\n    pass\n' }],
  };
  const features = extractFeatures(ctx);

  it('passes a valid python patch', async () => {
    const report = await verifyCandidate({
      text: JSON.stringify({
        summary: 'fix',
        edits: [{ file: 'prices.py', content: 'for item in items:\n    pass\n' }],
      }),
      ctx,
      cfg,
      dataDir: dir,
      features,
    });
    assert.equal(report.passed, true, JSON.stringify(report.blockers));
    assert.ok(report.score > 0.8);
  });

  it('fails a patch with broken syntax and says why', async (t) => {
    if (!HAS_PYTHON) return t.skip('python3 not installed');
    const report = await verifyCandidate({
      text: JSON.stringify({
        summary: 'fix',
        edits: [{ file: 'prices.py', content: 'for item in items(\n    pass\n' }],
      }),
      ctx,
      cfg,
      dataDir: dir,
      features,
    });
    assert.equal(report.passed, false);
    assert.ok(report.blockers.some((b) => /syntax/i.test(b)), report.blockers.join(' | '));
    assert.ok(report.score <= 0.3, 'a failed candidate must not look almost-good to the reward function');
  });

  it('fails a candidate that changes nothing', async () => {
    const report = await verifyCandidate({
      text: JSON.stringify({ summary: 'noop', edits: [{ file: 'prices.py', find: 'pass', replace: 'pass' }] }),
      ctx,
      cfg,
      dataDir: dir,
      features,
    });
    assert.equal(report.passed, false);
  });

  it('fails an edit task that returns no edits at all', async () => {
    const report = await verifyCandidate({
      text: JSON.stringify({ summary: 'I cannot do that', edits: [] }),
      ctx,
      cfg,
      dataDir: dir,
      features,
    });
    assert.equal(report.passed, false);
    assert.ok(report.blockers.some((b) => /contains none/.test(b)));
  });

  it('detects a refusal in an explanation task', async () => {
    const explainCtx = { task: 'Explain what this function does.', files: [{ path: 'a.py', content: GOOD_PY }] };
    const explainFeatures = extractFeatures(explainCtx);
    const report = await verifyCandidate({
      text: JSON.stringify({
        summary: 'x',
        answer: "I'm sorry, but as an AI I cannot analyze proprietary code in detail for you.",
      }),
      ctx: explainCtx,
      cfg,
      dataDir: dir,
      features: explainFeatures,
    });
    assert.equal(report.passed, false);
    assert.ok(report.blockers.some((b) => /refused/.test(b)));
  });

  it('accepts a substantive explanation', async () => {
    const explainCtx = { task: 'Explain what this function does.', files: [{ path: 'a.py', content: GOOD_PY }] };
    const explainFeatures = extractFeatures(explainCtx);
    const report = await verifyCandidate({
      text: JSON.stringify({
        summary: 'x',
        answer:
          'It sums the "price" field of every item and returns the total. It assumes each item is a mapping with a ' +
          '"price" key and will raise KeyError otherwise.',
      }),
      ctx: explainCtx,
      cfg,
      dataDir: dir,
      features: explainFeatures,
    });
    assert.equal(report.passed, true, JSON.stringify(report.blockers));
  });

  it('refuses a destructive mass deletion', async () => {
    const long = Array.from({ length: 60 }, (_, i) => `line_${i} = ${i}`).join('\n') + '\n';
    const report = await verifyCandidate({
      text: JSON.stringify({ summary: 'cleanup', edits: [{ file: 'big.py', content: 'x = 1\n' }] }),
      ctx: { task: 'Clean up big.py.', files: [{ path: 'big.py', content: long }] },
      cfg,
      dataDir: dir,
      features: extractFeatures({ task: 'Clean up big.py.', files: [{ path: 'big.py', content: long }] }),
    });
    assert.equal(report.passed, false);
    assert.ok(report.blockers.some((b) => /destructive/.test(b)));
  });

  it('refuses a patch larger than the configured cap', async () => {
    const small = testConfig(dir, { verify: { ...testConfig(dir).verify, maxPatchLines: 3 } });
    const report = await verifyCandidate({
      text: JSON.stringify({
        summary: 'big',
        edits: [{ file: 'prices.py', content: 'a\nb\nc\nd\ne\nf\ng\nh\n' }],
      }),
      ctx,
      cfg: small,
      dataDir: dir,
      features,
    });
    assert.equal(report.passed, false);
    assert.ok(report.blockers.some((b) => /above the 3-line cap/.test(b)));
  });

  it('reports unverified-as-neutral when verification is disabled', async () => {
    const off = testConfig(dir, { verify: { ...testConfig(dir).verify, enabled: false } });
    const report = await verifyCandidate({ text: 'anything', ctx, cfg: off, dataDir: dir, features });
    assert.equal(report.passed, true);
    assert.equal(report.score, 0.5);
  });

  it('runs the configured project tests and fails on a non-zero exit', async () => {
    const withTests = testConfig(dir, {
      verify: { ...testConfig(dir).verify, runTests: true, testCommand: 'exit 1' },
    });
    const report = await verifyCandidate({
      text: JSON.stringify({ summary: 'ok', edits: [{ file: 'prices.py', content: 'for item in items:\n    pass\n' }] }),
      ctx,
      cfg: withTests,
      dataDir: dir,
      features,
    });
    assert.equal(report.passed, false);
    assert.ok(report.blockers.some((b) => /test command failed/.test(b)));
  });

  it('writes nothing to the workspace during verification', async () => {
    const workspace = tempDir();
    const target = join(workspace, 'prices.py');
    writeFileSync(target, 'for i in range(len(items) + 1):\n    pass\n', 'utf8');
    const wsCtx = { task: 'Fix it.', workspace, files: [{ path: 'prices.py', content: 'for i in range(len(items) + 1):\n    pass\n' }] };
    const report = await verifyCandidate({
      text: JSON.stringify({ summary: 'fix', edits: [{ file: 'prices.py', content: 'for item in items:\n    pass\n' }] }),
      ctx: wsCtx,
      cfg: testConfig(workspace),
      dataDir: workspace,
      features: extractFeatures(wsCtx),
    });
    assert.equal(report.passed, true);
    const { readFileSync } = await import('node:fs');
    assert.match(readFileSync(target, 'utf8'), /range\(len\(items\) \+ 1\)/, 'verification must be read-only');
  });
});
