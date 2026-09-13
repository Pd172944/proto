/**
 * Tests for disposable-edit classification and search-pattern normalisation.
 *
 * These are the two silent failures that emptied SWE-bench patches: a repro
 * script counted as "the task was done", and `/stack/`-style patterns (or
 * `git grep -m` on this machine's git) made every content search return
 * zero hits so the model never found the change site.
 */

import { strict as assert } from 'node:assert';
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { isDisposableEdit, productionEdits } from '../src/agent/edits.ts';
import { normalizeSearchPattern, searchWithGit } from '../src/index/search.ts';
import { tempDir } from './helpers.ts';

describe('isDisposableEdit', () => {
  it('treats repro scripts as scratch, including numbered variants', () => {
    assert.equal(isDisposableEdit('repro.py'), true);
    assert.equal(isDisposableEdit('repro2.py'), true);
    assert.equal(isDisposableEdit('./reproduce.py'), true);
    assert.equal(isDisposableEdit('apply_fix.py'), true);
  });

  it('treats tests as scratch', () => {
    assert.equal(isDisposableEdit('xarray/tests/test_indexes.py'), true);
    assert.equal(isDisposableEdit('test_foo.py'), true);
  });

  it('keeps real library source', () => {
    assert.equal(isDisposableEdit('xarray/core/indexes.py'), false);
    assert.equal(isDisposableEdit('requests/utils.py'), false);
    assert.equal(isDisposableEdit('django/db/models/sql/query.py'), false);
  });

  it('filters productionEdits down to library files', () => {
    assert.deepEqual(productionEdits(['repro.py', 'xarray/core/indexes.py', 'tests/test_x.py']), [
      'xarray/core/indexes.py',
    ]);
  });
});

describe('normalizeSearchPattern', () => {
  it('strips JavaScript regex literal delimiters', () => {
    assert.equal(normalizeSearchPattern('/stack/'), 'stack');
    assert.equal(normalizeSearchPattern('/def stack/i'), 'def stack');
    assert.equal(normalizeSearchPattern('stack'), 'stack');
  });
});

describe('searchWithGit', () => {
  it('finds a literal identifier even when git grep -m is unsupported', () => {
    const root: string = tempDir('proto-search-');
    mkdirSync(join(root, 'pkg'));
    writeFileSync(join(root, 'pkg', 'indexes.py'), 'def stack():\n    return coord_dtype\n');
    execFileSync('git', ['init'], { cwd: root, stdio: 'ignore' });
    execFileSync('git', ['add', 'pkg/indexes.py'], { cwd: root, stdio: 'ignore' });
    const outcome = searchWithGit(root, { pattern: 'stack', caseSensitive: false });
    assert.ok(outcome, 'git grep should be available');
    assert.ok(outcome.hits.length > 0, 'expected at least one hit for stack');
    assert.equal(outcome.hits[0]?.path, 'pkg/indexes.py');
  });
});
