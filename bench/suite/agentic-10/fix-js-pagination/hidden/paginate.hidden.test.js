'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

// The workspace under test is named by the harness; never rely on the cwd, because
// this file runs from the hidden-test scratch directory.
const workspace = process.env.BENCH_WORKSPACE;
const { paginate } = require(path.join(workspace, 'paginate.js'));

const items = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];

test('page 1 is the first slice', () => {
  assert.deepEqual(paginate(items, 1, 3), ['a', 'b', 'c']);
});

test('page 2 is the second slice', () => {
  assert.deepEqual(paginate(items, 2, 3), ['d', 'e', 'f']);
});

test('out-of-range page returns an empty array', () => {
  assert.deepEqual(paginate(items, 9, 3), []);
});

test('exact multiple: the page after the last is empty', () => {
  assert.deepEqual(paginate([1, 2, 3, 4], 3, 2), []);
});

test('perPage larger than the list returns everything on page 1', () => {
  assert.deepEqual(paginate([1, 2], 1, 10), [1, 2]);
});
