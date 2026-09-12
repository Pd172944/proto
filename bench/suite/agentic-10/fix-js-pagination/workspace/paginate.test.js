'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { paginate } = require('./paginate.js');

const items = [1, 2, 3, 4, 5, 6, 7];

test('page 1 returns the first perPage items', () => {
  assert.deepEqual(paginate(items, 1, 3), [1, 2, 3]);
});

test('page 2 returns the next slice', () => {
  assert.deepEqual(paginate(items, 2, 3), [4, 5, 6]);
});

test('a partial final page returns the remainder', () => {
  assert.deepEqual(paginate(items, 3, 3), [7]);
});

test('a page past the end returns an empty array', () => {
  assert.deepEqual(paginate(items, 99, 3), []);
});
