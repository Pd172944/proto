/**
 * Evaluation corpus.
 *
 * A router cannot be improved without a fixed yardstick, and "it felt right in
 * the demo" is not one. This corpus is deliberately small, hand-written and
 * *opinionated*: each task carries the tier a careful engineer would choose,
 * plus the task class we expect the classifier to find.
 *
 * Two design notes:
 *
 *  - Tasks are labelled by *expected competence*, not by token count. A 3-line
 *    change to an auth check is a cloud task; a 40-line mechanical rename is a
 *    local task. A corpus where length predicts the label would let a trivial
 *    length heuristic score 100% and teach us nothing.
 *
 *  - `verifiable` marks whether the standard verifier can catch a wrong answer.
 *    Routing a task to local when it is unverifiable is a different (and worse)
 *    kind of mistake than when it is verifiable, so the report separates them.
 */

import type { TaskClass } from '../router/types.ts';

export interface EvalTask {
  id: string;
  /** What the user types. */
  task: string;
  /** Files in scope. */
  files?: Array<{ path: string; content: string }>;
  diff?: string;
  constraints?: string[];
  /** The tier a careful engineer would pick: cheap tier or expensive tier. */
  expected: 'local' | 'cloud';
  /** Expected classification, when the task has an unambiguous class. */
  expectedClass?: TaskClass;
  /** Can the verifier detect a wrong answer for this task? */
  verifiable: boolean;
  notes?: string;
}

const PY_LOOP = `def total_prices(items):
    total = 0
    for i in range(len(items) + 1):
        total += items[i]["price"]
    return total
`;

const PY_VALIDATION = `def parse_port(raw):
    port = int(raw)
    return port
`;

const TS_RENAME = `export function calcTotalPrice(items: Item[]): number {
  return items.reduce((sum, item) => sum + item.price, 0);
}

export function calcTotalTax(items: Item[], rate: number): number {
  return calcTotalPrice(items) * rate;
}
`;

const TS_FORMAT = `export const  config={name:"proto",retries:3,timeout:5000}
export function getConfig( ){return config}
`;

const PROMPT_TEXT = `SYSTEM_PROMPT = """You are a helpful assistant. Answer the question."""

def build_prompt(question):
    return SYSTEM_PROMPT + "\\n" + question
`;

const DOCSTRING = `def retry(fn, attempts=3):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            continue
    raise RuntimeError("retry exhausted")
`;

const AUTH = `def check_access(user, resource):
    if user.role == "admin":
        return True
    return resource.owner_id == user.id
`;

const SQL_MIGRATION = `CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  email TEXT NOT NULL
);
`;

const ASYNC_QUEUE = `async def process_all(jobs, worker):
    results = []
    for job in jobs:
        results.append(await worker(job))
    return results
`;

const SCHEMA_TS = `export interface Order {
  id: string;
  userId: string;
  totalCents: number;
}
export interface User {
  id: string;
  email: string;
}
`;

const EXPLAIN_SNIPPET = `def f(x, memo={}):
    if x in memo:
        return memo[x]
    memo[x] = x * 2
    return memo[x]
`;

const TEST_FILE = `from calc import add

def test_add():
    assert add(1, 2) == 3
`;

export const EVAL_TASKS: EvalTask[] = [
  /* ---------------- local: mechanical, bounded, high-prior ---------------- */
  {
    id: 'local-off-by-one-loop',
    task: 'Fix the off-by-one error in the loop so it does not go out of bounds.',
    files: [{ path: 'prices.py', content: PY_LOOP }],
    expected: 'local',
    expectedClass: 'bugfix-local',
    verifiable: true,
    notes: 'The motivating example from the project brief: a one-line, fully specified fix.',
  },
  {
    id: 'local-index-to-iteration',
    task: 'Rewrite this loop to iterate over `items` directly instead of using `range(len(items))`.',
    files: [{ path: 'prices.py', content: PY_LOOP }],
    expected: 'local',
    expectedClass: 'local-edit',
    verifiable: true,
  },
  {
    id: 'local-add-bounds-check',
    task: 'Add a guard so that an empty items list returns 0 instead of raising.',
    files: [{ path: 'prices.py', content: PY_LOOP }],
    expected: 'local',
    expectedClass: 'add-validation',
    verifiable: true,
  },
  {
    id: 'local-rename-helper',
    task: 'Rename `calcTotalPrice` to `calculateTotalPrice` everywhere in this file.',
    files: [{ path: 'cart.ts', content: TS_RENAME }],
    expected: 'local',
    expectedClass: 'rename',
    verifiable: true,
  },
  {
    id: 'local-format-object',
    task: 'Reformat this file to use consistent indentation and spaces after colons.',
    files: [{ path: 'config.ts', content: TS_FORMAT }],
    expected: 'local',
    expectedClass: 'format',
    verifiable: true,
  },
  {
    id: 'local-prompt-edit',
    task: 'Change the system prompt to say the assistant must answer in one short paragraph.',
    files: [{ path: 'prompts.py', content: PROMPT_TEXT }],
    expected: 'local',
    expectedClass: 'prompt-edit',
    verifiable: true,
    notes: 'Prompt editing is exactly the "easy but fiddly" class the local tier exists for.',
  },
  {
    id: 'local-docstring',
    task: 'Add a docstring to `retry` describing its arguments and the exception it raises.',
    files: [{ path: 'retry.py', content: DOCSTRING }],
    expected: 'local',
    expectedClass: 'docs',
    verifiable: true,
  },
  {
    id: 'local-validation-port',
    task: 'Add validation so `parse_port` rejects values outside 1-65535 and non-numeric input.',
    files: [{ path: 'ports.py', content: PY_VALIDATION }],
    expected: 'local',
    expectedClass: 'add-validation',
    verifiable: true,
  },
  {
    id: 'local-add-test-case',
    task: 'Add a test for the case where both arguments are negative.',
    files: [{ path: 'test_calc.py', content: TEST_FILE }],
    expected: 'local',
    expectedClass: 'write-tests',
    verifiable: true,
  },
  {
    id: 'local-explain-mutable-default',
    task: 'Explain what is wrong with the default argument in this function.',
    files: [{ path: 'cache.py', content: EXPLAIN_SNIPPET }],
    expected: 'local',
    expectedClass: 'explain',
    verifiable: false,
    notes: 'Explanation tasks are high-prior for local models but NOT machine-verifiable.',
  },
  {
    id: 'local-comment-behaviour',
    task: 'Document what this function returns for repeated inputs.',
    files: [{ path: 'cache.py', content: EXPLAIN_SNIPPET }],
    expected: 'local',
    expectedClass: 'explain',
    verifiable: false,
  },
  {
    id: 'local-remove-dead-branch',
    task: 'Remove the unreachable `else` branch in this function.',
    files: [
      {
        path: 'access.py',
        content: `def can(user):
    if user.active:
        return True
    else:
        return False
`,
      },
    ],
    expected: 'local',
    expectedClass: 'local-edit',
    verifiable: true,
  },

  /* ---------------- cloud: ambiguous, cross-cutting or high-risk ---------- */
  {
    id: 'cloud-auth-review',
    task: 'Harden the access check against privilege escalation and add tests for the bypass cases.',
    files: [{ path: 'auth.py', content: AUTH }],
    expected: 'cloud',
    expectedClass: 'security',
    verifiable: true,
    notes: 'A silently wrong auth change is expensive; class is hard-locked to the cloud.',
  },
  {
    id: 'cloud-schema-migration',
    task: 'Migrate the database from integer ids to UUIDs without downtime, keeping the API stable.',
    files: [{ path: 'migrations/001_users.sql', content: SQL_MIGRATION }],
    expected: 'cloud',
    expectedClass: 'migration',
    verifiable: false,
  },
  {
    id: 'cloud-parallelise-workers',
    task: 'Parallelise the worker fan-out with bounded concurrency and correct error propagation.',
    files: [{ path: 'worker.py', content: ASYNC_QUEUE }],
    expected: 'cloud',
    expectedClass: 'concurrency',
    verifiable: true,
  },
  {
    id: 'cloud-module-boundaries',
    task: 'Design the module boundaries between orders and users so that the API can evolve independently.',
    files: [{ path: 'types.ts', content: SCHEMA_TS }],
    expected: 'cloud',
    expectedClass: 'architecture',
    verifiable: false,
  },
  {
    id: 'cloud-refactor-across-repo',
    task: 'Refactor the duplicated validation logic across the codebase into a shared module.',
    files: [
      { path: 'a/validate.py', content: 'def validate(x):\n    return x is not None\n' },
      { path: 'b/check.py', content: 'def check(x):\n    return x is not None\n' },
      { path: 'c/guard.py', content: 'def guard(x):\n    return x is not None\n' },
      { path: 'd/verify.py', content: 'def verify(x):\n    return x is not None\n' },
    ],
    expected: 'cloud',
    expectedClass: 'refactor-multi',
    verifiable: true,
    notes: 'Four files in scope must upgrade a local-edit classification to refactor-multi.',
  },
  {
    id: 'cloud-perf-hot-loop',
    task: 'Profile and fix the performance regression in the hot path; it is 4x slower after the last release.',
    files: [{ path: 'hot.py', content: PY_LOOP }],
    expected: 'cloud',
    expectedClass: 'perf',
    verifiable: true,
  },
  {
    id: 'cloud-vague-debug',
    task: 'Sometimes the worker silently drops jobs. Figure out why and fix it.',
    expected: 'cloud',
    expectedClass: 'debug-unknown',
    verifiable: false,
    notes: 'No repro, no files, "sometimes": maximum ambiguity.',
  },
  {
    id: 'cloud-feature-endpoint',
    task: 'Implement a new paginated endpoint for order history with cursor pagination and filtering.',
    expected: 'cloud',
    expectedClass: 'feature-new',
    verifiable: true,
  },
  {
    id: 'cloud-algorithm-scheduler',
    task: 'Implement a topological scheduler for the task graph that minimises total makespan.',
    expected: 'cloud',
    expectedClass: 'algorithm',
    verifiable: true,
  },
  {
    id: 'cloud-multi-constraint',
    task:
      'Update the pricing rules so that discounts apply only to non-sale items, must not stack with coupons, ' +
      'must round half-up to the nearest cent, and must keep the existing API unchanged.',
    files: [{ path: 'pricing.py', content: PY_VALIDATION }],
    expected: 'cloud',
    expectedClass: 'local-edit',
    verifiable: true,
    notes: 'Four interacting constraints: individually easy, jointly a cloud task.',
  },
];

export function tasksById(): Map<string, EvalTask> {
  return new Map(EVAL_TASKS.map((t) => [t.id, t]));
}
