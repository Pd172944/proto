# Contributing

## Getting set up

```bash
git clone https://github.com/Pd172944/proto.git
cd proto
npm test          # no install step: there are no runtime dependencies
```

Node 22.6+ is required for TypeScript type stripping. On 22.x it needs a flag, which
`npm test` supplies; on 24+ it needs nothing.

Typechecking needs `typescript`, which is a devDependency and deliberately not a
runtime one:

```bash
npm install            # or: npm install --no-save typescript @types/node
npx tsc --noEmit
npm test
```

## The constraints that are not negotiable

These are load-bearing and a patch that breaks one will be sent back regardless of how
good it otherwise is.

**No runtime dependencies.** Everything runs on Node built-ins. `npm test` must work on
a fresh clone with no install step. CI enforces this.

**Erasable TypeScript only.** No `enum`, no `namespace`, no parameter properties
(`constructor(private x: T)`), no decorators. The code runs through Node's type
*stripping*, which removes annotations and nothing else. Use `import type` for types and
include the `.ts` extension on relative imports.

**The tests must pass, and new behaviour needs a test.** `npm test` is the whole
contract. If you are fixing a bug, the test that would have caught it is part of the
fix.

## Style

Read a neighbouring file before writing a new one. The house style is comments that
explain *why* a decision was made, what the alternative was, and where the approach
breaks — not comments that restate the code. If a tradeoff was made, say so and say
which way it falls. Numbers that came from a measurement should say so and include the
measurement.

Honest limitation notes are welcome and expected. "This is untested at scale" is a
useful comment; a confident comment that later turns out to be wrong is not.

## Commits

Explain the reasoning, not the diff. A commit message that says what changed and why,
including what was ruled out and what is still broken, is worth more than one that
lists files.
