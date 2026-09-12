# The codebase index

How `proto code` finds things in a repository too large to read.

## The problem it removes

The agent's original search walked the tree and read **every file into memory on every
call**, with a hardcoded list of directory names to skip. On a toy repository that is
fine. On a real one it is three failures at once:

- it is O(repository bytes) *per search* — hundreds of megabytes of I/O on a monorepo;
- it walks `dist/`, `coverage/`, `.turbo/`, vendored SDKs, and anything else nobody
  thought to name;
- it silently stopped after 4000 files, so a file could be "not found" without the
  search ever reaching it.

The model then compensates the only way it can: by reading files to find out what
exists. That is how a large repository consumes a context window in about six tool
calls, and it is not the model being lazy — it had no cheaper way to see the shape of
the code.

## The four pieces

```
                    ┌──────────────┐
  git ls-files ────►│  discovery   │  walk.ts    + ignore.ts, .protoignore
  or a gitignore-   └──────┬───────┘
  aware walk               │  paths + mtime + size
                    ┌──────▼───────┐
                    │  extraction  │  lang.ts     maskCode → defs, refs, imports
                    └──────┬───────┘              (26 languages, 40 extensions)
                    ┌──────▼───────┐
                    │  symbol cache│  store.ts    JSONL, mtime+size keyed,
                    └──────┬───────┘              references dictionary-encoded
                    ┌──────▼───────┐
                    │ reference    │  graph.ts    weighted edges → personalised PageRank
                    │ graph        │
                    └──────┬───────┘
                    ┌──────▼───────┐
                    │  repo map    │  repomap.ts  signatures, cut to a token budget
                    └──────────────┘
```

**Discovery.** When the workspace is inside a git repository, `git ls-files -z --cached
--others --exclude-standard` is the answer: git already applies every `.gitignore`,
nested ignores, `.git/info/exclude` and the user's global excludes, in C. The walk
fallback exists only for a plain directory of source and implements the parts of the
gitignore grammar that change which files get indexed.

`.protoignore` is a per-workspace exclusion list in gitignore syntax. It exists because
*"is this part of the project"* and *"is this worth indexing"* are different questions:
a repository can legitimately track benchmark fixtures or vendored reference code that
is real source to git and pure noise to a symbol search. This repository ignores
`bench/suite/` for exactly that reason.

**Extraction** is a lexer pass, not a parser. `maskCode` blanks the contents of comments
and string literals while preserving byte length and every newline, so line numbers stay
exact and a definition regex can run without special-casing anything. Without masking, a
commented-out function and a SQL string containing `select` both become confident
phantom symbols that pollute the graph. No tree-sitter, no grammar files, no dependency.

**The cache** is keyed on `mtimeMs + size`. Definitions are stored plainly; references
are stored as indices into one dictionary of every distinct identifier in the
repository. On a large codebase that is roughly a 4x reduction, which is the difference
between an index people tolerate and one they delete. It is written atomically (temp
file + rename) so a crash cannot leave a half-index that a later load would parse as
complete.

**The graph** gives file A an edge to file B when A mentions a name B defines. Weighting
matters more than topology:

- *Rarity.* A name defined in one file and mentioned in three is a strong link. A name
  like `Config` that forty files define is noise, and without a guard it makes every file
  look related to every other. Names defined in more than 24 files are skipped entirely.
- *Imports over mentions.* A resolved import is a declared dependency (weight 3) and
  outranks an incidental identifier match (weight 1). Resolution is a constant-time
  suffix lookup, not a scan.
- *Personalisation.* PageRank's restart vector is seeded with the files and symbols **the
  task text names**. Uniform PageRank returns the repository's "important" files, which
  are the same on every task and therefore useless. This is what makes the ranking about
  the request.

**The repo map** renders signatures, not names and not bodies. A bare symbol list is
cheap and nearly useless — `handleRoute` does not say whether it takes a request or a
path. A body is 20x the size for information the model only needs once it has decided to
look. The signature is where the compression ratio is best.

## Tools the agent gets

| tool | the question it answers |
| --- | --- |
| `repo_map` | what is in this repository, and what matters for my task |
| `find_symbol` | where is this defined — ignoring comments, strings and mere mentions |
| `find_references` | what else uses this, so a rename does not break an unopened file |
| `file_outline` | what is in this file, without paying for its body |
| `search` | find it by content when the name is unknown |

`search` is now backed by `git grep -F` for literal patterns (exact, complete, C-speed)
and by `git grep -l` + a JavaScript regex for real ones — the shortlist comes from git,
the decision comes from the same `RegExp` the model wrote, so the dialect is always
JavaScript. An engine that fails falls through to the next; only an exhaustive scan is
allowed to conclude that something is absent.

On a repository of 300 files or more the system prompt tells the model explicitly to
work map-first, because its default behaviour — start reading — is what works on the
small repositories most examples use.

## Measured, on this repository

```
107 files, 1381 definitions
cold build   192 ms
warm rebuild  46 ms      (one stat per file, no re-reads)
cache size   277 KB
```

The cache is ~2.6 KB per file here. `find_references ipsWeight` returns five real uses
and correctly omits the three mentions of that name in doc comments — which is the whole
point of it not being a text search.

Numbers for a genuinely large repository are **estimates, not measurements**: this has
not been run on one. The design is O(files) for a cold build and O(changed files) warm,
and the cache should land in single-digit megabytes for a 10k-file codebase. Treat that
as a prediction to verify, not a result.

## What it deliberately does not do

- **No embeddings, no vector store.** It would add a dependency, a build step, a
  multi-hundred-megabyte artifact and a staleness problem, and it would have to run on
  the user's machine. Symbol graphs and agentic search are how the mature agents scale to
  large repositories; this is the same idea with less machinery.
- **No real parser.** Extraction is approximate and will miss symbols in exotic syntax.
  It reports `lang: 'text'` and still collects references when it does not recognise a
  file, so the file stays linkable in the graph rather than vanishing.
- **No staleness by content hash.** `mtime + size` can be fooled by a rewrite in the same
  millisecond to the same length. Hashing every file on every refresh costs a full read
  of the repository — the exact expense the index exists to avoid. The window is one
  refresh; `proto index --force` closes it.

## The property that matters most

**The index is an accelerator, never a dependency.** Every tool answers correctly when the
index is empty, stale, half-built or corrupt, and every repository tool says "the index is
unavailable, use `search` instead" rather than returning an empty result. A stale index
that silently hides a file is the worst failure this subsystem could have, because the
model would conclude the file does not exist and stop looking. That is also why
extraction failures, unreadable caches and a read-only data directory all degrade to a
slower answer instead of a wrong one.

## Inspecting it

```bash
proto index                                  # build, report size and timing
proto index --stats                          # numbers only
proto index --force                          # ignore the cache and re-extract
proto index --map "the router quality floors in policy.ts"   # see the ranking
proto index --symbol decideRoute             # where is it defined
proto index --refs decideRoute               # who uses it
proto index --outline src/router/policy.ts   # what is in this file
```

`--map` is the debugging tool: it prints the focus it derived from your text, so a bad
ranking is diagnosable rather than mysterious.

## Known limits

- **Extraction fidelity varies by language.** Rust skips `'` character literals so that
  `'a` lifetimes are not read as unterminated strings. Lua's `--[[ ]]` degrades to a line
  comment. Python and Ruby nesting uses an indentation stack, not real DEDENT/`end`
  tracking, so a `def` inside a module-level `if` can inherit an enclosing class as its
  parent. Java and C# class fields are not indexed, to avoid field-initialiser false
  positives. Vue and Svelte index only the `<script>` region.
- **Reference lookup shortlists, then re-reads.** The cache stores *which* files mention a
  name, not where. `find_references` narrows to those files and re-scans them with the
  same masking the extractor used. That is a handful of files rather than the repository,
  but it is not free.
- **The map is a heuristic.** It is good at "which files are near this task" and it will
  occasionally miss the one file that mattered. `search` and `find_symbol` are the
  backstops, and the map says how many files it did not show so a cut map is never
  mistaken for the whole repository.
- **The whole ranking is unvalidated against a real large codebase.** It has been tested
  on a 107-file repository and on synthetic graphs in `test/index-core.test.ts`. Whether
  the weights are right at 50k files is an open question, and the first thing to check
  when someone points it at one.
