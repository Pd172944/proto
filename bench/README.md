# The agentic benchmark suite

Ten small tasks that measure `proto code` — the agent loop, the verifier, and how far a
given model tier actually gets. They exist because HumanEval measures a *completion*
model (one function, no repo, no tests to run) and tells you almost nothing about an
agent, while Terminal-Bench and SWE-bench measure the right thing but require Docker,
which is not available everywhere.

These tasks have the shape of Terminal-Bench — a real environment, a hidden verifier,
freedom to iterate — with none of the infrastructure. No containers, no network, no image
pulls. What you give up is scale and external validity: see *Honest limits* below.

## Running it

```bash
python3 scripts/bench.py list                  # what is in the suite
python3 scripts/bench.py validate              # prove every task is honest (see below)
python3 scripts/bench.py run --tier local      # the number you care about
python3 scripts/bench.py run --tier local --only fix-tax --verbose
python3 scripts/bench.py report --results /tmp/proto-bench/<label>/results.jsonl
```

Runs land in `/tmp/proto-bench/<label>/`: `results.jsonl` (one row per task) plus each
task's workspace, the agent's stdout and stderr, and the verifier's output. When a task
fails, the first question is always "what did it do?" — the transcripts are there so you
can answer it.

`--tier local` uses the local model, `--tier cloud` the configured cloud provider, and
`--tier route` lets the router choose per task. For the cloud tiers you need an API key;
`--model` overrides the model for the run.

## What `validate` is for

A task is only a task if it **fails before any work is done** and **passes after the
reference solution is applied**. `validate` checks both halves, because each catches a
different bug:

- a verifier that already passes is measuring nothing, and would hand out free points;
- a reference solution that does not pass means the task is not actually solvable as
  written, and every failure against it is meaningless.

Run it after touching any task. It is the reason a delegating author can be trusted:
a wrong task gets caught by the harness rather than by your conclusions.

## Isolation, and why it is not optional

Each run gets a fresh copy of the task workspace, a fresh `PROTO_HOME`, and a `PATH`
whose first entry is a shared virtualenv built from the system interpreter.

That last part is load-bearing. Before it, the agent inherited the machine's Python and
spent a third of its step budget fighting a third-party pytest plugin that failed to
import — a broken `langsmith`/`pydantic` chain in the user's `site-packages`. It was
being measured on the shell profile, not on the task. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
closes that particular hole; the venv closes the class of them.

**Grading never uses the workspace's own tests.** Visible tests may live in the workspace
so the agent can iterate, but `verify.sh` runs the hidden copies from `hidden/`, in a
scratch directory, with `BENCH_WORKSPACE` pointing at the agent's tree. Deleting or
weakening a test in the workspace cannot earn a pass.

## Task format

```
bench/suite/<suite>/<task-id>/
  task.md        # the prompt, verbatim. No hints about the solution.
  meta.json      # {"tags": [...], "timeoutSec": 300}
  workspace/     # the initial BROKEN state
  hidden/        # hidden tests, copied to a scratch dir at grading time
  verify.sh      # exits 0 iff solved; cwd = scratch, BENCH_WORKSPACE set
  solution/      # reference solution files, copied over the workspace
  solution.sh    # ...or the answer as commands, for things a file copy cannot express
  setup.sh       # optional: builds initial state that is not expressible as files
```

`setup.sh` and `solution.sh` run with cwd set to the workspace. A task may use both
`solution/` and `solution.sh`; the files are copied first, because the commands may need
to stage them.

Git-based tasks **must** use `setup.sh` rather than shipping a `.git` directory: a nested
repository inside this repo is recorded as a gitlink, not as a directory of files, and
would not survive a clone. The same is true of anything else in the tree that a
`.gitignore` rule would swallow — check `git status --ignored bench` after adding
fixtures.

## The tasks

| task | tags | what it exercises |
| --- | --- | --- |
| `fix-tax` | python, bugfix | two functions, two bugs, visible tests |
| `fix-bsearch` | python, bugfix, algorithms | an off-by-one that also fails to terminate |
| `log-extract` | shell, text | filter a log file into an exact output file |
| `csv-aggregate` | shell, text, data | group, sum and format a CSV into a report |
| `rename-symbol` | refactor, multi-file | rename a symbol at every call site |
| `git-recover` | git, terminal | restore a file from history |
| `config-edit` | json, config | edit a config to satisfy three constraints |
| `fix-js-pagination` | javascript, bugfix | an off-by-one in JS with `node --test` |
| `make-target` | make, build | add a default `test` target to a Makefile |
| `commit-hygiene` | git, terminal | commit, untrack a secret, ignore a build artifact |

## Honest limits

Read this before quoting a number.

- **Ten tasks is not a leaderboard.** The 95% interval around a 10-sample success rate is
  enormous: 8/10 is consistent with a true rate anywhere from roughly 49% to 94%. Two
  runs disagreeing by one task is noise, not a regression.
- **Run-to-run variance is real.** Sampling is stochastic (temperature 0.2), so a single
  run cannot distinguish "the model cannot do this" from "the model did not do this
  today". Run it more than once before drawing a conclusion, and read the transcripts
  rather than only the pass/fail column.
- **The tasks are written to be winnable.** A suite that floors at 0/10 tells you
  nothing, so several tasks are deliberately easy. If a tier scores 10/10 the suite is
  too easy for it, and the honest response is harder tasks, not a victory lap.
- **It cannot be compared to published SWE-bench or Terminal-Bench numbers.** Those
  suites are larger, differently distributed, and graded in containers. This measures
  whether *your* harness works and where *your* tiers sit relative to each other.
- **The verifier is a script you wrote.** It checks the properties someone thought to
  check. `validate` proves it fails before and passes after; it cannot prove it checks
  everything that matters.

## Going further

The ladder, in the order that gets information soonest:

1. **This suite.** Fast, no infrastructure, tells you whether the harness works at all.
2. **Terminal-Bench.** Needs Docker. The task shape is the same, so `scripts/bench.py`
   is a reasonable place to learn what an agent adapter has to do before writing one.
3. **SWE-bench Lite.** Needs Docker and, realistically, a much stronger model than a 9B.
   Expect mostly zeros from a small local model, and be careful not to read that as a
   harness defect.
