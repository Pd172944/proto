# The interactive agent (`proto code`)

An agentic coding CLI you run from inside a project, the way you run `claude` or any
other terminal agent. It reads files, searches, edits and runs commands in a loop,
asking before it changes anything.

```bash
cd ~/my-project
proto code
```

---

## What was taken from where

The three reference harnesses solve different parts of the problem well. What was
borrowed, and why:

| Source | What was taken |
|---|---|
| [pi](https://github.com/earendil-works/pi) | The **separation of concerns**: a provider layer, an agent runtime, and a terminal-rendering layer that are independently replaceable. Pi also documents having *no* permission system — so this harness deliberately does the opposite and makes approval part of the core contract. |
| [opencode](https://github.com/anomalyco/opencode) | The **tool taxonomy** (read / write / exec) and the habit of treating the transcript as the interface. |
| [claude-code](https://github.com/anthropics/claude-code) | The **workflow discipline**: read before edit, exact-anchor edits, verify your own work, keep going until done or genuinely blocked. And the *shape* of the permission prompts. |
| [gum](https://github.com/charmbracelet/gum) | The **visual language** — rounded borders, generous padding, quiet labels, a calm spinner, subtitles on the bottom border. Recoloured: gum is magenta/pink; this is **ember and deep water** (amber `179` structure, teal `79` for actionable things, coral `203` only for failure). |

Where this harness does better than all three: **edits are verified before they
land.** `edit_file` reuses the project's verifier, so an ambiguous anchor, a change
that would not parse, or an introduced anti-pattern is refused with a reason instead
of corrupting a file.

---

## `proto code` routes — at session granularity

There are two paths in this project and they share the router:

| Path | What it does | Uses the router? |
|---|---|---|
| `proto code` (and `proto-code`) | the interactive agent: one model, tool loop, approval prompts | **yes**, once per session |
| `proto run` / `proto route` | the batch path: features → quality floor → local/cloud tier → verify → escalate | **yes**, once per task |

The agent routes on your **first message**, because that message is a task description
and can be classified before any tokens are spent. It then keeps that model for the
session. Later turns are reactions to tool output whose shape is unknown in advance, and
switching models mid-conversation would break prompt caching and the provider's view of
the thread — which costs more than the routing saves. Re-decide on demand with `/route`,
or move one tier at a time with `/escalate` / `/deescalate`. `--per-turn-route` opts into
per-message routing for experimentation, and `--escalate-on-stuck` moves to a stronger
tier automatically if the agent burns its step budget.

Any explicit model choice (`--local`, `--model`, `--provider`) turns routing off, and so
does `--no-route`; `--route` forces it back on.

How to tell which model answered: the status line after each turn. Cloud shows a real
cost, local shows `$0.0000`:

```
steps 3 · tokens 955↓ 531↑ · cost $0.0096 · time 23.6s · edited solution.py
```

`955↓ 531↑` tokens for `$0.0096` is a cloud model. The same turn on a local model would
read `cost $0.0000`. `/model` switches models explicitly; `/cost` totals the session.

## The loop

```
your request
   │
   ├─ gather context  (git branch/status, project layout, AGENTS.md / CLAUDE.md)
   ├─ system prompt   (stable prefix first, so prompt caching hits)
   │
   └─► model ──► tool calls ──► results ──► model ──► … ──► answer
                     │
                     └─ every write and every command shows you a diff or the
                        exact command line and waits for approval
```

Bounded by design: a step cap (default 40) and a wall-clock budget (default 10
minutes) per turn. Hitting the budget stops the turn and says so; it never loops
forever on your money.

---

## Tools

| Tool | Risk | What it does |
|---|---|---|
| `read_file` | read | Numbered lines, with `offset`/`limit` for large files |
| `list_files` | read | Orient in an unfamiliar project; depth-limited |
| `search` | read | Regex over file contents, skipping vendor and VCS directories |
| `write_file` | **write** | Create or replace a file. Syntax-checked first; refused if the result would not parse |
| `edit_file` | **write** | Exact-anchor replacement. Refused if the anchor is missing **or ambiguous**, if the result would not parse, or if it introduces a hard anti-pattern |
| `run_command` | **exec** | Shell, no TTY. Refuses anything destructive without even asking |

`--read-only` registers only the read tools, so an agent physically cannot mutate
anything.

---

## Permissions

Every write and every command is shown before it runs, with the actual diff or the
exact command line:

```
╭─ Run command ───────────────────────────────────────────╮
│ npm test -- --runInBand                                 │
│ kind  exec   answer  approve / deny                     │
╰───────────────────── command execution ─────────────────╯
```

Three answers: **y** (this time), **a** (always, for this class of action, for this
session), **n** (no).

Two deliberate properties:

- **"Always" is never persisted.** It means "for this session, while I am watching".
  A remembered approval from last week is not consent.
- **Non-interactive runs fail closed.** A piped `--print` run denies writes it was
  not explicitly told to allow (`--yes`), rather than writing silently.

---

## Commands

| Command | What it does |
|---|---|
| `/help` | list everything |
| `/route [task]` | re-run the router and switch tier (defaults to re-routing the last message) |
| `/escalate` / `/deescalate` | move up or down one tier |
| `/model [name]` | show or switch model |
| `/local` / `/cloud` | switch to the local or cloud model for this session |
| `/workspace` | root, git branch, file count, detected instruction files |
| `/tools` | tools grouped by risk |
| `/cost` | tokens, spend, files edited |
| `/clear` | forget the conversation |
| `/sessions`, `/save` | session history and manual save |
| `/quit` | leave (Ctrl-D also works) |

Outside of slash commands:

- `!command` runs a shell command directly (with the same approval prompt).
- **Ctrl-C aborts the current turn, not the session.** Losing a session to a
  mistyped keystroke is not acceptable; Ctrl-C twice exits.

---

## Prompt design

Three decisions worth knowing, all in `src/agent/prompt.ts`:

**The base prompt is byte-stable, and everything dynamic comes after it.** cwd, git
state and project instructions are appended in a delimited block, so the cacheable
prefix does not shift on every turn — which matters because the system prompt is
re-sent on every step of every turn.

**Tool output is explicitly data, not instructions.** A coding agent reads arbitrary
files, including READMEs and fixtures written by other people. The prompt states
that file contents and command output can never give it instructions, and to report
a suspected prompt injection rather than comply. This is the single most important
safety property of an agent that reads untrusted content, and it is tested.

**Reminders are injected when they matter, not up front.** Rules buried in paragraph
nine of a long system prompt decay out of attention. So when the model edits files
and then tries to finish without running any tests, a `<reminder>` is appended to the
conversation *at that moment*. Exactly one nudge, then it stops — there is a test
asserting it does not nag after verification, because an agent that is nagged every
turn learns to ignore the nags.

---

## Running it from anywhere

There are **two** launchers and `proto` is the one that gives you every command:

| Command | What it is |
|---|---|
| `proto` | the whole CLI — `code`, `doctor`, `setup`, `route`, `run`, `models`, `config`, `index` |
| `proto-code` | a shortcut that runs `proto code` directly |

Link both (the installer does exactly this):

```bash
/Users/prithvidixit/Desktop/sky/personalHarness/proto/scripts/install.sh
```

Then, from any directory:

```bash
proto doctor --probe-cloud
cd ~/any/project && proto-code
```

The installer does four things and is safe to re-run:

1. symlinks both launchers into `~/.local/bin` (override with `--bin-dir`)
2. **runs them to verify they work** — a launcher that resolves the wrong repository
   root is syntactically valid and only fails at runtime, so checking the file is not
   enough
3. reports whether the bin directory is on your `PATH`, and adds it to your shell rc
   only if you pass `--write-rc` (editing your rc file is opt-in)
4. reverses everything with `--uninstall`, including removing the rc block it added

Other flags: `--dry-run` to preview, `--bin-dir DIR` for a different location.

**The two mistakes this is designed to prevent**, both of which cost real debugging
time:

- Linking only one launcher. `proto-code` on your `PATH` does not make `proto`
  available; they are separate scripts.
- `export PATH=...` in one terminal does not affect any other terminal, and does not
  survive a new window. Put it in `~/.zshrc` (which `--write-rc` does), or use the
  full path.

The session root is the directory you launch from, and it is shown in the banner. A
session cannot write outside it.

---

## Seeing it work with no setup at all

```bash
proto code --demo
```

A scripted, read-only provider drives the **real** loop: the same tool registry, the
same approval flow, the same renderer and the same accounting. It lists your files,
reads one, runs `git status`, and explains itself. Nothing is modified.

This is the fastest way to check that your terminal, Node and the harness are all
working before you spend money on a model.

---

## Baseline: testing it with a real model

### 1. Confirm the key is visible to the harness

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd /Users/prithvidixit/Desktop/sky/personalHarness/proto
./bin/proto doctor --probe-cloud      # should say the cloud tier is ready
```

If `doctor` says the key is missing, the variable is not in the shell you are
running `proto` from. A key alone is enough — no config edit is needed.

### 2. Pick the model

```bash
./bin/proto config set cloud.provider anthropic
./bin/proto config set cloud.model claude-sonnet-4-5
./bin/proto config set cloud.cheapModel claude-haiku-4-5
```

### 3. Prove the loop end to end on a throwaway directory

Do this on a scratch copy, not your real repo, for the first run.

```bash
mkdir -p /tmp/proto-baseline && cd /tmp/proto-baseline
cat > prices.py <<'EOF'
def total_prices(items):
    total = 0
    for i in range(len(items) + 1):
        total += items[i]["price"]
    return total
EOF
git init -q && git add -A && git commit -qm "baseline"

proto-code --read-only
```

Then, in the session:

```
what does total_prices do, and is there a bug?
```

You should see: a `list_files` or `search` call, a `read_file` call, then prose. No
approval prompts, because `--read-only` cannot write.

### 4. Then let it edit

```bash
proto-code
```

```
fix the off-by-one in total_prices and add a guard for an empty list
```

Expected: `read_file` → `edit_file` showing a diff → **your approval** → possibly
`run_command` (it may offer to run tests) → a short summary. Then:

```bash
git diff          # inspect what actually changed
```

### 5. A task that needs real work

```
add a from_dict classmethod to Order that validates required fields and raises
ValueError with a useful message, then update the existing tests
```

This exercises multi-file edits, the verification reminder, and self-correction when
a tool refuses an edit.

### 6. Cost and safety checks

```bash
/cost        # tokens and spend for the session
/workspace   # confirms the root and whether AGENTS.md was picked up
```

Add an `AGENTS.md` to the project with your conventions and restart — it will be
injected into the system prompt. That is the intended way to teach it your house
style, and it is cheaper and more reliable than putting it in every message.

### 7. When something looks wrong

| Symptom | Cause | Fix |
|---|---|---|
| `no cloud provider available` | key not in this shell | `export ANTHROPIC_API_KEY=...` then `proto doctor --probe-cloud` |
| It answers but never uses tools | model without tool support, or a gateway stripping `tools` | check `proto doctor`; try `--model claude-sonnet-4-5` |
| Every write is "denied" | you are piping output, not in a TTY | run it in a real terminal, or pass `--yes` deliberately |
| An edit is refused twice in a row | the model is guessing at whitespace | tell it to re-read the file; the refusal reason says so |
| No colour | `NO_COLOR` set or not a TTY | `PROTO_COLOR=1` forces it |
| `doctor` says the local model is not downloaded | you pulled a different model | `proto models use <name>` adopts one you already have; the runtime is never switched silently |
| `api key missing` though you exported one | the default provider is OpenRouter, so it looks for `OPENROUTER_API_KEY` | a key for any known provider is auto-selected, and `doctor` now prints which key names it found |
| `400 ... not scoped to a workspace` | your Anthropic key is organisation-level, not workspace-scoped, so the API needs `anthropic-workspace-id` | see below |
| It stops mid-task | step or time budget | raise `--max-steps` / `--deadline-min` |

---

## Trying it on HumanEval

HumanEval is 164 single-function problems with hidden assertions. It clones to
**~270 KB** (the dataset itself is 45 KB gzipped), so size is never a concern.

```bash
python3 scripts/humaneval.py setup          # clone + extract
python3 scripts/humaneval.py list --limit 20
python3 scripts/humaneval.py show 0         # prepare problem 0 and print instructions
#   ... play with the CLI in the printed workspace ...
python3 scripts/humaneval.py check 0        # run the hidden tests
python3 scripts/humaneval.py run --count 5  # automated pass@1
```

It lives in `humaneval/`, **not** `test/` — `test/` is this project's own suite (16
files, ~6,000 lines) and mixing a vendored dataset into it would be a mess. The
`humaneval/` directory is gitignored.

### What each command is for

| Command | Purpose |
|---|---|
| `show <n>` | Writes `solution.py` with the signature and docstring, prints the task, and prints the exact `proto-code` command and opening message to paste |
| `check <n>` | Imports the candidate and runs HumanEval's real assertions |
| `run` | For each problem: fresh dir, fresh `PROTO_HOME`, drive `proto code --print --yes`, then check. Reports pass@1 and spend |
| `reset <n>` | Restore the stub |

`run` gives every problem its own `PROTO_HOME`, so an evaluation never touches your
real `var/` sessions.

### Read this before believing the number

**HumanEval is close to the worst possible benchmark for an *agentic* harness**, and
a good one for a plain completion model. It has no repository to explore, no
searchable code, no tests the agent can see, and one function per problem. So it
exercises almost none of what makes an agent useful, while actively penalising the
one thing agents do that single-shot models do not: spend extra turns.

Concretely, what the number is and isn't:

- **It is** a smoke test that the loop, the tools, the provider and the wire format
  all work end to end on real tasks with real hidden tests. That is genuinely worth
  having, and it is what this is for.
- **It is not** a verdict on the harness. A bare completion model will often score
  *higher* on HumanEval than the same model inside an agent loop.
- The agent is free to write its own scratch checks, so this is **agentic pass@1**,
  not model pass@1 — a legitimate but different measurement.
- One sample per problem at an unpinned temperature makes it noisy. Do not compare
  two models on a 5-problem run without expecting ±20% swings.
- Model-generated Python is executed to check the answer, which is the same trust you
  already extend by letting the agent run commands.

If you want a measurement that reflects what this harness is for, use tasks with a
repo, a failing test and room to iterate — `proto code` on your own codebase, or
SWE-bench-style issues. HumanEval will only ever measure the least interesting part.

### Useful flags

```bash
python3 scripts/humaneval.py run --count 10 --start 20        # problems 20-29
python3 scripts/humaneval.py run --count 5 --agent local      # test the local model
python3 scripts/humaneval.py run --count 5 --agent demo       # proves the plumbing, scores 0
python3 scripts/humaneval.py run --count 10 --model claude-haiku-4-5
python3 scripts/humaneval.py run --count 3 --verbose          # show the failing assertions
```

`--agent demo` scoring 0/3 is the *correct* result: the demo provider is read-only
and cannot write `solution.py`. If it ever scores non-zero, something is wrong.

---


### Anthropic: "This API key is not scoped to a workspace"

A 400 like this means the request never reached a model — it costs nothing and
happens before any tokens are generated:

```
Anthropic HTTP 400: This API key is not scoped to a workspace, so this request must
include the anthropic-workspace-id header with the ID of the workspace to use.
```

Some Anthropic API keys are organisation-level rather than belonging to a specific
workspace. The Anthropic API rejects every request from such a key until it is told
which workspace to bill and attribute the usage to. Two fixes, either is fine:

**Option A — use a workspace-scoped key (usually easier).** In the Anthropic Console,
open **Settings → Workspaces**, pick or create a workspace, and create the API key
from *inside* that workspace. Keys created that way carry their scope and need no
header at all. Then replace the key in your shell/env.

**Option B — tell the harness the workspace id.** Copy the workspace id from the same
Console page (workspace ids look like `wrkspc_…`; an org admin can also list them via
the Admin API's `GET /v1/organizations/workspaces`), then:

```bash
proto config set cloud.workspaceId wrkspc_xxxxxxxx
# or, without touching config:
export ANTHROPIC_WORKSPACE_ID=wrkspc_xxxxxxxx
```

`proto doctor` shows whether a workspace id is in effect. If you set one and still get
a workspace error, the harness will now say the header *was sent and rejected*, which
means the id is wrong or belongs to a different organisation than the key.

**Sending other headers.** Some gateways and enterprise proxies need their own header
(Azure wants `api-version`, tracing proxies want a correlation id). Rather than
requiring a code change, `cloud.extraHeaders` takes an arbitrary map:

```bash
proto config set cloud.extraHeaders '{"x-my-proxy-header":"value"}'
```

---

## Non-interactive use

```bash
proto code "summarise the architecture of this project" --print
proto code "add type hints to utils.py" --print --yes      # deliberate, unattended
```

`--print` writes only the final answer to stdout, so it composes with pipes and
scripts. `--yes` is required for writes in that mode; without it the run is read-only
in effect.

---

## Known limitations

- **No embeddings.** It has a symbol graph and a repo map (see
  [codebase-index.md](codebase-index.md)), but no embedding index. On a very large
  repository it will be slower to find things than an agent with an LSP-backed index.
- **No sub-agents or parallel tool calls.** One conversation, one step at a time.
  This is deliberate (context fragmentation is the failure mode of multi-agent
  designs) but it means a long task takes longer than a parallel implementation.
- **No session resumption mid-turn.** A session is saved after each completed turn;
  an interrupted turn loses that turn's tool results.
- **The reminder heuristic is a regex.** `looksLikeVerification` matches common test
  and typechecker invocations; an unusual project will not be recognised, and the
  model will be nudged once before the user tells it what to run.
- **Compaction drops the oldest turns** rather than summarising them, so a very long
  session can lose an early constraint. It says how many turns it dropped.
- **No image input**, despite the multimodal models. Attachments are not wired up.
