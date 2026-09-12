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
| `/model [name]` | show or switch model |
| `/local` / `/cloud` | switch tier for this session |
| `/workspace` | root, git branch, detected instruction files |
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

The launcher needs to be on your `PATH` so you can `cd` into any project and start a
session there:

```bash
# one option: symlink it (no sudo, works for the current user)
mkdir -p ~/.local/bin
ln -sf /Users/prithvidixit/Desktop/sky/personalHarness/proto/bin/proto-code ~/.local/bin/proto-code
export PATH="$HOME/.local/bin:$PATH"     # add to ~/.zshrc to make it stick
```

Then:

```bash
cd ~/any/project && proto-code
```

If you would rather not touch your PATH, run it directly with the full path, or use
an alias:

```bash
alias pc='/Users/prithvidixit/Desktop/sky/personalHarness/proto/bin/proto-code'
```

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
| It stops mid-task | step or time budget | raise `--max-steps` / `--deadline-min` |

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

- **No repo-wide retrieval.** It searches and reads, but there is no embedding index
  or symbol graph. On a very large repository it will be slower to find things than
  an agent with an LSP-backed index.
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
