# Privacy and data handling

`proto-harness` is a local-first tool. It has no telemetry, no contribution channel,
and no episode log. What it writes, it writes into the data directory described below,
and the only thing that leaves your machine is the model request itself.

## What leaves your machine

When you run a task, the request sent to the model contains what the model needs to
answer: the task text, the files named with `--file` or that the agent read, the
conversation so far, and tool results such as command output. Where it goes is the
routing decision, not a privacy setting:

- a **local** tier sends it to the local runtime (`127.0.0.1` by default);
- a **cloud** tier sends it to the provider you configured, under your API key.

Nothing else leaves. There is no analytics endpoint, no crash reporting, and no
background upload. The data directory itself is never transmitted anywhere.

## What is stored on disk

The data directory (`dataDir`) defaults to `<repo>/var`. Override it with
`PROTO_HOME` or the global `--data-dir` flag (`resolveDataDir()` in
`src/config/load.ts`). Everything below is relative to it.

| Path | Contents |
| --- | --- |
| `config.json` | Non-secret configuration only. It names the *environment variable* for an API key, never the key itself |
| `secrets.json` | Optional API keys written by `proto config set-key`, mode `0600`. Environment variables take precedence |
| `sessions/<id>.json` | `proto code` session transcripts — see below |
| `index/<hash>.jsonl` | The codebase index cache: file paths, `mtime`/size, symbol definitions and reference identifier names. It stores *names*, not file bodies |
| `tmp/syntax-*.*` | Transient scratch files used for JS/TS syntax checks, deleted in a `finally` block immediately after the check. A crash mid-check can leave one behind |

`proto run`, `proto route` and `proto doctor` write nothing to the data directory. The
only ongoing writer is the interactive agent.

### Session transcripts

`proto code` saves one JSON file per session after every completed turn and again on
exit. A record contains:

- the conversation: your messages, the model's replies, every tool call and its result
  (file contents the agent read, command output, search results);
- the workspace path, the model and provider in use;
- token counts, estimated spend, edited files, commands run.

**Transcripts are not redacted.** There is no redactor in the harness, and the file
stores what the agent actually saw. If the agent reads a `.env`, a config file with a
token, or a file containing customer data, that text is in the session file, and it was
also in the request sent to whichever provider answered. Treat `sessions/` as
sensitive, the same way you would treat your shell history.

`--no-save` disables it for a run: `proto code --no-save` writes no transcript, and
`/save` refuses while the flag is set. `/sessions` lists recent sessions from disk;
`--resume` continues the most recent one.

There is **no retention policy and no prune command**. Sessions accumulate until you
delete them: `rm -rf <dataDir>/sessions` removes every transcript, or remove individual
`<id>.json` files. This is a deliberate simplification, not an oversight — an automatic
retention window that silently drops data is its own failure mode, and the directory is
small enough to inspect.

## What the threat model does not protect against

This is a local tool, not a hardened secret store.

| Not protected against | Mitigation |
| --- | --- |
| Anything the agent read being in a transcript (secrets, proprietary code, personal data) | Use `--no-save`; run sensitive work in a throwaway `PROTO_HOME`; do not point the agent at credential files |
| A compromised machine, or another local process with read access to the data dir | Keep the data dir outside synced/backed-up folders; use `--data-dir` on an encrypted volume |
| Backups, Time Machine, or a cloud-synced folder copying `var/` | Store the data dir elsewhere, or delete sessions after use |
| A cloud provider retaining the request content | Keep work that must not leave on the local tier (`--local`, `PROTO_DISABLE_CLOUD=1`), or do not configure a key |
| A shared or multi-user machine reading `secrets.json` | Prefer environment variables over `proto config set-key`; the file is `0600` but not encrypted |
| The index cache revealing repository structure | It is names and paths only, but delete `index/` if the file names themselves are sensitive |

## Quick audit

```bash
./bin/proto config path          # .../var/config.json — the data dir is its parent
ls var/sessions                  # transcript files, if any
./bin/proto code --no-save       # a session that writes nothing
```

There is nothing else to audit: no episodes, no datasets, no outbox, no adapter
directory, no logs beyond the transient syntax scratch files.
