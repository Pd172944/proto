# Privacy and data handling

This harness records data about the tasks you run, because that record powers deferred,
nearly-free local RL (router training and optional LoRA tuning). The default posture is
*local-only, redacted at rest, and easy to audit*. This document states exactly what is
stored, where, what is redacted, which consents exist, and what the design does **not**
protect against.

Three rules from the source shape everything below: nothing is written until it has been
through the redactor (`src/memory/redact.ts`); sharing anything off the machine is
opt-in, off by default, and never automatic; and two *independent* consent decisions
exist, plus a third separate opt-in for raw text (`src/contrib/consent.ts`).

## What is recorded per episode

An episode is one task, end to end: routing features, the decision, every attempt
(local, repair, escalation), verification verdicts, usage, reward, and any feedback. The
schema is `src/memory/types.ts`; the writer is `src/harness/loop.ts`.

| Field | Source | Notes |
| --- | --- | --- |
| `id`, `ts` | loop | Monotonic ULID (sorts by creation time) and an ISO timestamp |
| `schemaVersion`, `harnessVersion`, `platform` | loop | `process.platform-process.arch`, e.g. `darwin-arm64` |
| `task` | `memory.storeTaskText` | Redacted task text, **capped at 4000 chars**. Absent when `storeTaskText` is false |
| `taskHash` | always | Short, deterministic hash of the **raw** task text — a fingerprint, not reversible, but it links identical tasks across runs |
| `systemPrompt` | `memory.storePrompts` | The shared preamble, redacted, capped at 4000 chars |
| `systemPromptHash` | always | Hash of the **raw** preamble |
| `features` | router | `TaskFeatures`: sizes, token estimates, file count, task class, flags (has tests, security wording, explanation-only, …) |
| `vector`, `vectorVersion` | router | The numeric feature vector the learned scorer consumes |
| `decision` | `decisionFromRoute` | Tier, reason(s), `pLocalSuccess`, difficulty, task class, exploration/forced/unverified flags, expected cost and latency, vetoes |
| `environment` | loop | Local model/context window/availability/**warm-or-cold**, cloud provider/model/price/availability, configured quality floor, verifier availability, routing mode, exploration epsilon |
| `attempts[]` | loop | One `AttemptRecord` per model call — see below |
| `outcome` | loop | Status, final tier, escalated, `localSucceeded` label, total cost, total latency, reward + reward version, behavior propensity |
| `feedback` | `proto feedback` | `signal`: `accept`/`reject`/`edit`, optional free-form `note`, timestamp. Stored in `feedback.jsonl` and merged on read |
| `consent` | loop | Snapshot at write time of `localTraining` and `globalShare` (so later consent changes are auditable) |
| `redaction` | loop | `applied`, per-rule `counts`, `charsRemoved` |
| `tags` | loop | Prompt version, so datasets built later can tell which prompts are comparable |

Each `AttemptRecord` contains:

| Field | Notes |
| --- | --- |
| `n`, `tier`, `source` | Attempt index; `local-tiny`/`local`/`cloud-cheap`/`cloud-strong`; `initial`/`repair`/`escalation` |
| `providerId`, `model` | Which provider and the model id that actually answered |
| `prompt` | Redacted prompt, **only when `memory.storePrompts` is on**, capped at `memory.maxPromptChars` (default 12000) |
| `promptHash` | Hash of the redacted, capped prompt when `storePrompts` is on; otherwise a hash of the raw prompt |
| `promptTokens`, `outputTokens`, `cachedInputTokens` | Usage numbers |
| `costUsd`, `latencyMs`, `finishReason` | Cost, wall-clock latency, and why generation stopped |
| `output` | Redacted candidate text, **only when `memory.storeTaskText` is on**, capped at `memory.maxOutputChars` (default 8000). When it is off, only `outputHash` is kept |
| `outputHash` | Hash of the redacted, capped output when `storeTaskText` is on; otherwise a hash of the raw output |
| `verification` | `passed`, `score`, `blockers`, `failedChecks` (check ids), `durationMs`; `null` when verification was skipped |
| `error` | Provider error text (including any hint) when the attempt failed |

There is no field for file contents beyond what appears inside the stored task text or
prompt, and `proto episodes show <id>` prints the full redacted record — the source
comment says that if it ever surprises you, that is a bug. One asymmetry: the
per-episode `consent` snapshot records `localTraining` and `globalShare` only.
`shareCode` is **not** part of the episode record; it is recorded later, in the
contribution bundle's manifest.

## Where the data lives

The data directory (`dataDir`) defaults to `<repo>/var`. Override it with the
`PROTO_HOME` environment variable or the global `--data-dir` flag
(`resolveDataDir()` in `src/config/load.ts`). Everything below is relative to it.

| Path | Contents |
| --- | --- |
| `config.json` | Non-secret configuration only. The config names an *environment variable* for the API key, never the key itself |
| `secrets.json` | Optional API keys written by `proto config set-key`, mode `0600`. Environment variables take precedence |
| `episodes/<YYYY-MM-DD>.jsonl` | Episode shards, append-only. Oversized days roll to `<date>.1.jsonl`, …; a day overflows to `<date>.overflow.jsonl` after 1000 shards. Rotation size is `memory.maxShardBytes` (8 MiB) |
| `feedback.jsonl` | Separately appended feedback records, merged into episodes on read |
| `datasets/sft.jsonl`, `datasets/dpo.jsonl`, `datasets/router.jsonl` | Written only by `proto datasets build --write` |
| `router/weights.json` | The learned router scorer, written by `proto train router` |
| `contrib/consent.json` | The consent record with change history |
| `contrib/CONSENT.md` | Human-readable consent receipt |
| `contrib/identity.json` | The rotating pseudonym salt and generation |
| `contrib/outbox/*.jsonl` | Staged contribution bundles, waiting for an explicit upload |
| `train/jobs.json`, `train/state.json` | Training job history and scheduler state |
| `train/data/<jobId>/`, `train/logs/` | Per-job MLX data and logs |
| `models/adapters/<mode>-<id>/` | Trained LoRA adapters |
| `venv/` | The MLX virtualenv, when you create one (see `docs/local-models.md`) |

`<repo>/var` is a good default because it is self-contained and easy to inspect; it is
also easy to accidentally commit or back up, which is why the retention and redaction
controls below exist.

## Redaction

Redaction runs **before anything is written to disk**, in the `recordText()` helper in
`src/harness/loop.ts`, called while building each attempt and the episode itself. It is
controlled by `memory.redact` (default `true`) and can be disabled with
`proto config set memory.redact false` or `PROTO_DISABLE_REDACTION=1` — supported but
discouraged. If redaction happened only at upload time, plaintext secrets would sit in
`var/episodes/*.jsonl` on your disk and any bug in the upload path could leak them.
Redacting at rest makes the worst case a less useful training record, not a leaked key.

The built-in rules (`REDACTION_RULES`) replace matches with a typed placeholder of the
form `[REDACTED:<kind>]`, layered: structural secrets first, then known credential
formats, then the entropy heuristic, then personal identifiers.

| Rule name | Catches | Placeholder |
| --- | --- | --- |
| `private-key-block` | `-----BEGIN … PRIVATE KEY-----` … `-----END … PRIVATE KEY-----` blocks | `[REDACTED:private-key]` |
| `openai-key` | `sk-`, `sk-proj-`, `sk-svcacct-` style keys, 16+ chars | `[REDACTED:api-key]` |
| `anthropic-key` | `sk-ant-…` keys, 20+ chars | `[REDACTED:api-key]` |
| `github-token` | `ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_` tokens | `[REDACTED:github-token]` |
| `github-pat` | `github_pat_…` fine-grained PATs | `[REDACTED:github-token]` |
| `gitlab-token` | `glpat-…` tokens | `[REDACTED:gitlab-token]` |
| `aws-access-key` | `AKIA…` / `ASIA…` access key ids | `[REDACTED:aws-key]` |
| `google-api-key` | `AIza…` keys | `[REDACTED:google-key]` |
| `slack-token` | `xoxb-`, `xoxp-`, `xoxa-`, `xoxr-`, `xoxs-` tokens | `[REDACTED:slack-token]` |
| `huggingface-token` | `hf_…` tokens | `[REDACTED:hf-token]` |
| `stripe-key` | `sk_live_`, `pk_live_`, `sk_test_`, `pk_test_` keys | `[REDACTED:stripe-key]` |
| `jwt` | Three-segment `eyJ…` JWTs | `[REDACTED:jwt]` |
| `authorization-header` | `Authorization: Bearer/Basic/Token <credential>`; keeps the scheme, removes the credential | `[REDACTED:bearer]` |
| `url-credentials` | `scheme://user:password@host`; keeps the scheme | `[REDACTED:url-credentials]@` |
| `secret-assignment` | `SOMETHING_KEY=value`, `api_secret: value`, `PASSWORD=…`, `token=…`, `credential=…`, `private_key=…`, `access_key=…`, `auth…=…`; keeps the key name, removes the value | `[REDACTED:secret-value]` |
| `dotenv-known-secret` | `.env`-style lines whose names start with `AWS`, `AZURE`, `GCP`, `GOOGLE`, `OPENAI`, `ANTHROPIC`, `STRIPE`, `TWILIO`, `SENDGRID`, `SLACK`, `GITHUB`, `GITLAB`, `HF`, `HUGGINGFACE`, `DATABASE_URL`, `DB`, `REDIS`, `MONGO`, `SMTP`, `JWT`, `SESSION`, `OAUTH`, `CLIENT` | `[REDACTED:env-value]` |
| `email` | Email addresses | `[REDACTED:email]` |
| `ipv4` | Any IPv4 address (ports preserved) | `[REDACTED:ip]` |
| `home-path` | `/Users/<name>/` and `/home/<name>/` | `[REDACTED:home]/` |
| `macos-udid` | Mac device ids shaped `XXXXXXXX-XXXXXXXXXXXXXXXX` | `[REDACTED:device-id]` |

Two synthetic counters can also appear:

- **`high-entropy`** — the entropy heuristic, scanning for tokens of 40+ characters from
  `[A-Za-z0-9+/=_-]` and replacing them only when Shannon entropy is ≥ 3.6 **and** the
  token mixes at least two of lowercase/uppercase/digits. This is deliberately the
  **noisiest rule**: base64 fixtures, long hashes, minified blobs and generated ids get
  eaten, which is why it is reported separately from the format hits.
- **`custom-pattern`** — your regexes from the top-level `redactionPatterns` config
  array, applied after the built-ins. Invalid regexes are skipped.

Redaction never does partial masking (no "keep the first four characters of the key"
path): a prefix plus a known service narrows an attacker's search enormously. Test it
yourself before trusting it:

```bash
proto episodes redact-check --text "API_KEY=sk-proj-abcdefghijklmnop1234 dev@example.com /Users/me/x"
```

```text
redacted output
API_KEY=[REDACTED:secret-value] [REDACTED:email] [REDACTED:home]/x

counts: openai-key=1 secret-assignment=1 email=1 home-path=1 (0 chars removed)
```

`charsRemoved` is `max(0, original − redacted)` and can be 0 when placeholders are longer
than what they replace. Add your own patterns with the config array
(`proto config set redactionPatterns '["INTERNAL-[0-9]{6}"]'`). Every episode records a
`redaction` report, and `proto episodes stats` aggregates the counts. Caps interact with
redaction: text is truncated to the cap first, then redacted, and truncation is recorded
on the result.

## Retention

Episodes are kept for `memory.retentionDays` (default `90`). Pruning is by whole day
shard, and `proto episodes prune` is a **dry run by default**:

```bash
proto episodes prune                       # lists the shards that would be deleted
proto episodes prune --days 30             # dry run with a different window
proto episodes prune --yes                 # actually deletes
```

Scope matters: `EpisodeStore.prune()` deletes old `episodes/*.jsonl` shards only — not
`feedback.jsonl`, `datasets/`, `router/weights.json`, `contrib/`, `train/` logs, or
adapters. Remove those explicitly if you want them gone. `memory.enabled=false` (or
`PROTO_DISABLE_MEMORY=1`) stops new episodes being written.

## The three consents

Conflating "improve my model" with "share my data" is the classic mistake in
"help improve the model" features, so this project keeps them separate. All three default
to **off**, and they are independent.

| Decision | Config key | CLI | Meaning |
| --- | --- | --- | --- |
| Private local training | `train.enabled` | `proto train enable` / `proto train disable`, or `proto contrib consent --local-training on` | Your episodes may fine-tune *your own* local model. No data leaves the machine |
| Global sharing | `contrib.enabled` | `proto contrib consent --global on` | Derived data may be contributed to a shared pool |
| Raw text sharing | `contrib.shareCode` | `proto contrib consent --share-code on` | Redacted task/output text may be *included* in a contribution; only meaningful with global sharing on |

Consent lives in `<dataDir>/contrib/consent.json`, with a bounded change history (last
100 changes) and an `updatedAt`. No file on disk means "never asked", which is not yes:
`loadConsent()` falls back to the current config values. `proto contrib consent …` writes
the record and a `CONSENT.md` receipt, and keeps `config.json` in sync. Turning sharing
off does **not** retroactively send anything already staged.

```bash
proto contrib consent                                  # show current consent
proto contrib consent --local-training on               # private fine-tuning only
proto contrib consent --global on                       # derived data sharing only
proto contrib consent --global on --share-code on       # also include redacted text
proto contrib consent --global off                      # revoke sharing
```

## What a contribution contains

`buildBundle()` in `src/contrib/index.ts` builds a bundle. By default (global sharing on,
`shareCode` off) it contains only derived data:

- **Router records** — the rounded feature vector, the verification label (`0`/`1`), the
  behaviour propensity, and the task class.
- **Preference records** — the fact that one tier was rejected and another chosen
  (`tiers: { rejected, chosen }`), the task class, and the feature vector.
- **Content-addressed ids** for server-side de-duplication (not reversible) and a
  **rotating pseudonym**.

Raw task text, file contents and model outputs are included **only** when `shareCode` is
on, and even then they have already been through the redactor at write time. The JSONL
starts with a manifest line:

```json
{
  "manifest": {
    "version": 1,
    "createdAt": "2026-01-01T00:00:00.000Z",
    "pseudonym": "3f9c1a...",
    "pseudonymGeneration": 2,
    "harnessVersion": "0.1.0",
    "platform": "darwin-arm64",
    "shareCode": false,
    "counts": { "router": 120, "preference": 40 },
    "redactionsApplied": 7,
    "note": "Derived routing features and verification labels only. No task text, file contents or model outputs are included."
  }
}
```

`redactionsApplied` is a transparency signal: the sum of redaction counts across the
source episodes, so a receiver can see how much was stripped. When `shareCode` is on,
`note` says so explicitly. Staging is capped at `contrib.maxBytesPerDay` (default 5 MiB)
per day.

## Pseudonym rotation

The device pseudonym is a salted hash, not a stable device id:

- A random salt is stored in `<dataDir>/contrib/identity.json`.
- `pseudonym = sha256(salt + ":" + hostname).slice(0, 16)`. The hostname is mixed in so
  two machines with the same salt do not collide, and it never leaves the machine in the
  clear.
- The salt rotates every `contrib.saltRotateDays` (default `7`; clamped to a minimum of
  one day so a config typo cannot rotate on every run). On rotation **the old salt is
  deleted** and a `generation` counter increments.
- `proto contrib rotate` forces a rotation immediately.

The honest framing, straight from `src/contrib/consent.ts`: this is **forward privacy /
linkability reduction**, not differential privacy. Contributions before and after a
rotation cannot be linked without the deleted salt, but no noise is added, the feature
vectors remain in each record, and your IP and timing are still visible to the endpoint.
Do not treat it as an anonymity guarantee.

## The upload path

Nothing is ever uploaded automatically. An upload must pass **four** independent guards
in `uploadBundle()`, each sufficient on its own to stop it: `globalShare` consent is on;
`contrib.endpoint` is non-empty; `contrib.requireConfirmation` (default `true`) is
satisfied by an explicit `--yes`; and the bundle is not empty.

The normal workflow stages first and sends second, so you can inspect what would leave:

```bash
proto contrib status          # consent, endpoint, cap, staged bundles
proto contrib preview         # build a bundle in memory; sends and stages nothing
proto contrib preview --live  # preview the worst case: shareCode forced on
proto contrib stage           # write a bundle into contrib/outbox/
proto contrib outbox          # list staged bundles
proto contrib upload --yes    # POST the bundle to contrib.endpoint
```

`proto contrib preview` shows the consent lines, a summary (record counts, bundle size,
pseudonym and generation, whether raw text is included, redactions applied) and a real
sample record, plus blockers if it cannot upload yet. `--live` forces `shareCode: true`
for that preview only.

`stageBundle()` writes to `<dataDir>/contrib/outbox/<timestamp>-<pseudonym>.jsonl`;
`uploadBundle()` POSTs it as `application/x-ndjson` with one retry. If the request fails,
the bundle stays in the outbox for inspection. Already-staged records are excluded from
later bundles by content-addressed id, so re-running stage/upload does not duplicate
them. The endpoint is entirely user-configured (`proto config set contrib.endpoint
https://…`, or `PROTO_CONTRIB_ENDPOINT`); there is no default collector and no telemetry.

## Threat model

This is a local-first tool with a redactor, not an anonymity system.

| Not protected against | Mitigation |
| --- | --- |
| A compromised machine: malware or an attacker with read access to the data dir sees episodes (and can disable redaction) | Keep the machine clean; put `PROTO_HOME` on an encrypted volume; `PROTO_DISABLE_MEMORY=1` |
| Other local processes, backups, Time Machine, or a cloud-synced folder reading the data dir | Store the data dir outside synced/backed-up paths, or shorten retention |
| A malicious or merely curious endpoint you configure: it sees IP, timing, bundle sizes, and any raw text when `shareCode` is on | Leave `contrib.endpoint` empty (the default); review `proto contrib preview --live`; enable global sharing only for an endpoint you trust |
| Redaction false negatives in a secret format none of the rules knows | Test with `proto episodes redact-check`; add `redactionPatterns`; keep `shareCode` off; `PROTO_DISABLE_MEMORY=1` for sensitive work |
| Re-identification from content even after redaction (unique identifiers, rare phrasing, proprietary code) | Keep `shareCode` off — the default contribution is features, labels and hashes, not content |
| Linkage across a pseudonym generation using stable features | Rotation limits linkability, but it is not differential privacy; do not rely on it against a determined analyst |
| Secrets leaking from the environment some other way | API keys are read from env vars or `secrets.json` (0600), never written into `config.json` |

Practical hardening checklist:

```bash
export PROTO_DISABLE_MEMORY=1                       # no episodes written at all
export PROTO_HOME=/Volumes/Encrypted/proto-data     # keep state on an encrypted volume
proto config set memory.storeTaskText false
proto config set memory.storePrompts false
proto config set memory.retentionDays 7
proto episodes prune --yes
proto contrib consent --global off --share-code off
```

The last line is the default state: never enabling global sharing is the strongest
available protection, and it costs only the shared-pool feature.

## Quick audit

```bash
proto config path                          # where the data dir is
proto config get memory                     # what is recorded
proto episodes ls --limit 5                # recent episodes
proto episodes show <id>                   # the full redacted record
proto episodes stats                       # counts, redaction totals, success rates
proto episodes redact-check --text "…"     # test the redactor
proto episodes prune                       # dry-run retention check
proto contrib status                       # consent + staged bundles
proto contrib preview --live               # worst-case view of a contribution
```
