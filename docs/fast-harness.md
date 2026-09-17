# The fast path: a self-hosted endpoint, and thinking turned off

How to run `proto` against a model you host yourself, and what actually makes it fast.

## The two levers, in order of size

**1. Turn thinking off.** This is the biggest single win and it is not obvious. A
reasoning model decides for itself how long to think, and on a latency budget it is
spending most of its tokens on text nobody reads. Measured against Qwen3.8-27B on the
same one-line prompt:

| | time to first token | total | reasoning tokens |
| --- | --- | --- | --- |
| thinking on (default) | 0.29s | **2.94s** | 119 |
| thinking off | 0.17s | **0.38s** | 0 |

Nearly 8x, for a task that did not need reasoning at all.

```bash
proto config set cloud.thinking false
```

Leave it unset and the endpoint decides. Turn it on for work that genuinely benefits —
a plan over an unfamiliar codebase, a tricky diagnosis — and off for everything
mechanical. `proto doctor` reports which way it is set.

**2. Put the endpoint near the model.** The difference between a hosted gateway and a
box on your own network is mostly round-trip time and queueing, not tokens per second.
Against the same model on OpenRouter, a four-step agent turn on a two-bug Python fix:

| route | wall clock | cost |
| --- | --- | --- |
| OpenRouter (free tier) | 55.8s | $0.00 |
| self-hosted vLLM, thinking off | **6.9–7.4s** | $0.00 |

## Pointing proto at a server you host

```bash
proto config set cloud.provider  custom
proto config set cloud.baseUrl   http://your-host:8000/v1
proto config set cloud.model     Qwen/Qwen3.8-27B
proto config set cloud.requiresKey false      # no credential on a box you own
proto config set cloud.thinking  false
proto doctor --probe-cloud                    # proves it answers, not just that it exists
```

`cloud.requiresKey false` matters more than it looks. Without it the tier is reported
unavailable and never used — and worse, `autoSelectCloudProvider` will silently
redirect every request to whichever hosted provider happens to have a key in
`secrets.json`, so the config file says one thing and the traffic goes somewhere else.

**Cost.** A keyless endpoint is priced at zero, because nobody is metering you per
token. If your endpoint *is* metered but keyless, say so explicitly:

```bash
proto config set cloud.price.in  0.5
proto config set cloud.price.out 1.5
```

Otherwise the pessimistic unknown-model fallback ($5/$20 per M) applies, and the router
will treat the fastest tier it has as the most expensive one.

## The split: the cloud plans, the local model types

`proto run --split` runs a different shape of turn. A strong model reads the task and
produces a plan — what to change and why, as intents, not code. A local model takes one
file and one instruction at a time and produces the actual edits. The assembled result
goes through the same verifier as a single-model answer.

```bash
proto run "fix the off-by-one in the pagination helper" --file src/page.ts --split
proto run "..." --split --apply          # write it, once verification passes
```

Why it can be faster: the executor's prompt is tiny. A full agent turn re-sends the
whole transcript every step — 13,000 tokens on the run above — while an executor step
sends one file and one sentence, measured at 343–456 tokens. And the steps are
independent, so they run concurrently (`--split-concurrency`, default 4).

That concurrency is not a nicety. A 9B local model at ~21 tok/s is *slow per token*, and
without several steps in flight it loses to a fast remote model outright. The split is
what makes the local model's economics (free, private, unmetered) usable in a latency
budget.

### What the split does not do

- **No tool loop, no exploration.** The planner sees the files you pass with `--file`
  and commits to a plan. It cannot go looking. If the task needs the agent to discover
  things, `proto code` is the right tool and this is not.
- **Retries the executor, not the planner.** A failed verification re-runs the
  executor for the offending file, because an ambiguous anchor is a local mistake with a
  local fix. The plan is reused, so a retry costs local time and no cloud time.
- **Form is not correctness.** See below.

## Verification, and what "passed" means

By default `verify.runTests` is off, so the verifier checks that an edit is *well
formed*: anchors resolve, the file still parses, no forbidden patterns, nothing
swallowed. It does not run your tests and therefore says nothing about whether the code
now works.

This is worth stating plainly because it produced a real failure here. A split run
reported `verification: passed (score 1.00)` and wrote a file in which one of two bugs
was still present. The edit it received was perfectly well formed; the second edit had
been silently dropped. The verdict string now says which of the two it checked, and the
CLI prints a reminder when behaviour was not verified. For anything that matters:

```bash
proto config set verify.runTests true
proto config set verify.testCommand "npm test"
```

## Production notes

- **Plain HTTP.** A self-hosted endpoint is typically `http://`. That is fine on a
  trusted network and not fine across one you do not control; put TLS in front of it
  (or a tunnel) before it leaves your LAN. `cloud.extraHeaders` exists for a token if
  the server can enforce one.
- **Keyless means unauthenticated.** Anyone who can reach the port can use the model.
  `cloud.requiresKey false` tells proto not to *send* a credential; it does not make the
  server private.
- **32k context is the real limit.** The endpoint advertises `max_model_len`, and a
  long agent turn with several large files will hit it. `proto code` compacts the
  transcript near the limit; the split path bounds each executor prompt to roughly
  `local.contextWindow * 1.8` characters.
- **`stream_options`.** proto asks for token usage on streamed responses, which most
  servers support. One that rejects the unknown field should set
  `cloud.streamUsage false`; otherwise streaming fails rather than reporting no usage.
- **Health is checked, not assumed.** `proto doctor --probe-cloud` makes a real request.
  `--split` probes both halves before doing any work, so a dead executor is not
  discovered after paying for a plan.

## What is measured, and what is not

Measured on a two-bug Python fixture, endpoint and executor on the same vLLM box:

| | |
| --- | --- |
| plain agent turn (`proto code`) | 6.9–7.4s, $0.00, ~0.5s local CPU |
| split (`proto run --split`) | 6.69s (2.12s planner, 4.57s executor incl. one retry) |

**Not measured:** the split with a genuinely local executor. Every number above used
the endpoint for both roles, because starting Ollama would have loaded a 9B model onto
the machine, and that was explicitly out of scope at the time. The design claim — that
a small local model in the executor role beats one in the agent loop because its prompts
are ~30x smaller and its steps are concurrent — is a reasoned prediction from the
prompt sizes, not a result. Run it with Ollama up before trusting it.
