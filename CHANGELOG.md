# Changelog

Notable changes, newest first. This project does not yet promise semantic versioning;
it promises that breaking changes are written down here.

## 0.1.0

First version worth using.

**The harness**
- Two-tier routing: a heuristic policy decides per task between a local model and a
  cloud model, with quality floors, hard-locked task classes and cost/latency comparison.
- Verification decides success, not the model's own confidence. An edit is applied only
  if it passes: anchors resolve, the file still parses, no forbidden patterns.
- Escalation on a failed local attempt, bounded so a routing mistake costs cents.
- `proto code`: an interactive agent with an approval gate on every write and command.
- A codebase index — symbols, references, a ranked repository map — so a large
  repository fits in a small context window. 26 languages, no parser dependency.
- An agentic benchmark suite and runner (`bench/`, `scripts/bench.py`).

**The fast path**
- Self-hosted OpenAI-compatible endpoints: keyless operation, explicit pricing, and
  `proto doctor --probe-cloud` to prove one answers.
- Reasoning models supported. Thinking is a first-class toggle, and turning it off was
  measured at ~8x on a short answer.
- `proto run --split`: a cloud model plans, a local model writes the edits, and the
  result goes through the same verifier. Measured warm at ~1.8x the latency of using the
  remote endpoint for both roles, and chosen instead for being unmetered and offline.

**Not in this version**
- No learning loop. An earlier design logged episodes and fine-tuned a local model from
  them; it was removed rather than repaired, because the trainer targeted a different
  model than anyone had downloaded and the corpus never exceeded single digits.
- No measured comparison against SWE-bench or Terminal-Bench. `bench/README.md` explains
  why, and what the ten hand-written tasks do and do not tell you.
