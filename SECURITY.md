# Security

## Reporting

Open a private security advisory on GitHub, or email the address in `package.json`.
Please do not open a public issue for anything exploitable.

## What this tool can do to you

`proto code` reads files in a workspace, writes files, and runs shell commands, with
your approval. That is the point of it. Three properties are worth understanding before
you point it at anything you care about.

**Approval is per action, and the risk is declared by the tool, not by the model.** A
read is always allowed; a write or a command prompts unless you pass `--yes` or have
remembered that kind of approval. `--yes` exists for scripted use and removes the only
thing standing between a confused model and your working tree. Use it in a container or
a scratch checkout, not in your home directory.

**Tool output is treated as data, not instructions.** File contents, command output and
search results routinely contain text that looks like a directive — a README saying
"run this", a comment saying "ignore previous instructions". The system prompt tells the
model to treat all of it as data and to report an injection attempt rather than comply.
That is a mitigation, not a guarantee.

**A remote model sees what you send it.** Whatever the agent reads into its context goes
to whichever provider you configured — including your source code. A local model keeps
it on your machine; a cloud model does not. That choice is yours and it is explicit in
the tier you select.

## The self-hosted endpoint

If you run `cloud.provider custom` against a server you host:

- **Plain HTTP is the default and it is not private.** Anything on the path can read
  the prompts and the code in them. Put TLS in front of it, or a tunnel, before it
  leaves a network you control.
- **`cloud.requiresKey false` means unauthenticated, not private.** It tells proto not
  to *send* a credential. Anyone who can reach the port can use the model.
- **Session transcripts are written to disk unredacted** by default. `--no-save` turns
  that off. If you are working with secrets, pass it.

## What is stored

See `docs/privacy.md`. In short: session transcripts, config, secrets, and a codebase
index cache, all under the data directory. No telemetry, no phone-home, no upload path.
