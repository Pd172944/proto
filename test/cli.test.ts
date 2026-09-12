/**
 * CLI integration tests.
 *
 * Everything else in the suite imports modules directly, which means the actual
 * entry point — argv handling, exit codes, JSON output, config loading — had no
 * end-to-end coverage. Exit codes in particular are load-bearing: a cron or
 * launchd agent reads them, so "declined to run" must not look like "failed".
 */

import { strict as assert } from 'node:assert';
import { execFile } from 'node:child_process';
import { join } from 'node:path';
import { describe, it } from 'node:test';
import { promisify } from 'node:util';

import { tempDir } from './helpers.ts';
import { writeJsonAtomic } from '../src/util/fsx.ts';
import { symlinkSync } from 'node:fs';
import { isAbsolute } from 'node:path';

const run = promisify(execFile);
const REPO = new URL('..', import.meta.url).pathname;

interface CliResult {
  code: number;
  stdout: string;
  stderr: string;
}

/**
 * Run the CLI in a fresh, isolated PROTO_HOME.
 *
 * `withCloudKey` writes a (fake) secrets.json so the router sees a usable cloud
 * tier. Without it every task is forced local, because a fresh install has no
 * key — which is correct behaviour but hides the routing decision under test.
 */
async function cli(
  args: string[],
  opts: { env?: Record<string, string>; withCloudKey?: boolean } = {},
): Promise<CliResult> {
  const home = tempDir();
  if (opts.withCloudKey) {
    writeJsonAtomic(join(home, 'secrets.json'), { openrouter: 'test-key-not-real' });
  }
  try {
    const { stdout, stderr } = await run(
      process.execPath,
      ['--experimental-strip-types', '--disable-warning=ExperimentalWarning', join(REPO, 'src', 'cli.ts'), ...args],
      {
        cwd: REPO,
        env: { ...process.env, PROTO_HOME: home, PROTO_LOG: 'silent', ...(opts.env ?? {}) },
        timeout: 60_000,
        maxBuffer: 8 * 1024 * 1024,
      },
    );
    return { code: 0, stdout, stderr };
  } catch (err) {
    const e = err as { code?: number; stdout?: string; stderr?: string };
    return { code: typeof e.code === 'number' ? e.code : 1, stdout: e.stdout ?? '', stderr: e.stderr ?? '' };
  }
}

describe('cli: launchers actually execute', () => {
  /**
   * These tests exist because of a real bug: `bin/proto-code` computed the repository
   * root as `<repo>/bin` instead of `<repo>`, so it looked for `<repo>/bin/src/cli.ts`
   * and died with a Node MODULE_NOT_FOUND.
   *
   * `bash -n` cannot catch that — it is syntactically valid — and the mistake is
   * invisible when the script is run from inside the repo, which is why the bug
   * survived until a user tried it from a different directory. Hence: execute the real
   * scripts, from a directory that is not the repo, and through a symlink.
   */
  const runLauncher = async (script: string, args: string[], opts: { cwd: string }): Promise<CliResult> => {
    // `join(REPO, absolutePath)` would silently mangle an absolute path into
    // REPO + path, which is how this helper first "failed" the symlink case.
    const full = isAbsolute(script) ? script : join(REPO, script);
    try {
      const { stdout, stderr } = await run(full, args, {
        cwd: opts.cwd,
        env: { ...process.env, PROTO_HOME: tempDir(), PROTO_LOG: 'silent', NO_COLOR: '1' },
        timeout: 60_000,
        maxBuffer: 4 * 1024 * 1024,
      });
      return { code: 0, stdout, stderr };
    } catch (err) {
      const e = err as { code?: number; stdout?: string; stderr?: string };
      return { code: typeof e.code === 'number' ? e.code : 1, stdout: e.stdout ?? '', stderr: e.stderr ?? '' };
    }
  };

  it('bin/proto works when invoked by absolute path from an unrelated directory', async () => {
    // This is the invocation that broke: cwd has nothing to do with the repo.
    const res = await runLauncher('bin/proto', ['help'], { cwd: tempDir() });
    assert.equal(res.code, 0, res.stderr);
    assert.match(res.stdout, /two-regime coding harness|usage: proto/);
  });

  it('bin/proto-code works when invoked by absolute path from an unrelated directory', async () => {
    const res = await runLauncher('bin/proto-code', ['--help'], { cwd: tempDir() });
    assert.equal(res.code, 0, res.stderr);
    assert.match(res.stdout, /interactive coding agent/);
  });

  it('bin/proto-code works through a symlink on PATH', async () => {
    // The documented install pattern, and the one that exercises the launcher's
    // symlink-following loop.
    const bin = tempDir();
    const link = join(bin, 'proto-code');
    symlinkSync(join(REPO, 'bin', 'proto-code'), link);
    const res = await runLauncher(link, ['--help'], { cwd: tempDir() });
    assert.equal(res.code, 0, res.stderr);
    assert.match(res.stdout, /interactive coding agent/);
  });

  it('bin/proto works through a symlink on PATH', async () => {
    // `proto` and `proto-code` are separate launchers, and linking only one of them
    // produces a confusing "command not found" for the other. Both paths are covered.
    const bin = tempDir();
    const link = join(bin, 'proto');
    symlinkSync(join(REPO, 'bin', 'proto'), link);
    const res = await runLauncher(link, ['help'], { cwd: tempDir() });
    assert.equal(res.code, 0, res.stderr);
    assert.match(res.stdout, /usage: proto/);
  });

  it('bin/proto-code resolves its root correctly from a relative invocation', async () => {
    const res = await runLauncher('bin/proto-code', ['--help'], { cwd: REPO });
    assert.equal(res.code, 0, res.stderr);
    assert.match(res.stdout, /interactive coding agent/);
  });

  it('both launchers are executable and carry a shebang', async () => {
    const { readFileSync, statSync } = await import('node:fs');
    for (const script of ['bin/proto', 'bin/proto-code']) {
      const full = join(REPO, script);
      assert.match(readFileSync(full, 'utf8').split('\n')[0] ?? '', /^#!.*bash/, `${script} needs a bash shebang`);
      assert.ok((statSync(full).mode & 0o111) !== 0, `${script} must be executable (chmod +x)`);
    }
  });
});

describe('cli: basics', () => {
  it('prints help and exits 0', async () => {
    const res = await cli(['help']);
    assert.equal(res.code, 0);
    assert.match(res.stdout, /two-regime coding harness/);
    for (const cmd of ['doctor', 'setup', 'route', 'run', 'episodes', 'train', 'eval', 'replay', 'contrib']) {
      assert.ok(res.stdout.includes(cmd), `help must list "${cmd}"`);
    }
  });

  it('errors clearly on an unknown command and suggests a near match', async () => {
    const res = await cli(['rout', 'x']);
    assert.equal(res.code, 2);
    assert.match(res.stderr, /unknown command: rout/);
    assert.match(res.stderr, /did you mean `proto route`/);
  });

  it('rejects an unknown flag instead of silently ignoring it', async () => {
    // A silently ignored `--aply` would look like a successful dry run while the
    // user believed their file had been written.
    const res = await cli(['run', 'do something', '--aply']);
    assert.equal(res.code, 2);
    assert.match(res.stderr, /unknown flag: --aply/);
  });

  it('reports its version', async () => {
    const res = await cli(['--version']);
    assert.equal(res.code, 0);
    assert.match(res.stdout.trim(), /^\d+\.\d+\.\d+$/);
  });
});

describe('cli: routing and evaluation', () => {
  it('routes a task and emits parseable JSON', async () => {
    const res = await cli(['route', 'Fix the off-by-one error in this loop so it does not go out of bounds', '--json', '--offline']);
    assert.equal(res.code, 0);
    const parsed = JSON.parse(res.stdout) as { ok: boolean; decision: { tier: string; pLocalSuccess: number } };
    assert.equal(parsed.ok, true);
    assert.ok(parsed.decision.tier.startsWith('local'), `expected a local tier, got ${parsed.decision.tier}`);
    assert.ok(parsed.decision.pLocalSuccess > 0.5);
  });

  it('sends an architecture task to the cloud when a cloud tier exists', async () => {
    const res = await cli(
      ['route', 'Design the module boundaries between billing and accounts', '--json', '--offline'],
      { withCloudKey: true },
    );
    const parsed = JSON.parse(res.stdout) as { decision: { tier: string; vetoes: string[] } };
    assert.ok(parsed.decision.tier.startsWith('cloud'), `got ${parsed.decision.tier}`);
    assert.ok(parsed.decision.vetoes.some((v) => v.includes('hard-locked')));
  });

  it('forces the local tier when no cloud key is configured, and says why', async () => {
    // Refusing to work is worse than attempting local: the user gets an answer
    // plus an explicit explanation of what is missing.
    const res = await cli(['route', 'Design the module boundaries between billing and accounts', '--json', '--offline']);
    const parsed = JSON.parse(res.stdout) as { decision: { tier: string; forced: boolean; vetoes: string[] } };
    assert.ok(parsed.decision.tier.startsWith('local'));
    assert.equal(parsed.decision.forced, true);
    assert.ok(parsed.decision.vetoes.some((v) => v.includes('hard-locked')));
    assert.ok(parsed.decision.vetoes.some((v) => v.includes('cloud tier unavailable')));
  });

  it('scores the eval corpus above the regression floor', async () => {
    const res = await cli(['eval', 'run', '--json']);
    assert.equal(res.code, 0);
    const parsed = JSON.parse(res.stdout) as { report: { metrics: { accuracy: number; confusion: { cloud: { local: number } } } } };
    assert.ok(parsed.report.metrics.accuracy >= 0.9, `accuracy regressed to ${parsed.report.metrics.accuracy}`);
    assert.equal(parsed.report.metrics.confusion.cloud.local, 0, 'no cloud task may be routed local');
  });

  it('runs a dry run that writes nothing and reports the decision', async () => {
    const res = await cli(['run', 'Rename the variable a to count', '--dry-run', '--json']);
    assert.equal(res.code, 0);
    const parsed = JSON.parse(res.stdout) as { result: { status: string; episode: unknown; writtenFiles: string[] } };
    assert.equal(parsed.result.status, 'dry-run');
    assert.equal(parsed.result.episode, null);
    assert.deepEqual(parsed.result.writtenFiles, []);
  });

  it('replays an empty log without failing', async () => {
    const res = await cli(['replay', '--json']);
    assert.equal(res.code, 0);
    const parsed = JSON.parse(res.stdout) as { report: { episodes: number; notes: string[] } };
    assert.equal(parsed.report.episodes, 0);
    assert.ok(parsed.report.notes.some((n) => /no model calls/.test(n)));
  });
});

describe('cli: exit-code contract', () => {
  it('exits 0 when a tick correctly declines to train', async () => {
    // This is the normal outcome: the gates are doing their job. A cron agent
    // must not read it as a failure.
    const res = await cli(['train', 'tick']);
    assert.equal(res.code, 0, res.stderr);
  });

  it('exits 0 for a dry-run plan', async () => {
    const res = await cli(['train', 'plan']);
    assert.equal(res.code, 0, res.stderr);
  });

  it('exits non-zero when a forced session cannot start', async () => {
    const res = await cli(['train', 'now']);
    assert.equal(res.code, 1, `expected 1, got ${res.code}`);
  });

  it('exits non-zero when a task cannot be completed', async () => {
    // No local runtime and no cloud key in a fresh PROTO_HOME, so the run cannot
    // produce a verified answer; the CLI must say so in its exit code.
    const res = await cli(['run', 'Fix the off-by-one error in this loop']);
    assert.equal(res.code, 3, `expected the "failed" code 3, got ${res.code}`);
  });
});

describe('cli: introspection commands', () => {
  it('doctor reports readiness and missing pieces as JSON', async () => {
    const res = await cli(['doctor', '--json']);
    assert.equal(res.code, 0);
    const parsed = JSON.parse(res.stdout) as { ok: boolean; report: Record<string, unknown>; nextSteps: string[] };
    assert.equal(parsed.ok, true);
    assert.ok(parsed.report['environment']);
    assert.ok(parsed.report['local']);
    assert.ok(parsed.report['cloud']);
    assert.ok(Array.isArray(parsed.nextSteps));
  });

  it('setup prints instructions and downloads nothing', async () => {
    const res = await cli(['setup', '--json']);
    assert.equal(res.code, 0);
    const parsed = JSON.parse(res.stdout) as { commands: Record<string, string[]>; downloaded: unknown };
    assert.equal(parsed.downloaded, null, 'setup must not download by default');
    assert.ok(parsed.commands['pull']?.length);
  });

  it('setup refuses a download without a separate --yes', async () => {
    const res = await cli(['setup', '--download']);
    assert.equal(res.code, 0);
    assert.match(res.stdout, /refusing to download/);
  });

  it('explains the training gates truthfully', async () => {
    const res = await cli(['train', 'status', '--json']);
    assert.equal(res.code, 0);
    const parsed = JSON.parse(res.stdout) as { status: { gates: { allowed: boolean; checks: Array<{ ok: boolean; detail: string }> } } };
    assert.equal(parsed.status.gates.allowed, false);
    assert.ok(parsed.status.gates.checks.length >= 8, 'every gate must be reported');
    assert.ok(parsed.status.gates.checks.every((c) => c.detail.length > 0), 'every gate must explain itself');
  });

  it('previews a contribution and refuses to upload it', async () => {
    const res = await cli(['contrib', 'preview', '--json']);
    assert.equal(res.code, 0);
    const parsed = JSON.parse(res.stdout) as { preview: { wouldUpload: boolean; uploadBlockers: string[] } };
    assert.equal(parsed.preview.wouldUpload, false);
    assert.ok(parsed.preview.uploadBlockers.length > 0);
  });

  it('shows the redactor working, with counts', async () => {
    const res = await cli([
      'episodes',
      'redact-check',
      '--text',
      'API_KEY=sk-proj-abcdefghijklmnop1234 mail me at dev@example.com',
      '--json',
    ]);
    assert.equal(res.code, 0);
    const parsed = JSON.parse(res.stdout) as { text: string; counts: Record<string, number> };
    assert.ok(!parsed.text.includes('sk-proj-abcdefghijklmnop1234'));
    assert.ok(!parsed.text.includes('dev@example.com'));
    assert.ok(Object.keys(parsed.counts).length > 0);
  });
});

/**
 * `--print` is the scripted interface, not the TUI: it must hand the caller the model's
 * text exactly once, unwrapped and unstyled, so a shell can pipe it into a file or a
 * diff without stripping rail markers out first.
 */
describe('proto code --print is scriptable', () => {
  it('prints the final answer once, with no rail markers and no truncation', async () => {
    const res = await cli(['code', '--demo', '--no-save', 'show me what this harness does'], {
      env: { COLUMNS: '56', PROTO_COLOR: '0' },
    });
    assert.equal(res.code, 0, res.stderr);

    const marker = 'That was the demo provider';
    assert.equal(res.stdout.split(marker).length - 1, 1, `answer repeated or missing:\n${res.stdout}`);
    assert.ok(!res.stdout.includes('| That was'), 'print mode leaked the TUI rail');
    // Tool chatter belongs on stderr: `proto code --print ... > answer.md` must not get
    // `+ read_file` lines spliced into the file.
    assert.ok(!res.stdout.includes('read_file'), `chrome leaked into stdout:\n${res.stdout}`);
    assert.ok(res.stderr.includes('read_file'), 'tool chatter should still be visible on stderr');
    // Long lines survive: wrapping is a presentation choice, and this path has no terminal.
    assert.ok(
      res.stdout.split('\n').some((l) => [...l].length > 56),
      'print mode should not re-wrap the answer for a terminal it does not have',
    );
  });
});
