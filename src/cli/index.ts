/**
 * `proto` CLI.
 *
 * Conventions that hold for every command:
 *  - Human output goes to stdout via the returned `human` lines; logs go to
 *    stderr. `--json` replaces the human lines with a single JSON document, so
 *    the CLI composes with `jq` and with scripts.
 *  - Commands never mutate anything outside `var/` unless a flag says so
 *    (`--apply`, `--yes`, `--write`). The dangerous flags are named explicitly
 *    and are never implied by another flag.
 *  - A non-zero exit code means "the command could not do what you asked",
 *    not "the model failed". Model failures are data.
 */

import { flagBool, flagList, flagNumber, flagString, parseArgs, renderFlagHelp } from '../util/args.ts';
import type { FlagSpec, ParsedArgs } from '../util/args.ts';
import { log, out, style } from '../util/log.ts';
import { loadConfig } from '../config/load.ts';
import type { LoadedConfig } from '../config/load.ts';
import type { ProtoConfig } from '../config/schema.ts';

export interface CommandContext {
  cfg: ProtoConfig;
  dataDir: string;
  configPath: string;
  warnings: string[];
  positionals: string[];
  args: ParsedArgs;
  json: boolean;
}

export interface CommandResult {
  human: string[];
  json: unknown;
  exitCode?: number;
}

export interface Command {
  name: string;
  summary: string;
  usage: string;
  flags: FlagSpec[];
  run(ctx: CommandContext): Promise<CommandResult>;
}

const GLOBAL_FLAGS: FlagSpec[] = [
  { name: 'json', type: 'boolean', description: 'emit machine-readable JSON on stdout' },
  { name: 'data-dir', type: 'string', description: 'override the state directory (default: <repo>/var)' },
  { name: 'log', type: 'string', description: 'log level: debug|info|warn|error|silent' },
  { name: 'help', type: 'boolean', alias: 'h', description: 'show help for the command' },
];

export async function main(argv: string[]): Promise<number> {
  const commands = await loadCommands();
  const byName = new Map(commands.map((c) => [c.name, c]));

  const first = argv[0];
  if (!first || first === 'help' || first === '--help' || first === '-h') {
    // Both bare `proto` and `proto help` get the banner: a bare invocation is
    // usually a first run, and `help` is an explicit request for orientation.
    printTopLevelHelp(commands, true);
    return 0;
  }
  if (first === '--version' || first === '-v' || first === 'version') {
    const { HARNESS_VERSION } = await import('../version.ts');
    out(HARNESS_VERSION);
    return 0;
  }

  const command = byName.get(first);
  if (!command) {
    const suggestion = closestMatch(first, [...byName.keys()]);
    process.stderr.write(`unknown command: ${first}\n`);
    if (suggestion) process.stderr.write(`did you mean \`proto ${suggestion}\`?\n`);
    process.stderr.write(`run \`proto help\` for the command list\n`);
    return 2;
  }

  const parsed = parseArgs(argv.slice(1), [...command.flags, ...GLOBAL_FLAGS]);
  if (parsed.errors.length > 0) {
    process.stderr.write(parsed.errors.join('\n') + '\n\n');
    process.stderr.write(`usage: ${command.usage}\n\n${renderFlagHelp(command.flags)}\n`);
    return 2;
  }
  if (flagBool(parsed, 'help')) {
    out(`proto ${command.name} — ${command.summary}`);
    out('');
    out(`usage: ${command.usage}`);
    out('');
    out(renderFlagHelp(command.flags));
    return 0;
  }

  const logLevel = flagString(parsed, 'log');
  if (logLevel) process.env['PROTO_LOG'] = logLevel;

  let loaded: LoadedConfig;
  try {
    loaded = loadConfig({
      ...(flagString(parsed, 'data-dir') ? { dataDir: flagString(parsed, 'data-dir') as string } : {}),
    });
  } catch (err) {
    process.stderr.write(`failed to load config: ${err instanceof Error ? err.message : String(err)}\n`);
    return 1;
  }

  const json = flagBool(parsed, 'json');
  const ctx: CommandContext = {
    cfg: loaded.config,
    dataDir: loaded.dataDir,
    configPath: loaded.configPath,
    warnings: loaded.warnings,
    positionals: parsed.positionals,
    args: parsed,
    json,
  };

  try {
    const result = await command.run(ctx);
    if (json) {
      out(JSON.stringify(result.json, null, 2));
    } else {
      for (const line of result.human) out(line);
    }
    return result.exitCode ?? 0;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (json) {
      out(JSON.stringify({ ok: false, error: message }, null, 2));
    } else {
      log.error(message);
      if (err instanceof Error && err.stack && process.env['PROTO_LOG'] === 'debug') {
        process.stderr.write(err.stack + '\n');
      }
    }
    return 1;
  }
}

function printTopLevelHelp(commands: Command[], showBanner: boolean): void {
  if (showBanner) {
    out(style.bold('proto') + ' — two-regime coding harness (fast local model + cloud model)');
    out('');
    out('Routes each task to the tier that can handle it, verifies the result, escalates on failure,');
    out('and learns from what happened — locally, cheaply, and only with your consent.');
    out('');
  }
  out(style.bold('usage:') + ' proto <command> [options]');
  out('');
  const width = Math.max(...commands.map((c) => c.name.length)) + 2;
  for (const c of commands) {
    out(`  ${c.name.padEnd(width)}${c.summary}`);
  }
  out('');
  out(`global flags: ${GLOBAL_FLAGS.map((f) => `--${f.name}`).join(', ')}`);
  out('');
  out('start here:');
  out(`  ${style.bold('proto code')}              interactive coding agent in the current directory`);
  out('  proto doctor            check what is installed and what is missing');
  out('  proto setup             print the exact commands to download a local model (nothing runs)');
  out('  proto route "fix this off-by-one" --file src/a.py     see the routing decision');
  out('  proto run   "fix this off-by-one" --file src/a.py     run it (dry by default)');
}

function closestMatch(input: string, candidates: string[]): string | null {
  let best: string | null = null;
  let bestScore = Infinity;
  for (const c of candidates) {
    const d = levenshtein(input, c);
    if (d < bestScore) {
      bestScore = d;
      best = c;
    }
  }
  return bestScore <= 3 ? best : null;
}

function levenshtein(a: string, b: string): number {
  const dp = Array.from({ length: a.length + 1 }, (_, i) => [i, ...new Array<number>(b.length).fill(0)]);
  for (let j = 0; j <= b.length; j++) (dp[0] as number[])[j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      const row = dp[i] as number[];
      const prev = dp[i - 1] as number[];
      row[j] = Math.min((prev[j] as number) + 1, (row[j - 1] as number) + 1, (prev[j - 1] as number) + cost);
    }
  }
  return (dp[a.length] as number[])[b.length] as number;
}

/** Commands are loaded lazily so `proto --help` stays instant. */
async function loadCommands(): Promise<Command[]> {
  const [{ coreCommands }, { codeCommand }] = await Promise.all([
    import('./commands-core.ts'),
    import('./commands-code.ts'),
  ]);
  // `code` is listed first because for most users it is the whole product.
  return [codeCommand, ...coreCommands];
}

/* Re-exported for command modules. */
export { flagBool, flagList, flagNumber, flagString };
export type { FlagSpec, ParsedArgs };
