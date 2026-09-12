/**
 * `proto code` — the interactive coding agent.
 *
 * Design goals, taken from what the mature agents get right and made explicit:
 *
 *  - **One command, run from anywhere.** `cd` into a project, run `proto code`, and
 *    you are in a session rooted at that directory. No config file required.
 *  - **The transcript is the interface.** Tool calls are shown with their arguments
 *    *before* they run, so the user always knows what is being touched. Nothing
 *    mutating happens without an approval prompt showing the actual diff or command.
 *  - **Streaming, not stalling.** Text appears as it is generated; a spinner covers
 *    the gap before the first token and during tool execution. A slow model that
 *    shows nothing for 40 seconds feels broken even when it is working.
 *  - **Interruption is always available.** Ctrl-C aborts the *turn*, not the session,
 *    because losing a working session to a mistyped keystroke is infuriating.
 *  - **Graceful degradation.** Piped output, `NO_COLOR`, and `--print` all work: the
 *    renderer falls back to plain text and approvals either fail closed or are
 *    pre-granted by an explicit flag.
 */

import { createInterface } from 'node:readline';
import type { Interface as ReadlineInterface } from 'node:readline';
import { resolve } from 'node:path';

import { flagBool, flagNumber, flagString } from './index.ts';
import type { Command, CommandContext, CommandResult } from './index.ts';
import { buildCloudProvider, buildLocalProvider, cloudTierModel } from '../providers/index.ts';
import type { Provider } from '../providers/types.ts';
import { ProviderError } from '../providers/types.ts';
import { DemoProvider } from '../providers/demo.ts';
import { buildToolRegistry } from '../tools/files.ts';
import { ToolRegistry } from '../tools/types.ts';
import type { ApprovalRequest } from '../tools/types.ts';
import { runAgentTurn } from '../agent/loop.ts';
import type { AgentEvent } from '../agent/loop.ts';
import { Session, listSessions, loadSession, latestSession, saveSession } from '../agent/session.ts';
import { gatherProjectContext } from '../agent/prompt.ts';
import {
  approvalPanel,
  banner,
  box,
  codeBlock,
  diff,
  glyph,
  kv,
  markdownLite,
  palette,
  resetColorMode,
  rule,
  setColorEnabled,
  spinnerFrame,
  statusLine,
} from '../tui/theme.ts';
import { HARNESS_VERSION } from '../version.ts';
import { formatDuration, readTextOrNull } from '../util/fsx.ts';

/* ------------------------------------------------------------------ */
/* Rendering helpers                                                   */
/* ------------------------------------------------------------------ */

function termWidth(): number {
  const c = process.stdout.columns;
  return Math.min(120, Math.max(60, c && c > 0 ? c : 100));
}

function isTty(): boolean {
  return process.stdout.isTTY === true && process.stdin.isTTY === true;
}

/** A spinner that owns exactly one terminal line and always cleans up after itself. */
class Spinner {
  private timer: NodeJS.Timeout | null = null;
  private frame = 0;
  private active = false;
  private readonly label: () => string;

  constructor(label: () => string) {
    this.label = label;
  }

  start(): void {
    if (this.active || !isTty()) return;
    this.active = true;
    this.timer = setInterval(() => {
      this.frame = (this.frame + 1) % 10;
      process.stdout.write(`\r\x1b[2K${palette.accent(spinnerFrame(this.frame))} ${palette.dim(this.label())}`);
    }, 80);
  }

  /** Clear the spinner line so real output can take it over. */
  stop(): void {
    if (!this.active) return;
    this.active = false;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    if (isTty()) process.stdout.write('\r\x1b[2K');
  }
}

function summarizeToolArgs(name: string, args: unknown): string {
  const a = (typeof args === 'object' && args !== null ? args : {}) as Record<string, unknown>;
  const str = (k: string): string => (typeof a[k] === 'string' ? (a[k] as string) : '');
  switch (name) {
    case 'read_file':
      return `${str('path')}${a['offset'] ? `:${String(a['offset'])}` : ''}`;
    case 'list_files':
      return str('path') || '.';
    case 'search':
      return `/${str('pattern')}/`;
    case 'write_file':
      return str('path');
    case 'edit_file':
      return str('path');
    case 'run_command': {
      const c = str('command');
      return c.length > 72 ? `${c.slice(0, 69)}…` : c;
    }
    default:
      return Object.keys(a).slice(0, 3).join(', ');
  }
}

/** Tool-specific, human-scale argument preview (the interesting part of a write). */
function toolPreview(name: string, args: unknown, workspace: string): string | null {
  const a = (typeof args === 'object' && args !== null ? args : {}) as Record<string, unknown>;
  if (name === 'edit_file') {
    const path = typeof a['path'] === 'string' ? a['path'] : '';
    const oldS = typeof a['old_string'] === 'string' ? a['old_string'] : '';
    const newS = typeof a['new_string'] === 'string' ? a['new_string'] : '';
    const before = readTextOrNull(resolve(workspace, path));
    if (before === null) return null;
    const occurrences = before.split(oldS).length - 1;
    if (occurrences !== 1) return null;
    return diff(before, before.replace(oldS, newS), path, { context: 3, maxLines: 40 });
  }
  if (name === 'write_file') {
    const path = typeof a['path'] === 'string' ? a['path'] : '';
    const content = typeof a['content'] === 'string' ? a['content'] : '';
    const before = readTextOrNull(resolve(workspace, path));
    return before === null
      ? codeBlock(content, path.split('.').pop(), { maxLines: 40 })
      : diff(before, content, path, { context: 2, maxLines: 40 });
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Provider selection                                                  */
/* ------------------------------------------------------------------ */

interface ProviderChoice {
  provider: Provider;
  label: string;
  isLocal: boolean;
}

function chooseProvider(ctx: CommandContext, want: 'cloud' | 'local'): ProviderChoice {
  if (want === 'local') {
    const provider = buildLocalProvider(ctx.cfg);
    return { provider, label: `${ctx.cfg.local.runtime} · ${ctx.cfg.local.model}`, isLocal: true };
  }
  const provider = buildCloudProvider(ctx.cfg, ctx.dataDir, cloudTierModel(ctx.cfg, 'cloud-strong'));
  if (!provider) {
    throw new Error(
      `no cloud provider available. Set an API key (e.g. ANTHROPIC_API_KEY) or use --local with a running local runtime.`,
    );
  }
  return { provider, label: `${ctx.cfg.cloud.provider} · ${ctx.cfg.cloud.model}`, isLocal: false };
}

/* ------------------------------------------------------------------ */
/* The command                                                        */
/* ------------------------------------------------------------------ */

const SLASH_HELP: Array<[string, string]> = [
  ['/help', 'show this list'],
  ['/model [name]', 'show or switch the model'],
  ['/local', 'switch to the local model for this session'],
  ['/cloud', 'switch back to the cloud model'],
  ['/workspace', 'show the session root, git branch and file count'],
  ['/tools', 'list the available tools, grouped by risk'],
  ['/cost', 'tokens and estimated spend for this session'],
  ['/clear', 'forget the conversation (keeps the session and approvals reset)'],
  ['/sessions', 'list recent sessions on disk'],
  ['/save', 'write the transcript to the data directory now'],
  ['/quit', 'leave (Ctrl-D also works)'],
];

export const codeCommand: Command = {
  name: 'code',
  summary: 'interactive coding agent: reads, edits and runs commands in the current directory',
  usage: 'proto code ["one-shot prompt"] [--workspace dir] [--local] [--yes] [--print]',
  flags: [
    { name: 'workspace', type: 'string', description: 'project root (default: current directory)' },
    { name: 'local', type: 'boolean', description: 'use the local model instead of the cloud model' },
    { name: 'yes', type: 'boolean', description: 'auto-approve file writes and commands (non-interactive/dangerous)' },
    { name: 'read-only', type: 'boolean', description: 'refuse every write and command' },
    { name: 'print', type: 'boolean', description: 'one-shot: print the final answer and exit (no TUI)' },
    { name: 'max-steps', type: 'number', description: 'max tool/model steps per turn', default: 40 },
    { name: 'deadline-min', type: 'number', description: 'wall-clock budget per turn, minutes', default: 10 },
    { name: 'temperature', type: 'number', description: 'sampling temperature', default: 0.2 },
    { name: 'resume', type: 'boolean', description: 'continue the most recent session' },
    { name: 'no-color', type: 'boolean', description: 'disable colour output' },
    { name: 'demo', type: 'boolean', description: 'scripted read-only demo: see the full harness with no key and no model' },
  ],
  async run(ctx): Promise<CommandResult> {
    // Colour mode is decided by the theme module's auto rules (NO_COLOR, TTY,
    // PROTO_COLOR) unless the user asks explicitly. Forcing it on here would
    // override NO_COLOR, which is a standard the user has already expressed.
    if (flagBool(ctx.args, 'no-color')) setColorEnabled(false);
    else resetColorMode();

    const workspace = resolve(flagString(ctx.args, 'workspace') ?? process.cwd());
    const oneShot = ctx.positionals.join(' ').trim();
    const print = flagBool(ctx.args, 'print') || (!isTty() && oneShot.length > 0);
    const autoYes = flagBool(ctx.args, 'yes');
    const readOnly = flagBool(ctx.args, 'read-only');
    const interactive = isTty() && !print;

    const useDemo = flagBool(ctx.args, 'demo');
    const selectProvider = (want: 'cloud' | 'local', model?: string): ProviderChoice => {
      if (useDemo) return { provider: new DemoProvider(), label: 'demo · scripted', isLocal: false };
      if (want === 'local') return chooseProvider(ctx, 'local');
      if (model) {
        const provider = buildCloudProvider(ctx.cfg, ctx.dataDir, model);
        if (!provider) throw new Error(`no cloud provider available for model "${model}"`);
        return { provider, label: `${ctx.cfg.cloud.provider} · ${model}`, isLocal: false };
      }
      return chooseProvider(ctx, 'cloud');
    };

    let choice: ProviderChoice;
    try {
      choice = selectProvider(flagBool(ctx.args, 'local') ? 'local' : 'cloud');
    } catch (err) {
      return { human: [palette.error(err instanceof Error ? err.message : String(err))], json: { ok: false, error: String(err) }, exitCode: 1 };
    }

    const tools: ToolRegistry = readOnly ? readOnlyRegistry() : buildToolRegistry();

    const session = flagBool(ctx.args, 'resume') ? (latestSession(ctx.dataDir) ?? newSession(choice)) : newSession(choice);
    function newSession(c: ProviderChoice): Session {
      return new Session({ workspace, model: c.provider.model, provider: c.provider.id });
    }
    session.model = choice.provider.model;
    session.provider = choice.provider.id;

    const project = await gatherProjectContext(workspace);

    // ---------------------------------------------------------------- render
    const out = (s = ''): void => {
      process.stdout.write(`${s}\n`);
    };
    const err = (s: string): void => {
      process.stderr.write(`${s}\n`);
    };

    if (!print) {
      out(banner([
        `${palette.bold('proto code')}  ${palette.dim(HARNESS_VERSION)}`,
        palette.dim('a coding agent that reads, edits and runs commands'),
      ], termWidth()));
      out('');
      out(kv([
        { key: 'model', value: choice.label, note: choice.isLocal ? 'local' : 'cloud' },
        { key: 'workspace', value: workspace },
        ...(project.gitBranch ? [{ key: 'git', value: project.gitBranch, note: project.gitStatus ? 'dirty' : 'clean' }] : []),
        { key: 'tools', value: `${tools.list().length}`, note: readOnly ? 'read-only mode' : `${tools.byRisk().write.length} write, ${tools.byRisk().exec.length} exec` },
      ]));
      out('');
      out(palette.dim(`  /help for commands · Ctrl-C stops the current turn · ${readOnly ? 'read-only' : 'you will be asked before any write or command'}`));
      out('');
    }

    // ---------------------------------------------------------------- readline
    const rl: ReadlineInterface = createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: interactive,
      historySize: 500,
      prompt: `${palette.primary(glyph.arrow)} `,
    });

    let asking = false;
    const ask = (question: string): Promise<string> =>
      new Promise((res) => {
        asking = true;
        rl.question(question, (answer) => {
          asking = false;
          res(answer);
        });
      });

    let abortController: AbortController | null = null;
    let exitCode = 0;

    const onSigint = (): void => {
      if (abortController && !abortController.signal.aborted) {
        abortController.abort();
        process.stdout.write(`\n${palette.warn('interrupted')} ${palette.dim('(Ctrl-C again to exit)')}\n`);
        return;
      }
      if (asking) return;
      process.stdout.write(`\n${palette.dim('bye')}\n`);
      rl.close();
    };
    process.on('SIGINT', onSigint);

    /* ------------------------------------------------------------ approve */
    const approve = async (req: ApprovalRequest): Promise<'allow-once' | 'allow-always' | 'deny'> => {
      if (readOnly) return 'deny';
      // The demo provider is scripted and read-only by construction, so approving
      // its commands keeps the demo to one interaction. Writes are still shown and
      // would still prompt, which means the demo cannot quietly mutate anything.
      if (useDemo && req.kind !== 'write') return 'allow-once';
      const remembered = session.preApproved(req.kind);
      if (remembered) return remembered;
      if (autoYes) return 'allow-once';
      if (!interactive) {
        // Fails closed: a piped run must not silently write files.
        err(palette.warn(`  skipped (needs approval, non-interactive): ${req.title}`));
        return 'deny';
      }
      spinner.stop();
      out('');
      out(approvalPanel(req, termWidth()));
      const answer = (await ask(`${palette.primary('allow?')} ${palette.dim('[y]es  [a]lways for this kind  [n]o ')}`)).trim().toLowerCase();
      if (answer === 'a' || answer === 'always') {
        session.rememberApproval(req.kind);
        return 'allow-always';
      }
      if (answer === 'y' || answer === 'yes' || answer === '') return 'allow-once';
      return 'deny';
    };

    /* ------------------------------------------------------------ turn */
    const spinner = new Spinner(() => 'working…');

    const runTurn = async (input: string): Promise<void> => {
      abortController = new AbortController();
      const turnStart = Date.now();
      let sawDelta = false;
      let stepLabel = '';
      let turnCost = 0;
      let turnIn = 0;
      let turnOut = 0;

      spinner.start();

      const handle = (event: AgentEvent): void => {
        switch (event.type) {
          case 'context':
            break;

          case 'model-start':
            stepLabel = event.step === 1 ? 'thinking…' : `step ${event.step}…`;
            spinner.start();
            break;

          case 'text-delta':
            if (!sawDelta) {
              spinner.stop();
              out('');
              sawDelta = true;
            }
            process.stdout.write(event.text);
            break;

          case 'text':
            // Only print the settled text when nothing was streamed, otherwise the
            // user would see the whole answer twice.
            if (!sawDelta && !print) {
              spinner.stop();
              out('');
              out(markdownLite(event.text, termWidth() - 2));
              sawDelta = true;
            }
            break;

          case 'tool-start': {
            spinner.stop();
            const name = palette.accent(event.toolCall.name);
            const args = palette.dim(summarizeToolArgs(event.toolCall.name, event.toolCall.args));
            out(`  ${palette.primary(glyph.bullet)} ${name} ${args}`);
            break;
          }

          case 'tool-end': {
            const r = event.result;
            const mark = r.ok ? palette.success(glyph.check) : palette.error(glyph.cross);
            out(`    ${mark} ${palette.dim(r.title)}`);
            if (!r.ok && r.output) {
              out(box('reason', palette.dim(r.output.slice(0, 600)), { width: termWidth() - 4, accent: 'error' }));
            }
            break;
          }

          case 'reminder':
            out(`  ${palette.thinking('↻')} ${palette.dim('reminder injected')}`);
            break;

          case 'approval':
            if (event.decision === 'deny') out(`    ${palette.error('denied')}`);
            break;

          case 'notice':
            out(`  ${event.level === 'error' ? palette.error('✗') : palette.warn('!')} ${palette.dim(event.text)}`);
            break;

          case 'usage':
            turnCost += event.costUsd;
            turnIn += event.usage.inputTokens;
            turnOut += event.usage.outputTokens;
            break;

          case 'done':
            if (event.reason === 'max-steps') err(palette.warn(`  stopped after ${event.steps} steps`));
            if (event.reason === 'deadline') err(palette.warn('  stopped: turn budget exhausted'));
            if (event.reason === 'error') exitCode = 1;
            break;
        }
      };

      try {
        const result = await runAgentTurn({
          provider: choice.provider,
          tools,
          workspace,
          history: session.messages,
          input,
          onEvent: handle,
          approve,
          interactive,
          maxSteps: flagNumber(ctx.args, 'max-steps') ?? 40,
          deadlineMs: (flagNumber(ctx.args, 'deadline-min') ?? 10) * 60_000,
          temperature: flagNumber(ctx.args, 'temperature') ?? 0.2,
          project,
          signal: abortController.signal,
        });

        spinner.stop();
        process.stdout.write('\n');

        session.appendTurn(input, result.messages);
        session.recordTurn({
          steps: result.steps,
          usage: result.usage,
          costUsd: result.costUsd,
          editedFiles: result.editedFiles,
          commands: result.commands,
        });

        if (!print) {
          out(
            statusLine([
              { key: 'steps', value: String(result.steps) },
              { key: 'tokens', value: `${result.usage.inputTokens.toLocaleString()}↓ ${result.usage.outputTokens.toLocaleString()}↑` },
              { key: 'cost', value: `$${turnCost.toFixed(4)}` },
              { key: 'time', value: formatDuration(Date.now() - turnStart) },
              ...(result.editedFiles.length > 0 ? [{ key: 'edited', value: result.editedFiles.join(' ') }] : []),
              ...(result.ranVerification ? [{ key: 'verified', value: 'yes' }] : []),
            ], termWidth()),
          );
          out('');
        }
      } catch (err2) {
        spinner.stop();
        const message = err2 instanceof ProviderError ? `${err2.message}${err2.hint ? ` — ${err2.hint}` : ''}` : String(err2);
        err(palette.error(`  ${message}`));
        exitCode = 1;
      } finally {
        abortController = null;
      }
    };

    /* ------------------------------------------------------------ slash */
    const handleSlash = async (line: string): Promise<boolean> => {
      const [cmd, ...rest] = line.slice(1).split(/\s+/);
      const arg = rest.join(' ').trim();
      switch (cmd) {
        case 'help':
          out('');
          out(box('commands', kv(SLASH_HELP.map(([k, v]) => ({ key: k, value: v })), { indent: 0 }), { width: termWidth(), accent: 'primary' }));
          out('');
          return true;
        case 'quit':
        case 'exit':
          return false;
        case 'model':
          if (arg) {
            // Providers are immutable by design (the model is fixed at construction
            // and baked into cost accounting), so switching rebuilds the provider
            // rather than mutating it.
            try {
              choice = selectProvider(choice.isLocal ? 'local' : 'cloud', arg);
              session.model = choice.provider.model;
              session.provider = choice.provider.id;
              out(palette.dim(`  model → ${choice.label}`));
            } catch (e) {
              out(palette.error(`  ${e instanceof Error ? e.message : String(e)}`));
            }
          } else {
            out(kv([{ key: 'current', value: choice.label }, { key: 'tier', value: choice.isLocal ? 'local' : 'cloud' }]));
          }
          return true;
        case 'local':
        case 'cloud': {
          try {
            choice = selectProvider(cmd === 'local' ? 'local' : 'cloud');
            session.model = choice.provider.model;
            session.provider = choice.provider.id;
            out(palette.dim(`  using ${cmd}: ${choice.label}`));
          } catch (e) {
            out(palette.error(`  ${e instanceof Error ? e.message : String(e)}`));
          }
          return true;
        }
        case 'workspace': {
          const p = await gatherProjectContext(workspace);
          out(kv([
            { key: 'root', value: p.workspace },
            ...(p.gitBranch ? [{ key: 'branch', value: p.gitBranch }] : []),
            ...(p.gitStatus ? [{ key: 'status', value: p.gitStatus.split('\n').length + ' changed file(s)' }] : []),
            ...(p.topLevel ? [{ key: 'entries', value: p.topLevel }] : []),
            ...(p.instructions ? [{ key: 'instructions', value: p.instructions.map((i) => i.path).join(', ') }] : []),
          ]));
          return true;
        }
        case 'tools':
          out('');
          for (const risk of ['read', 'write', 'exec'] as const) {
            const names = tools.byRisk()[risk];
            out(box(risk, names.length > 0 ? names.join('  ') : palette.dim('(none)'), { width: termWidth(), accent: risk === 'exec' ? 'warn' : risk === 'write' ? 'primary' : 'accent' }));
          }
          out('');
          return true;
        case 'cost':
          out(kv([
            { key: 'turns', value: String(session.stats.turns) },
            { key: 'steps', value: String(session.stats.steps) },
            { key: 'tokens in', value: session.stats.inputTokens.toLocaleString(), note: session.stats.cachedInputTokens > 0 ? `${session.stats.cachedInputTokens.toLocaleString()} cached` : undefined },
            { key: 'tokens out', value: session.stats.outputTokens.toLocaleString() },
            { key: 'spend', value: `$${session.stats.costUsd.toFixed(4)}` },
            { key: 'files edited', value: session.stats.filesEdited.length > 0 ? session.stats.filesEdited.join(' ') : '(none)' },
          ]));
          return true;
        case 'clear':
          session.clear();
          out(palette.dim('  conversation cleared'));
          return true;
        case 'save': {
          const path = saveSession(ctx.dataDir, session);
          out(palette.dim(`  saved ${path}`));
          return true;
        }
        case 'sessions': {
          const list = listSessions(ctx.dataDir, 8);
          out(list.length === 0
            ? palette.dim('  no sessions yet')
            : kv(list.map((s) => ({ key: s.updatedAt.slice(0, 16).replace('T', ' '), value: `${s.id}  ${s.model}  ${s.stats.turns} turn(s)`, note: s.workspace }))));
          return true;
        }
        default:
          out(palette.warn(`  unknown command: /${cmd}. Try /help`));
          return true;
      }
    };

    /* ------------------------------------------------------------ one-shot */
    if (oneShot || (useDemo && !isTty())) {
      const prompt = oneShot || 'show me what this harness does';
      if (!print) out(palette.dim(`  › ${prompt}`));
      await runTurn(prompt);
      if (print) out(result_text_of(session));
      cleanup();
      const saved = saveSession(ctx.dataDir, session);
      return {
        human: [],
        json: { ok: exitCode === 0, session: session.id, transcript: saved, stats: session.stats },
        exitCode,
      };
    }

    /* ------------------------------------------------------------ repl */
    out(palette.dim('  type a request, or /help. Ctrl-D to exit.'));
    out('');

    await new Promise<void>((resolveLoop) => {
      const loop = (): void => {
        rl.prompt();
        rl.once('line', (line) => {
          void (async () => {
            const text = line.trim();
            if (text === '') return loop();

            if (text.startsWith('/')) {
              const keepGoing = await handleSlash(text);
              if (!keepGoing) return resolveLoop();
              return loop();
            }

            if (text.startsWith('!')) {
              const command = text.slice(1).trim();
              const decision = await approve({ kind: 'exec', title: 'Run command', detail: command });
              if (decision !== 'deny') {
                const { exec } = await import('../util/proc.ts');
                const res = await exec('/bin/sh', ['-c', command], { cwd: workspace, timeoutMs: 300_000, maxOutputBytes: 256 * 1024 });
                process.stdout.write(res.stdout);
                if (res.stderr) process.stderr.write(res.stderr);
              } else {
                out(palette.dim('  skipped'));
              }
              return loop();
            }

            await runTurn(text);
            saveSession(ctx.dataDir, session);
            loop();
          })();
        });
        rl.once('close', () => resolveLoop());
      };
      loop();
    });

    cleanup();

    function cleanup(): void {
      process.removeListener('SIGINT', onSigint);
      try {
        rl.close();
      } catch {
        /* already closed */
      }
    }

    return {
      human: [],
      json: { ok: exitCode === 0, session: session.id, stats: session.stats },
      exitCode,
    };
  },
};

/**
 * The last assistant message in the session, which in `--print` mode is the answer.
 * Extracted rather than inlined so the one-shot path and any future scripting path
 * agree on what "the answer" means.
 */
function result_text_of(session: Session): string {
  for (let i = session.messages.length - 1; i >= 0; i--) {
    const m = session.messages[i];
    if (m && m.role === 'assistant' && m.content.trim()) return m.content;
  }
  return '';
}

/** A registry with only the read tools, for `--read-only`. */
function readOnlyRegistry(): ToolRegistry {
  const registry = new ToolRegistry();
  for (const tool of buildToolRegistry().list()) if (tool.risk === 'read') registry.register(tool);
  return registry;
}
