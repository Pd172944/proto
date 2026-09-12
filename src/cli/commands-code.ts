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
import { routeTask } from '../router/index.ts';
import type { RouteDecision, Tier } from '../router/types.ts';
import { EpisodeStore } from '../memory/store.ts';
import type { AgentEvent } from '../agent/loop.ts';
import { Session, listSessions, loadSession, latestSession, saveSession } from '../agent/session.ts';
import { gatherProjectContext } from '../agent/prompt.ts';
import {
  activityLine,
  approvalPanel,
  box,
  chip,
  codeBlock,
  colorEnabled,
  createRailWriter,
  diff,
  footerHints,
  glyph,
  kv,
  markdownLite,
  palette,
  resetColorMode,
  setColorEnabled,
  spinnerFrame,
  statusLine,
  turnDivider,
  welcomePanel,
} from '../tui/theme.ts';
import { HARNESS_VERSION } from '../version.ts';
import { formatDuration, readTextOrNull } from '../util/fsx.ts';

/* ------------------------------------------------------------------ */
/* Rendering helpers                                                   */
/* ------------------------------------------------------------------ */

function termWidth(): number {
  // `process.stdout.columns` is undefined when output is piped, so fall back to COLUMNS
  // (a pager or a test harness usually sets it) and only then to a sane 100.
  const env = Number(process.env['COLUMNS']);
  const known = process.stdout.columns && process.stdout.columns > 0 ? process.stdout.columns : env > 0 ? env : 100;
  // Never exceed the real terminal: a panel wider than the screen hard-wraps and the
  // frame breaks. 40 is a readability floor rather than a claim about narrow terminals —
  // below it an unbreakable token (a shell command inside a code block) can still run over.
  return Math.max(40, Math.min(120, known));
}

function isTty(): boolean {
  return process.stdout.isTTY === true && process.stdin.isTTY === true;
}

/** A spinner that owns exactly one terminal line and always cleans up after itself. */
/**
 * The live activity line, in the spirit of a bubbletea spinner: one line that owns
 * itself, animates, and reports how long the wait has been and what it has cost so far.
 *
 * A bare spinner tells you the program is alive. A spinner with an elapsed clock tells
 * you whether the local model is merely slow — which is the difference between waiting
 * and reaching for Ctrl-C.
 */
class Activity {
  private timer: NodeJS.Timeout | null = null;
  private frame = 0;
  private active = false;
  private startedAt = 0;
  private labelText = 'working…';
  private tokensOut = 0;
  private costUsd = 0;

  setLabel(label: string): void {
    this.labelText = label;
    this.render();
  }

  /** Called as tokens stream so the counter moves while the model is talking. */
  addTokens(n: number): void {
    this.tokensOut += n;
    this.render();
  }

  addCost(usd: number): void {
    this.costUsd += usd;
    this.render();
  }

  start(label?: string): void {
    if (label) this.labelText = label;
    if (this.active) return;
    this.active = true;
    if (this.startedAt === 0) this.startedAt = Date.now();
    if (!isTty()) return;
    this.timer = setInterval(() => {
      this.frame = (this.frame + 1) % 10;
      this.render();
    }, 80);
  }

  /** Reset the counters for a new turn. */
  reset(): void {
    this.startedAt = Date.now();
    this.tokensOut = 0;
    this.costUsd = 0;
  }

  private render(): void {
    if (!this.active || !isTty()) return;
    process.stdout.write(
      `\r\x1b[2K${activityLine({
        frame: this.frame,
        label: this.labelText,
        elapsedMs: Date.now() - this.startedAt,
        tokensOut: this.tokensOut,
        costUsd: this.costUsd,
        width: termWidth() - 2,
      })}`,
    );
  }

  /** Clear the line so real output can take it over. */
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
  ['/route [task]', 're-run the router and switch tier (default: re-route the last message)'],
  ['/escalate', 'move up one tier (local → cloud-cheap → cloud-strong)'],
  ['/deescalate', 'move down one tier'],
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
    { name: 'provider', type: 'string', description: 'override the cloud provider for this session (e.g. anthropic)' },
    { name: 'model', type: 'string', description: 'override the model for this session' },
    { name: 'route', type: 'boolean', description: 'let the router pick the tier for the session (default: on unless --local/--model/--provider given)' },
    { name: 'no-route', type: 'boolean', description: 'skip routing and use the configured model' },
    { name: 'per-turn-route', type: 'boolean', description: 're-route on every message (slower, breaks prompt caching; for experimentation)' },
    { name: 'escalate-on-stuck', type: 'boolean', description: 'move to a stronger tier automatically if the agent hits its step budget' },
    { name: 'no-save', type: 'boolean', description: 'do not write a session transcript to the data directory' },
    { name: 'budget-usd', type: 'number', description: 'stop the session once estimated spend exceeds this' },
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

    /**
     * Turn a tier from the router into a concrete provider.
     *
     * `local-tiny` and `local` use the local runtime; `cloud-cheap` uses the cheaper
     * cloud model when one is configured, `cloud-strong` the main one. This is the same
     * mapping the batch path uses, so a tier means the same thing on both paths.
     */
    const providerForTier = (tier: Tier, modelOverride?: string): ProviderChoice => {
      if (tier === 'local' || tier === 'local-tiny') return chooseProvider({ ...ctx, cfg: runCfg }, 'local');
      const model = modelOverride ?? cloudTierModel(runCfg, tier === 'cloud-cheap' ? 'cloud-cheap' : 'cloud-strong');
      const provider = buildCloudProvider(runCfg, ctx.dataDir, model);
      if (!provider) {
        throw new Error(
          `the router chose "${tier}" but no cloud provider is available. ` +
            `Set an API key, or force a tier with --local.`,
        );
      }
      return { provider, label: `${runCfg.cloud.provider} · ${provider.model}`, isLocal: false };
    };

    const selectProvider = (want: 'cloud' | 'local', model?: string): ProviderChoice => {
      if (useDemo) return { provider: new DemoProvider(), label: 'demo · scripted', isLocal: false };
      if (want === 'local') return chooseProvider({ ...ctx, cfg: runCfg }, 'local');
      if (model) {
        const provider = buildCloudProvider(runCfg, ctx.dataDir, model);
        if (!provider) throw new Error(`no cloud provider available for model "${model}"`);
        return { provider, label: `${runCfg.cloud.provider} · ${model}`, isLocal: false };
      }
      return chooseProvider({ ...ctx, cfg: runCfg }, 'cloud');
    };

    // A per-run provider/model override is what makes scripted evaluation possible:
    // comparing two models on the same task set without editing config between runs.
    const providerFlag = flagString(ctx.args, 'provider');
    const modelFlag = flagString(ctx.args, 'model');
    const runCfg = providerFlag ? { ...ctx.cfg, cloud: { ...ctx.cfg.cloud, provider: providerFlag } } : ctx.cfg;

    /* ---------------------------------------------------------------- routing */
    // Routing happens at *session* granularity, not per turn. The first message is a
    // task description, so it is classifiable exactly like a batch task; later turns
    // are not (they are reactions to tool output whose shape is unknowable in advance),
    // and switching models mid-conversation would break prompt caching and the
    // provider's view of the thread. `/route` re-decides on demand, and the tier can be
    // escalated explicitly or automatically when the agent gets stuck.
    // Routing is on by default, and is switched off by any explicit model selection
    // (`--local`, `--model`, `--provider`) or by `--no-route`. Passing `--route`
    // alongside one of those is a contradiction we resolve in routing's favour, noisily.
    const explicitNoRoute = flagBool(ctx.args, 'no-route');
    const explicitModelChoice = Boolean(providerFlag) || Boolean(modelFlag) || flagBool(ctx.args, 'local');
    const routingEnabled = !useDemo && !explicitNoRoute && (flagBool(ctx.args, 'route') || !explicitModelChoice);
    let routed = false;
    let decision: RouteDecision | null = null;
    let tier: Tier | null = null;

    const readSpendToday = (): number => {
      try {
        return new EpisodeStore(ctx.dataDir).stats().spendTodayUsd;
      } catch {
        return 0;
      }
    };

    /** Route `task` and return the resulting provider choice, or null if routing is off. */
    const routeFor = async (task: string, files: Array<{ path: string; content: string }> = []): Promise<ProviderChoice | null> => {
      if (!routingEnabled) return null;
      decision = await routeTask({
        cfg: runCfg,
        dataDir: ctx.dataDir,
        ctx: { task, files, workspace },
        verifierAvailable: true,
        cloudSpendTodayUsd: readSpendToday(),
      });
      tier = decision.tier;
      return providerForTier(tier);
    };

    const nudgeTier = (up: boolean): Tier | null => {
      const order: Tier[] = ['local-tiny', 'local', 'cloud-cheap', 'cloud-strong'];
      if (!tier) return null;
      const i = order.indexOf(tier);
      const next = order[Math.min(order.length - 1, Math.max(0, i + (up ? 1 : -1)))];
      return next ?? null;
    };

    /**
     * Files the message appears to reference, for the router's benefit.
     *
     * The batch path is told which files are in scope via `--file`; an interactive
     * message has no such flag. Pulling the paths straight out of the text gives the
     * router real code to look at (file count, nesting, language) instead of routing on
     * prose alone, which materially changes the decision for a task like "fix the loop
     * in prices.py". Capped at three files so a long message cannot stall startup.
     */
    const filesMentionedIn = (text: string): Array<{ path: string; content: string }> => {
      const found: Array<{ path: string; content: string }> = [];
      const re = /(?:^|[\s"'`(\[])([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]{1,6})(?=[\s"'`)\].,:;]|$)/g;
      const seen = new Set<string>();
      let match: RegExpExecArray | null;
      while ((match = re.exec(text)) !== null) {
        const rel = (match[1] ?? '').replace(/^\.\//, '');
        if (!rel || seen.has(rel) || found.length >= 3) continue;
        seen.add(rel);
        const abs = resolve(workspace, rel);
        if (abs !== workspace && !abs.startsWith(`${workspace}/`)) continue;
        const content = readTextOrNull(abs);
        if (content === null || content.length > 200_000) continue;
        found.push({ path: rel, content });
      }
      return found;
    };

    let choice: ProviderChoice;
    try {
      if (useDemo) {
        choice = { provider: new DemoProvider(), label: 'demo · scripted', isLocal: false };
      } else if (routingEnabled) {
        // Provisional: the router decides on the first message. Shown in the banner so
        // the user is not looking at a blank model field while they type.
        choice = {
          provider: buildCloudProvider(runCfg, ctx.dataDir, cloudTierModel(runCfg, 'cloud-strong')) ?? buildLocalProvider(runCfg),
          label: 'router decides on your first message',
          isLocal: false,
        };
      } else if (flagBool(ctx.args, 'local')) {
        choice = chooseProvider(ctx, 'local');
      } else {
        const provider = buildCloudProvider(runCfg, ctx.dataDir, modelFlag ?? cloudTierModel(runCfg, 'cloud-strong'));
        if (!provider) throw new Error('no cloud provider available. Set an API key, or use --local / --demo.');
        choice = {
          provider,
          label: `${runCfg.cloud.provider} · ${provider.model}`,
          isLocal: false,
        };
      }
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
    /**
     * Progress chrome (tool chips, notices, reminders) writes to stderr in `--print` mode
     * so stdout carries the model's answer and nothing else: `proto code --print ... > patch.diff`
     * must not have tool chatter spliced into the file.
     */
    const chatter = (s = ''): void => {
      if (print) process.stderr.write(`${s}\n`);
      else out(s);
    };

    if (!print) {
      const notes: Array<{ key: string; value: string; note?: string }> = [
        { key: 'model', value: choice.label, note: choice.isLocal ? 'local' : 'cloud' },
        { key: 'workspace', value: workspace },
      ];
      if (project.gitBranch) {
        notes.push({ key: 'git', value: project.gitBranch, note: project.gitStatus ? 'dirty' : 'clean' });
      }
      notes.push({
        key: 'tools',
        value: `${tools.list().length}`,
        note: readOnly ? 'read-only mode' : `${tools.byRisk().write.length} write, ${tools.byRisk().exec.length} exec`,
      });
      if (routingEnabled) {
        notes.push({
          key: 'routing',
          value: 'on',
          note: explicitModelChoice
            ? '--route overrides --local/--model/--provider this session'
            : 'chosen from your first message',
        });
      }
      if (flagBool(ctx.args, 'no-save')) {
        notes.push({ key: 'transcripts', value: 'not saved', note: '--no-save: nothing is written to disk' });
      }

      out(welcomePanel(notes, { subtitle: `${HARNESS_VERSION} · reads, edits and runs commands in your repo`, width: termWidth() }));
      out('');
      out(
        '  ' +
          footerHints(
            [
              { key: '/help', label: 'commands' },
              { key: 'Ctrl-C', label: 'stop the turn' },
              { key: 'Ctrl-D', label: 'quit' },
              { key: readOnly ? 'read-only' : 'approval', label: readOnly ? '' : 'before any write' },
            ],
            termWidth() - 2,
          ),
      );
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
    // The last failure reason, so `--json` carries it. Without this a scripted caller
    // sees `ok: false` with no explanation and has to scrape prose off stdout.
    let lastError: string | undefined;

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
    const spinner = new Activity();
    let turnIndex = 0;

    const runTurn = async (input: string): Promise<void> => {
      // Route once per session, on the first message, because that message is a task
      // description. Later messages are reactions to tool output and are not
      // independently classifiable, and switching models mid-thread would invalidate the
      // provider's cached view of the conversation.
      if (routingEnabled && !routed) {
        routed = true;
        try {
          const routedChoice = await routeFor(input, filesMentionedIn(input));
          if (routedChoice) {
            choice = routedChoice;
            session.model = choice.provider.model;
            session.provider = choice.provider.id;
            const d = decision;
            if (d) {
              out('');
              out(
                `  ${palette.accent('routing')} ${palette.bold(d.tier)}  ` +
                  palette.dim(`p(local)=${d.pLocalSuccess.toFixed(2)} · difficulty ${d.difficulty.toFixed(2)} · ${d.taskClass}`),
              );
              out(palette.dim(`  ${d.reason}`));
              out(palette.dim('  /route to re-decide · /escalate to move up a tier · /model to pick explicitly'));
            }
          }
        } catch (routeErr) {
          // Routing must never stop work: fall back to the provisional provider.
          err(palette.warn(`  routing unavailable (${routeErr instanceof Error ? routeErr.message : String(routeErr)}); using ${choice.label}`));
        }
      }

      abortController = new AbortController();
      turnIndex += 1;
      if (!print && turnIndex > 1) {
        out('');
        out(turnDivider(`turn ${turnIndex}`, termWidth()));
        out('');
      }
      const turnStart = Date.now();
      const writeAnswer = createRailWriter(termWidth(), (t) => process.stdout.write(t));
      let sawDelta = false;
      let stepLabel = '';
      let turnCost = 0;
      let turnIn = 0;
      let turnOut = 0;
      let toolStartedAt = 0;

      spinner.reset();
      spinner.start('thinking\u2026');

      const handle = (event: AgentEvent): void => {
        switch (event.type) {
          case 'context':
            break;

          case 'model-start':
            stepLabel = event.step === 1 ? 'thinking…' : `step ${event.step}…`;
            spinner.setLabel(stepLabel);
            spinner.start();
            break;

          case 'text-delta':
            // In `--print` mode the answer is emitted once, complete, by the caller: the
            // deltas would otherwise be written to stdout and then repeated verbatim.
            if (print) break;
            if (!sawDelta) {
              spinner.stop();
              out('');
              sawDelta = true;
            }
            writeAnswer.write(event.text);
            break;

          case 'text':
            // Only print the settled text when nothing was streamed, otherwise the
            // user would see the whole answer twice.
            if (!sawDelta && !print) {
              spinner.stop();
              out('');
              const mark = colorEnabled() ? `${palette.accent('\u2503')} ` : '| ';
              out(
                markdownLite(event.text, termWidth() - 4)
                  .split('\n')
                  .map((line) => (line.trim() === '' ? line : mark + line))
                  .join('\n'),
              );
              sawDelta = true;
            }
            break;

          case 'tool-start': {
            spinner.stop();
            toolStartedAt = Date.now();
            const args = summarizeToolArgs(event.toolCall.name, event.toolCall.args);
            const line = '  ' + chip('run', event.toolCall.name, args, { width: termWidth() - 2 });
            // On a terminal the finished chip overwrites this line in place, the way a
            // bubbletea view updates rather than appends. Piped, both lines are kept:
            // a script reading stderr still learns what ran and how it went.
            if (isTty()) process.stdout.write(line);
            else chatter(line);
            break;
          }

          case 'tool-end': {
            const r = event.result;
            const ms = toolStartedAt === 0 ? 0 : Date.now() - toolStartedAt;
            toolStartedAt = 0;
            const detail = r.ok ? `${r.title} · ${formatDuration(ms)}` : r.title;
            const line = '  ' + chip(r.ok ? 'ok' : 'fail', event.toolCall.name, detail, { width: termWidth() - 2 });
            if (isTty()) process.stdout.write(`\r\x1b[2K${line}\n`);
            else chatter(line);
            if (!r.ok && r.output) {
              chatter('  ' + box('reason', palette.dim(r.output.slice(0, 600)), { width: termWidth() - 4, accent: 'error' }).split('\n').join('\n  '));
            }
            // Output is the agent's next input, not decoration: the activity line goes
            // back up so the user keeps seeing that it is working, not hung.
            spinner.start(stepLabel === '' ? 'working…' : stepLabel);
            break;
          }

          case 'reminder':
            chatter(`  ${palette.thinking('↻')} ${palette.dim('reminder injected')}`);
            break;

          case 'approval':
            if (event.decision === 'deny') chatter(`    ${palette.error('denied')}`);
            break;

          case 'notice':
            chatter(`  ${event.level === 'error' ? palette.error('✗') : palette.warn('!')} ${palette.dim(event.text)}`);
            if (event.level === 'error') lastError = event.text;
            break;

          case 'usage':
            turnCost += event.costUsd;
            turnIn += event.usage.inputTokens;
            turnOut += event.usage.outputTokens;
            if (event.usage.outputTokens > 0) spinner.addTokens(event.usage.outputTokens);
            if (event.costUsd > 0) spinner.addCost(event.costUsd);
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
        if (!print) writeAnswer.end();

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

        // Escalate when the agent burned its whole step budget: being stuck is the one
        // signal from an agent turn that reliably means "this model is not up to it",
        // and it is the session-scale equivalent of the batch path's escalation.
        if (result.reason === 'max-steps' && flagBool(ctx.args, 'escalate-on-stuck') && tier) {
          const next = nudgeTier(true);
          if (next && next !== tier) {
            try {
              choice = providerForTier(next);
              tier = next;
              session.model = choice.provider.model;
              session.provider = choice.provider.id;
              out(palette.warn(`  escalated to ${next} (${choice.label}) after hitting the step budget`));
              out('');
            } catch {
              /* keep the current tier if the next one is unavailable */
            }
          }
        }

        // Budget guard. A long agent session can quietly accumulate spend, and a hard
        // stop with an explanation is better than a surprise on the bill.
        const budget = flagNumber(ctx.args, 'budget-usd');
        if (budget !== undefined && session.stats.costUsd >= budget) {
          err(palette.warn(`  session spend $${session.stats.costUsd.toFixed(4)} has reached the --budget-usd limit of $${budget.toFixed(2)}`));
          exitCode = 0;
          if (!print) {
            err(palette.dim('  raise it with --budget-usd, or /quit and start a new session'));
            rl.close();
          }
        }
      } catch (err2) {
        spinner.stop();
        const message = err2 instanceof ProviderError ? `${err2.message}${err2.hint ? ` — ${err2.hint}` : ''}` : String(err2);
        err(palette.error(`  ${message}`));
        lastError = message;
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
        case 'route': {
          if (useDemo) {
            out(palette.dim('  --demo does not route'));
            return true;
          }
          const task = arg || [...session.messages].reverse().find((m) => m.role === 'user')?.content || '';
          if (!task.trim()) {
            out(palette.warn('  nothing to route on yet — send a message first, or pass text: /route <task>'));
            return true;
          }
          const previous = tier;
          const routedChoice = await routeFor(task, filesMentionedIn(task));
          if (!routedChoice || !decision) {
            out(palette.warn('  routing is off for this session'));
            return true;
          }
          choice = routedChoice;
          session.model = choice.provider.model;
          session.provider = choice.provider.id;
          out('');
          out(
            `  ${palette.accent('routing')} ${palette.bold(decision.tier)}` +
              (previous && previous !== decision.tier ? palette.dim(`  (was ${previous})`) : ''),
          );
          out(palette.dim(`  ${decision.reason}`));
          return true;
        }
        case 'escalate':
        case 'deescalate': {
          if (!tier) {
            out(palette.warn('  no routed tier to move (the session is pinned to an explicit model)'));
            return true;
          }
          const next = nudgeTier(cmd === 'escalate');
          if (!next || next === tier) {
            out(palette.dim(`  already at the ${cmd === 'escalate' ? 'strongest' : 'cheapest'} tier (${tier})`));
            return true;
          }
          try {
            choice = providerForTier(next);
            tier = next;
            session.model = choice.provider.model;
            session.provider = choice.provider.id;
            out(palette.dim(`  tier → ${next}  (${choice.label})`));
          } catch (e) {
            out(palette.error(`  ${e instanceof Error ? e.message : String(e)}`));
          }
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
          if (flagBool(ctx.args, 'no-save')) {
            out(palette.dim('  --no-save is set, so nothing is written'));
            return true;
          }
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
      const saved = flagBool(ctx.args, 'no-save') ? '(not saved: --no-save)' : saveSession(ctx.dataDir, session);
      return {
        human: [],
        json: {
          ok: exitCode === 0,
          ...(lastError ? { error: lastError } : {}),
          session: session.id,
          transcript: saved,
          stats: session.stats,
        },
        exitCode,
      };
    }

    /* ------------------------------------------------------------ repl */
    out(palette.dim('  type a request, or /help. Ctrl-D to exit.'));
    out('');

    /*
     * One persistent 'line' listener, with the input stream paused while a line is
     * being handled.
     *
     * The previous version re-registered `rl.once('line', ...)` *after* awaiting the
     * handler. Piped input arrives all at once, so every line after the first was
     * emitted while no listener existed and was silently dropped — which made the
     * REPL unusable from a pipe or a script, and untestable. Pausing the stream makes
     * the ordering explicit instead of relying on timing, and resume() puts it back.
     */
    await new Promise<void>((resolveLoop) => {
      let finished = false;
      let stdinClosed = false;
      let pending = 0;
      /** Resolve only once stdin has ended *and* every queued line has been handled. */
      const maybeFinish = (): void => {
        if (finished || !stdinClosed || pending > 0) return;
        finished = true;
        resolveLoop();
      };
      rl.on('close', () => {
        stdinClosed = true;
        maybeFinish();
      });

      /** Returns false when the session should end. */
      const handleLine = async (line: string): Promise<boolean> => {
        const text = line.trim();
        if (text === '') return true;

        if (text.startsWith('/')) return await handleSlash(text);

        if (text.startsWith('!')) {
          const command = text.slice(1).trim();
          const decision = await approve({ kind: 'exec', title: 'Run command', detail: command });
          if (decision !== 'deny') {
            const { exec } = await import('../util/proc.ts');
            const res = await exec('/bin/sh', ['-c', command], {
              cwd: workspace,
              timeoutMs: 300_000,
              maxOutputBytes: 256 * 1024,
            });
            process.stdout.write(res.stdout);
            if (res.stderr) process.stderr.write(res.stderr);
          } else {
            out(palette.dim('  skipped'));
          }
          return true;
        }

        await runTurn(text);
        if (!flagBool(ctx.args, 'no-save')) saveSession(ctx.dataDir, session);
        return true;
      };

      /*
       * Lines are handled through a FIFO promise chain, not concurrently.
       *
       * `rl.pause()` alone is not enough: when stdin is a pipe, Node reads the whole
       * buffer at once and emits every 'line' event in the same tick, before any pause
       * takes effect. Handlers then interleave — `/escalate` ran before the `/route`
       * that was supposed to precede it, and saw no tier because routing had not
       * finished. Chaining makes the ordering explicit rather than dependent on timing,
       * which is also what makes the REPL usable from a script or a test.
       */
      let chain: Promise<void> = Promise.resolve();

      rl.on('line', (line: string) => {
        if (finished) return;
        pending++;
        rl.pause();
        chain = chain.then(async () => {
          let keepGoing = true;
          try {
            keepGoing = await handleLine(line);
          } catch (e) {
            err(palette.error(`  ${e instanceof Error ? e.message : String(e)}`));
          } finally {
            pending--;
            if (!keepGoing) {
              stdinClosed = true;
              rl.close();
            } else if (!stdinClosed) {
              rl.resume();
              rl.prompt();
            }
            maybeFinish();
          }
        });
      });

      rl.prompt();
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
      json: { ok: exitCode === 0, ...(lastError ? { error: lastError } : {}), session: session.id, stats: session.stats },
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
