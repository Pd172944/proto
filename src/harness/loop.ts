/**
 * The batch task loop: route a task, attempt it, verify it, escalate if needed.
 *
 * One task in, one verified result out. The shape is:
 *
 *     route -> (local attempt -> verify -> repair) -> escalate -> verify
 *
 * Four things about this loop are deliberate and worth stating, because they are
 * where a naive implementation would quietly go wrong:
 *
 *  1. **Verification, not vibes, decides success.** An attempt "succeeded" iff the
 *     verifier passed it. This is what makes escalation trustworthy: we never
 *     escalate because a model *sounded* unsure, only because a check failed.
 *
 *  2. **Escalation is bounded.** `maxCloudAttempts` caps worst-case spend. A
 *     routing bug can cost the user a few cents, not a surprise bill.
 *
 *  3. **Nothing is written to the workspace unless asked.** `apply` is off by
 *     default. The verifier works on an in-memory copy of the files.
 *
 *  4. **Provider transport errors are not model failures.** If the local server is
 *     down we record the error on the attempt and escalate, rather than scoring the
 *     local model as wrong — conflating the two would make the router's own
 *     statistics lie about why a task failed.
 */

import { buildCloudProvider, buildLocalProvider, cloudTierModel } from '../providers/index.ts';
import type { ChatResponse, Message, Provider } from '../providers/types.ts';
import { ProviderError } from '../providers/types.ts';
import { resolveApiKeyFor } from '../config/load.ts';
import type { ProtoConfig } from '../config/schema.ts';
import { routeTask } from '../router/index.ts';
import type { RouteDecision, TaskContext, Tier } from '../router/types.ts';
import { verifyCandidate } from '../verify/index.ts';
import type { VerificationReport } from '../verify/types.ts';
import { buildMessages, buildRepairPrompt, buildUserPrompt, repairSystemPrompt, systemPromptFor } from './prompt.ts';
import { ensureDir, writeTextAtomic } from '../util/fsx.ts';
import { dirname, resolve, sep } from 'node:path';

/**
 * Difficulty above which an escalation (or a cloud retry) uses the strongest
 * cloud model. Deliberately lower than the router's own cheap/strong boundary:
 * we already have evidence the task is harder than the router thought.
 */
const ESCALATION_STRONG_THRESHOLD = 0.4;

export type AttemptSource = 'initial' | 'repair' | 'escalation';

/** What one model call did. Kept for cost accounting and for `--explain`. */
interface Attempt {
  n: number;
  tier: Tier;
  source: AttemptSource;
  providerId: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  latencyMs: number;
  error?: string;
  verification: { passed: boolean; score: number; blockers: string[]; durationMs: number } | null;
}

/** How the task ended. */
export type RunStatus =
  | 'local-success'
  | 'cloud-success'
  | 'escalated-cloud-success'
  | 'failed'
  | 'abandoned'
  | 'dry-run';

export interface RunOptions {
  cfg: ProtoConfig;
  dataDir: string;
  ctx: TaskContext;
  /** Write verified edits to disk. Off by default. */
  apply?: boolean;
  /** Force a tier, bypassing the router. */
  forceTier?: Tier;
  /** Route only; make no model calls (except routing health probes). */
  dryRun?: boolean;
  /** Override local repair attempts. */
  maxRepair?: number;
  /** Injectable providers for tests. */
  localProvider?: Provider;
  cloudProvider?: Provider;
  /** Skip routing health probes (useful offline / in tests). */
  offlineRoute?: boolean;
  /** Progress callback for the CLI. */
  onEvent?: (event: RunEvent) => void;
  /** Unload the local model after the task to free RAM. */
  unloadLocalAfter?: boolean;
}

export type RunEvent =
  | { type: 'route'; decision: RouteDecision }
  | { type: 'attempt-start'; tier: Tier; source: AttemptSource; model: string }
  | { type: 'attempt-end'; tier: Tier; source: AttemptSource; response: ChatResponse }
  | { type: 'provider-error'; tier: Tier; message: string; hint?: string }
  | { type: 'verify'; tier: Tier; passed: boolean; score: number; blockers: string[] }
  | { type: 'escalate'; from: Tier; to: Tier; reason: string }
  | { type: 'applied'; files: string[] };

export interface RunResult {
  decision: RouteDecision;
  report: VerificationReport | null;
  /** Final candidate text (raw model output). */
  finalText: string;
  /** Files written, when `apply` was set. */
  writtenFiles: string[];
  status: RunStatus;
  /** Every model call made, for cost reporting. */
  attempts: Attempt[];
  totalCostUsd: number;
  /** True when every attempt consumed zero cloud spend. */
  usedOnlyLocal: boolean;
  warnings: string[];
}

export async function runTask(opts: RunOptions): Promise<RunResult> {
  const { cfg, dataDir, ctx } = opts;
  const onEvent = opts.onEvent ?? ((): void => {});
  const warnings: string[] = [];

  // --- routing -------------------------------------------------------------
  // `cloudSpendTodayUsd` is 0: the harness keeps no cross-invocation spend ledger.
  // Per-task spend is bounded by `routing.maxCloudAttempts`, which is the limit
  // that actually matters here — a single task cannot loop.
  let decision: RouteDecision;
  try {
    decision = await routeTask({
      cfg,
      dataDir,
      ctx,
      ...(opts.offlineRoute ? { offline: true } : {}),
      cloudSpendTodayUsd: 0,
    });
  } catch (err) {
    // Routing must never be the thing that stops work.
    throw new Error(`routing failed: ${err instanceof Error ? err.message : String(err)}`);
  }

  if (opts.forceTier) {
    decision = {
      ...decision,
      tier: opts.forceTier,
      forced: true,
      reasons: [`tier forced to "${opts.forceTier}" by the caller`, ...decision.reasons],
      reason: `forced to ${opts.forceTier}`,
    };
  }
  onEvent({ type: 'route', decision });

  if (opts.dryRun) {
    return {
      decision,
      report: null,
      finalText: '',
      writtenFiles: [],
      status: 'dry-run',
      attempts: [],
      totalCostUsd: 0,
      usedOnlyLocal: true,
      warnings,
    };
  }

  // --- prompts -------------------------------------------------------------
  const system = systemPromptFor(decision.features);
  const userPrompt = buildUserPrompt({
    ctx,
    features: decision.features,
    maxContextChars: Math.max(2000, Math.floor(cfg.local.contextWindow * 3.2)),
  });
  const baseMessages = buildMessages({
    system: system.content,
    user: userPrompt,
    cacheSystem: cfg.cloud.promptCaching,
  });

  const attempts: Attempt[] = [];
  const providers = resolveProviders(opts, cfg, dataDir);

  // --- attempt machinery ---------------------------------------------------
  const runAttempt = async (input: {
    provider: Provider;
    tier: Tier;
    source: AttemptSource;
    messages: Message[];
  }): Promise<{ response: ChatResponse | null; attempt: Attempt }> => {
    const n = attempts.length + 1;
    onEvent({ type: 'attempt-start', tier: input.tier, source: input.source, model: input.provider.model });
    let response: ChatResponse | null = null;
    let error: string | undefined;
    try {
      response = await input.provider.chat({
        messages: input.messages,
        maxTokens:
          input.tier === 'local' || input.tier === 'local-tiny'
            ? cfg.local.maxOutputTokens
            : cfg.cloud.maxOutputTokens,
        temperature:
          input.tier === 'local' || input.tier === 'local-tiny' ? cfg.local.temperature : cfg.cloud.temperature,
        meta: { tier: input.tier, source: input.source },
      });
    } catch (err) {
      const pe = err instanceof ProviderError ? err : null;
      error = pe ? `${pe.message}${pe.hint ? ` (${pe.hint})` : ''}` : err instanceof Error ? err.message : String(err);
      onEvent({
        type: 'provider-error',
        tier: input.tier,
        message: error,
        ...(pe?.hint ? { hint: pe.hint } : {}),
      });
    }

    if (response) onEvent({ type: 'attempt-end', tier: input.tier, source: input.source, response });

    const attempt: Attempt = {
      n,
      tier: input.tier,
      source: input.source,
      providerId: input.provider.id,
      model: response?.model ?? input.provider.model,
      inputTokens: response?.usage.inputTokens ?? 0,
      outputTokens: response?.usage.outputTokens ?? 0,
      costUsd: response?.costUsd ?? 0,
      latencyMs: response?.latencyMs ?? 0,
      verification: null,
    };
    if (error !== undefined) attempt.error = error;
    else if (response?.error) attempt.error = response.error;
    attempts.push(attempt);
    return { response, attempt };
  };

  const verifyAttempt = async (attempt: Attempt, text: string): Promise<VerificationReport> => {
    const report = await verifyCandidate({
      text,
      ctx,
      cfg,
      dataDir,
      features: decision.features,
    });
    attempt.verification = {
      passed: report.passed,
      score: report.score,
      blockers: report.blockers,
      durationMs: report.durationMs,
    };
    onEvent({
      type: 'verify',
      tier: attempt.tier,
      passed: report.passed,
      score: report.score,
      blockers: report.blockers,
    });
    return report;
  };

  // --- primary path --------------------------------------------------------
  let finalReport: VerificationReport | null = null;
  let finalText = '';
  let escalated = false;
  let localAttempted = false;
  let localFailed = false;
  let refusal = false;

  const isLocal = (t: Tier): boolean => t === 'local' || t === 'local-tiny';

  if (isLocal(decision.tier)) {
    localAttempted = true;
    const maxRepair = opts.maxRepair ?? cfg.routing.maxLocalRepairAttempts;
    let messages = baseMessages;
    let lastText = '';

    for (let round = 0; round <= maxRepair; round++) {
      const { response, attempt } = await runAttempt({
        provider: providers.local,
        tier: decision.tier,
        source: round === 0 ? 'initial' : 'repair',
        messages,
      });

      if (!response || response.finishReason === 'error') {
        // Transport failure: not a model-quality signal. Break to escalation.
        localFailed = true;
        break;
      }
      lastText = response.text;
      const report = await verifyAttempt(attempt, response.text);
      if (report.checks.some((c) => c.id === 'refusal' && !c.ok)) refusal = true;

      if (report.passed) {
        finalReport = report;
        finalText = response.text;
        break;
      }

      localFailed = true;
      if (round < maxRepair) {
        messages = [
          { role: 'system', content: repairSystemPrompt() },
          { role: 'user', content: userPrompt },
          {
            role: 'user',
            content: buildRepairPrompt({
              previousText: lastText,
              blockers: report.blockers,
              failedChecks: report.checks.filter((c) => !c.ok).map((c) => ({ detail: c.detail, evidence: c.evidence })),
            }),
          },
        ];
      }
    }

    if (!finalReport && providers.cloud) {
      escalated = true;
      // A failed local attempt is *evidence the task is harder than estimated*, so
      // escalation bumps the cloud tier at a lower difficulty than the router's own
      // cheap/strong boundary. This is deliberate, named, and different from
      // CLOUD_STRONG_DIFFICULTY_THRESHOLD on purpose.
      const to: Tier = decision.difficulty >= ESCALATION_STRONG_THRESHOLD ? 'cloud-strong' : 'cloud-cheap';
      onEvent({
        type: 'escalate',
        from: decision.tier,
        to,
        reason: 'local attempt did not pass verification',
      });
      const cloudResult = await runCloudAttempts({
        cfg,
        tier: to,
        decision,
        baseMessages,
        userPrompt,
        providers,
        runAttempt,
        verifyAttempt,
        warnings,
      });
      finalReport = cloudResult.report;
      finalText = cloudResult.text;
      refusal = refusal || cloudResult.refusal;
    } else if (!finalReport && !providers.cloud) {
      warnings.push(
        `local attempt failed verification and no cloud tier is available (${providers.cloudUnavailable ?? 'unknown reason'}); ` +
          `the task could not be completed`,
      );
    }
  } else {
    // Cloud-first path.
    const cloudResult = await runCloudAttempts({
      cfg,
      tier: decision.tier,
      decision,
      baseMessages,
      userPrompt,
      providers,
      runAttempt,
      verifyAttempt,
      warnings,
    });
    finalReport = cloudResult.report;
    finalText = cloudResult.text;
    refusal = cloudResult.refusal;
  }

  // --- apply ---------------------------------------------------------------
  let writtenFiles: string[] = [];
  if (opts.apply && finalReport?.passed) {
    writtenFiles = applyToWorkspace(finalReport, ctx);
    if (writtenFiles.length > 0) onEvent({ type: 'applied', files: writtenFiles });
  } else if (opts.apply && !finalReport?.passed) {
    warnings.push('refusing to write files: verification did not pass');
  }

  // --- outcome -------------------------------------------------------------
  const tierUsed = attempts[attempts.length - 1]?.tier ?? decision.tier;
  const status: RunStatus = finalReport?.passed
    ? escalated
      ? 'escalated-cloud-success'
      : isLocal(tierUsed)
        ? 'local-success'
        : 'cloud-success'
    : refusal || localFailed || attempts.length > 0
      ? 'failed'
      : 'abandoned';

  const totalCostUsd = round6(attempts.reduce((a, r) => a + r.costUsd, 0));

  if (opts.unloadLocalAfter && providers.local.kind === 'local') {
    await maybeUnload(providers.local);
  }

  return {
    decision,
    report: finalReport,
    finalText,
    writtenFiles,
    status,
    attempts,
    totalCostUsd,
    usedOnlyLocal: totalCostUsd === 0,
    warnings,
  };
}

/* ------------------------------------------------------------------ */
/* Cloud attempts (with bounded retry)                                 */
/* ------------------------------------------------------------------ */

async function runCloudAttempts(input: {
  cfg: ProtoConfig;
  tier: Tier;
  decision: RouteDecision;
  baseMessages: Message[];
  userPrompt: string;
  providers: ProviderBundle;
  runAttempt: (i: { provider: Provider; tier: Tier; source: AttemptSource; messages: Message[] }) => Promise<{
    response: ChatResponse | null;
    attempt: Attempt;
  }>;
  verifyAttempt: (attempt: Attempt, text: string) => Promise<VerificationReport>;
  warnings: string[];
}): Promise<{ report: VerificationReport | null; text: string; refusal: boolean }> {
  const { cfg, decision, providers } = input;
  let report: VerificationReport | null = null;
  let text = '';
  let refusal = false;

  const maxAttempts = Math.max(1, cfg.routing.maxCloudAttempts);
  let tier: Tier = input.tier;

  for (let i = 0; i < maxAttempts; i++) {
    // Escalate the cloud tier on a retry only when the task looks hard; otherwise a
    // cheap model retry is the cost-appropriate response.
    if (i > 0) tier = decision.difficulty >= ESCALATION_STRONG_THRESHOLD ? 'cloud-strong' : 'cloud-cheap';
    const provider = providers.cloudFor(tier);
    if (!provider) {
      input.warnings.push(`cloud tier "${tier}" is not available`);
      break;
    }

    const messages: Message[] = i === 0
      ? input.baseMessages
      : [
          { role: 'system', content: repairSystemPrompt() },
          { role: 'user', content: input.userPrompt },
          {
            role: 'user',
            content: buildRepairPrompt({
              previousText: text,
              blockers: report?.blockers ?? ['the previous answer was rejected'],
              failedChecks: (report?.checks ?? []).filter((c) => !c.ok).map((c) => ({ detail: c.detail, evidence: c.evidence })),
            }),
          },
        ];

    const { response, attempt } = await input.runAttempt({
      provider,
      tier,
      source: i === 0 ? 'escalation' : 'repair',
      messages,
    });

    if (!response || response.finishReason === 'error') {
      input.warnings.push(`cloud attempt ${i + 1} failed at the transport level${attempt.error ? `: ${attempt.error}` : ''}`);
      continue;
    }
    text = response.text;
    report = await input.verifyAttempt(attempt, response.text);
    if (report.checks.some((c) => c.id === 'refusal' && !c.ok)) refusal = true;
    if (report.passed) return { report, text, refusal };
  }

  if (report && !report.passed) {
    input.warnings.push('the cloud answer also failed verification; returning it as an unverified best effort');
  }
  return { report, text, refusal };
}

/* ------------------------------------------------------------------ */
/* Provider resolution                                                 */
/* ------------------------------------------------------------------ */

interface ProviderBundle {
  local: Provider;
  cloud: Provider | null;
  cloudUnavailable?: string;
  cloudFor: (tier: Tier) => Provider | null;
}

function resolveProviders(opts: RunOptions, cfg: ProtoConfig, dataDir: string): ProviderBundle {
  const local = opts.localProvider ?? buildLocalProvider(cfg);

  // Cloud readiness is inferred from config + key presence, matching the router.
  // Probing with a real request would cost money on every task.
  const keyPresent = Boolean(resolveApiKeyFor(cfg, dataDir));
  const cloudReady = Boolean(opts.cloudProvider) || (cfg.cloud.enabled && keyPresent);
  const cloudUnavailable = cloudReady
    ? undefined
    : !cfg.cloud.enabled
      ? 'cloud is disabled in config'
      : `no API key found for provider "${cfg.cloud.provider}"`;

  const cache = new Map<string, Provider | null>();
  const cloudFor = (tier: Tier): Provider | null => {
    if (!cloudReady) return null;
    if (opts.cloudProvider) return opts.cloudProvider;
    const model = cloudTierModel(cfg, tier === 'cloud-strong' ? 'cloud-strong' : 'cloud-cheap');
    if (cache.has(model)) return cache.get(model) ?? null;
    const provider = buildCloudProvider(cfg, dataDir, model);
    cache.set(model, provider);
    return provider;
  };

  return {
    local,
    cloud: cloudReady ? (opts.cloudProvider ?? buildCloudProvider(cfg, dataDir, cloudTierModel(cfg, 'cloud-strong'))) : null,
    ...(cloudUnavailable ? { cloudUnavailable } : {}),
    cloudFor,
  };
}

/* ------------------------------------------------------------------ */
/* Disk application                                                    */
/* ------------------------------------------------------------------ */

/** Merge multi-edit reports to one final content per file, then write them. */
export function applyToWorkspace(report: VerificationReport, ctx: TaskContext): string[] {
  if (!ctx.workspace) return [];
  const final = new Map<string, string>();
  for (const f of report.applied) final.set(f.path, f.after);
  const written: string[] = [];
  const root = resolve(ctx.workspace);
  for (const [rel, content] of final) {
    const full = resolve(root, rel);
    // Defence in depth: `validateEditPath` already rejected traversal, but the
    // consequence of a mistake here is writing outside the user's project.
    if (full !== root && !full.startsWith(root + sep)) continue;
    ensureDir(dirname(full));
    writeTextAtomic(full, content);
    written.push(rel);
  }
  return written;
}

async function maybeUnload(provider: Provider): Promise<void> {
  const withUnload = provider as unknown as { unload?: () => Promise<boolean> };
  if (typeof withUnload.unload === 'function') {
    try {
      await withUnload.unload();
    } catch {
      /* unloading is best-effort */
    }
  }
}

function round6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}
