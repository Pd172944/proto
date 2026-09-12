/**
 * The agent loop.
 *
 * One task in, one episode out. The shape is:
 *
 *     route -> (local attempt -> verify -> repair) -> escalate -> verify -> record
 *
 * Five things about this loop are deliberate and worth stating, because they
 * are where a naive implementation would quietly go wrong:
 *
 *  1. **Verification, not vibes, decides success.** An attempt "succeeded" iff
 *     the verifier passed it. This is what makes `localSucceeded` a usable RL
 *     label and what makes escalation trustworthy.
 *
 *  2. **Escalation is bounded.** `maxCloudAttempts` caps worst-case spend, and
 *     the cloud budget is checked before every cloud call. A routing bug can
 *     cost the user a few cents, not a surprise bill.
 *
 *  3. **Nothing is written to the workspace unless asked.** `apply` is off by
 *     default. The verifier works on an in-memory copy of the files.
 *
 *  4. **Failures still produce training data.** A local failure followed by a
 *     cloud success is the single most valuable record in the system (it is a
 *     verified preference pair). The loop records it even when the user's task
 *     ultimately fails, and records it *before* any early return.
 *
 *  5. **Provider transport errors are not model failures.** If the local server
 *     is down we record `error` on the attempt and escalate, rather than
 *     scoring the local model as wrong. Conflating the two poisons the router's
 *     training data with "the model is bad" labels for an unrelated outage.
 */

import { buildCloudProvider, buildLocalProvider, cloudTierModel } from '../providers/index.ts';
import type { ChatResponse, Message, Provider } from '../providers/types.ts';
import { ProviderError } from '../providers/types.ts';
import { priceFor, resolveApiKeyFor } from '../config/load.ts';
import type { ProtoConfig } from '../config/schema.ts';
import { routeTask } from '../router/index.ts';
import type { RouteDecision, TaskContext, Tier } from '../router/types.ts';
import { FEATURE_VECTOR_VERSION } from '../router/types.ts';
import { verifyCandidate } from '../verify/index.ts';
import type { VerificationReport } from '../verify/types.ts';
import { EpisodeStore } from '../memory/store.ts';
import { behaviorPropensity, computeReward } from '../memory/reward.ts';
import type { AttemptRecord, AttemptSource, Episode, RedactionReport } from '../memory/types.ts';
import { EPISODE_SCHEMA_VERSION, decisionFromRoute } from '../memory/types.ts';
import { redact } from '../memory/redact.ts';
import { buildMessages, buildRepairPrompt, buildUserPrompt, repairSystemPrompt, systemPromptFor, PROMPT_VERSION } from './prompt.ts';
import { HARNESS_VERSION } from '../version.ts';
import { ensureDir, writeTextAtomic } from '../util/fsx.ts';
import { sha256Short } from '../util/text.ts';
import { ulid } from '../util/ids.ts';
import { dirname, resolve, sep } from 'node:path';

/**
 * Difficulty above which an escalation (or a cloud retry) uses the strongest
 * cloud model. Deliberately lower than the router's own cheap/strong boundary:
 * we already have evidence the task is harder than the router thought.
 */
const ESCALATION_STRONG_THRESHOLD = 0.4;

export interface RunOptions {
  cfg: ProtoConfig;
  dataDir: string;
  ctx: TaskContext;
  /** Write verified edits to disk. Off by default. */
  apply?: boolean;
  /** Force a tier, bypassing the router (still recorded in the episode). */
  forceTier?: Tier;
  /** Route only; make no model calls (except routing health probes). */
  dryRun?: boolean;
  /** Override local repair attempts. */
  maxRepair?: number;
  /** Injectable providers for tests and eval. */
  localProvider?: Provider;
  cloudProvider?: Provider;
  /** Skip routing health probes (useful offline / in tests). */
  offlineRoute?: boolean;
  /** Pre-built store (tests). */
  store?: EpisodeStore;
  /** Persist the episode. Off for eval runs that must not pollute the log. */
  persist?: boolean;
  /** Progress callback for the CLI. */
  onEvent?: (event: RunEvent) => void;
  /** Force a random draw for exploration (tests). */
  random?: () => number;
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
  | { type: 'applied'; files: string[] }
  | { type: 'recorded'; episodeId: string; reward: number };

export interface RunResult {
  decision: RouteDecision;
  report: VerificationReport | null;
  /** Final candidate text (raw model output). */
  finalText: string;
  /** Files written, when `apply` was set. */
  writtenFiles: string[];
  episode: Episode | null;
  status: Episode['outcome']['status'] | 'dry-run';
  /** True when every attempt consumed zero cloud spend. */
  usedOnlyLocal: boolean;
  warnings: string[];
}

export async function runTask(opts: RunOptions): Promise<RunResult> {
  const { cfg, dataDir, ctx } = opts;
  const store = opts.store ?? new EpisodeStore(dataDir);
  const onEvent = opts.onEvent ?? ((): void => {});
  const warnings: string[] = [];

  // --- routing -------------------------------------------------------------
  let spendToday = 0;
  if (cfg.routing.cloudBudgetUsdPerDay > 0) {
    try {
      spendToday = store.stats().spendTodayUsd;
    } catch {
      warnings.push('could not read today\'s spend from the episode log; assuming $0');
    }
  }

  let decision: RouteDecision;
  try {
    decision = await routeTask({
      cfg,
      dataDir,
      ctx,
      ...(opts.offlineRoute ? { offline: true } : {}),
      cloudSpendTodayUsd: spendToday,
      ...(opts.random ? { random: opts.random } : {}),
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
      exploration: false,
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
      episode: null,
      status: 'dry-run',
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

  const attempts: AttemptRecord[] = [];
  const redactionCounts: Record<string, number> = {};
  let redactionApplied = false;
  let charsRemoved = 0;

  const recordText = (text: string, cap: number): { text: string; hash: string } => {
    const truncated = text.length > cap;
    const capped = truncated ? text.slice(0, cap) + '\n…[truncated by memory cap]' : text;
    if (cfg.memory.redact) {
      const r = redact(capped, {
        extraPatterns: cfg.redactionPatterns,
        maxChars: cap,
      });
      redactionApplied = true;
      charsRemoved += r.charsRemoved;
      for (const [k, v] of Object.entries(r.counts)) redactionCounts[k] = (redactionCounts[k] ?? 0) + v;
      return { text: r.text, hash: sha256Short(r.text) };
    }
    return { text: capped, hash: sha256Short(capped) };
  };

  const providers = resolveProviders(opts, cfg, dataDir);

  // --- attempt machinery ---------------------------------------------------
  const runAttempt = async (input: {
    provider: Provider;
    tier: Tier;
    source: AttemptSource;
    messages: Message[];
  }): Promise<{ response: ChatResponse | null; record: AttemptRecord }> => {
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

    const promptText = input.messages.map((m) => m.content).join('\n\n');
    const storedPrompt = cfg.memory.storePrompts
      ? recordText(promptText, cfg.memory.maxPromptChars)
      : undefined;
    const storedOutput =
      response && cfg.memory.storeTaskText
        ? recordText(response.text, cfg.memory.maxOutputChars)
        : response
          ? { text: '', hash: sha256Short(response.text) }
          : undefined;

    const record: AttemptRecord = {
      n,
      tier: input.tier,
      source: input.source,
      providerId: input.provider.id,
      model: response?.model ?? input.provider.model,
      ...(storedPrompt && cfg.memory.storePrompts ? { prompt: storedPrompt.text } : {}),
      promptHash: storedPrompt?.hash ?? sha256Short(promptText),
      promptTokens: response?.usage.inputTokens ?? 0,
      outputTokens: response?.usage.outputTokens ?? 0,
      ...(response?.usage.cachedInputTokens !== undefined
        ? { cachedInputTokens: response.usage.cachedInputTokens }
        : {}),
      costUsd: response?.costUsd ?? 0,
      latencyMs: response?.latencyMs ?? 0,
      finishReason: response?.finishReason ?? 'error',
      ...(storedOutput && cfg.memory.storeTaskText ? { output: storedOutput.text } : {}),
      outputHash: storedOutput?.hash ?? '',
      verification: null,
      ...(error ? { error } : {}),
    };
    if (response?.error && !error) record.error = response.error;
    attempts.push(record);
    return { response, record };
  };

  const verifyAttempt = async (record: AttemptRecord, text: string): Promise<VerificationReport> => {
    const report = await verifyCandidate({
      text,
      ctx,
      cfg,
      dataDir,
      features: decision.features,
    });
    const failedChecks = report.checks.filter((c) => !c.ok).map((c) => c.id);
    record.verification = {
      passed: report.passed,
      score: report.score,
      blockers: report.blockers,
      failedChecks,
      durationMs: report.durationMs,
    };
    onEvent({
      type: 'verify',
      tier: record.tier,
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
  let localPassed: boolean | null = null;
  let localFailed = false;
  let refusal = false;
  let envelopeDrift = false;

  const isLocal = (t: Tier): boolean => t === 'local' || t === 'local-tiny';

  if (isLocal(decision.tier)) {
    localAttempted = true;
    const maxRepair = opts.maxRepair ?? cfg.routing.maxLocalRepairAttempts;
    let messages = baseMessages;
    let lastText = '';

    for (let round = 0; round <= maxRepair; round++) {
      const { response, record } = await runAttempt({
        provider: providers.local,
        tier: decision.tier,
        source: round === 0 ? 'initial' : 'repair',
        messages,
      });

      if (!response || response.finishReason === 'error') {
        // Transport failure: not a model-quality signal. Break to escalation.
        localFailed = true;
        localPassed = null;
        break;
      }
      lastText = response.text;
      const report = await verifyAttempt(record, response.text);
      if (report.checks.some((c) => c.id === 'refusal' && !c.ok)) refusal = true;
      if (report.checks.some((c) => c.id === 'envelope-drift' && !c.ok)) envelopeDrift = true;

      if (report.passed) {
        finalReport = report;
        finalText = response.text;
        localPassed = true;
        break;
      }

      localPassed = false;
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
      // A failed local attempt is *evidence the task is harder than estimated*,
      // so escalation bumps the cloud tier at a lower difficulty than the
      // router's own cheap/strong boundary. This is deliberate, named, and
      // different from CLOUD_STRONG_DIFFICULTY_THRESHOLD on purpose.
      const to: Tier = decision.difficulty >= ESCALATION_STRONG_THRESHOLD ? 'cloud-strong' : 'cloud-cheap';
      onEvent({
        type: 'escalate',
        from: decision.tier,
        to,
        reason: 'local attempt did not pass verification',
      });
      const cloudResult = await runCloudAttempts({
        cfg,
        dataDir,
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
      envelopeDrift = envelopeDrift || cloudResult.envelopeDrift;
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
      dataDir,
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
    envelopeDrift = cloudResult.envelopeDrift;
  }

  // --- apply ---------------------------------------------------------------
  let writtenFiles: string[] = [];
  if (opts.apply && finalReport?.passed) {
    writtenFiles = applyToWorkspace(finalReport, ctx);
    if (writtenFiles.length > 0) onEvent({ type: 'applied', files: writtenFiles });
  } else if (opts.apply && !finalReport?.passed) {
    warnings.push('refusing to write files: verification did not pass');
  }

  // --- outcome & reward ----------------------------------------------------
  const tierUsed = attempts[attempts.length - 1]?.tier ?? decision.tier;
  const status: Episode['outcome']['status'] = finalReport?.passed
    ? escalated
      ? 'escalated-cloud-success'
      : isLocal(tierUsed)
        ? 'local-success'
        : 'cloud-success'
    : refusal || localFailed || attempts.length > 0
      ? 'failed'
      : 'abandoned';

  const totalCostUsd = round6(attempts.reduce((a, r) => a + r.costUsd, 0));
  const totalLatencyMs = attempts.reduce((a, r) => a + r.latencyMs, 0);

  const rewardResult = computeReward({
    status,
    escalated,
    localAttempted,
    localPassed,
    finalScore: finalReport?.score ?? 0,
    refusal,
    envelopeDrift,
    exploration: decision.exploration,
    hadBlockers: (finalReport?.blockers.length ?? 0) > 0,
    cloudCostUsd: totalCostUsd,
  });

  // Single conversion point: the episode records the decision and the behaviour
  // propensity computed from that *same* decision object, so the two can never
  // drift apart.
  const episodeDecision = decisionFromRoute(decision);

  const episode: Episode = {
    id: ulid(),
    schemaVersion: EPISODE_SCHEMA_VERSION,
    ts: new Date().toISOString(),
    harnessVersion: HARNESS_VERSION,
    platform: `${process.platform}-${process.arch}`,
    ...(cfg.memory.storeTaskText ? { task: recordText(ctx.task, 4000).text } : {}),
    taskHash: sha256Short(ctx.task),
    ...(cfg.memory.storePrompts ? { systemPrompt: recordText(system.content, 4000).text } : {}),
    systemPromptHash: sha256Short(system.content),
    features: decision.features,
    vector: decision.vector,
    vectorVersion: FEATURE_VECTOR_VERSION,
    decision: episodeDecision,
    environment: {
      // Recorded from the *router's* environment, not from the provider set. They
      // normally agree, but injected providers (tests, eval) can differ, and the
      // episode must describe what the policy actually saw or replay becomes
      // unsound.
      localAvailable: decision.env.localAvailable,
      localModel: cfg.local.model,
      localContextWindow: cfg.local.contextWindow,
      localModelLoaded: decision.env.localModelLoaded,
      cloudAvailable: decision.env.cloudAvailable,
      cloudModel: cloudTierModel(cfg, 'cloud-strong'),
      cloudProvider: cfg.cloud.provider,
      cloudPrice: priceFor(cfg, cloudTierModel(cfg, 'cloud-strong')),
      configuredQualityFloor: cfg.routing.qualityFloor,
      verifierAvailable: cfg.verify.enabled,
      routingMode: cfg.routing.mode,
      explorationEpsilon: cfg.routing.exploration.epsilon,
    },
    attempts,
    outcome: {
      status,
      finalTier: tierUsed,
      escalated,
      localSucceeded: localAttempted ? localPassed : null,
      totalCostUsd,
      totalLatencyMs,
      reward: rewardResult.reward,
      rewardVersion: rewardResult.version,
      behaviorPropensity: behaviorPropensity(episodeDecision, cfg.routing.exploration.epsilon),
    },
    consent: {
      localTraining: cfg.train.enabled,
      globalShare: cfg.contrib.enabled,
    },
    redaction: {
      applied: redactionApplied,
      counts: redactionCounts,
      charsRemoved,
    } satisfies RedactionReport,
    tags: [PROMPT_VERSION],
  };

  if (opts.persist !== false && cfg.memory.enabled) {
    try {
      store.appendBounded(episode, cfg.memory.maxShardBytes);
      onEvent({ type: 'recorded', episodeId: episode.id, reward: rewardResult.reward });
    } catch (err) {
      warnings.push(`failed to persist episode: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  if (opts.unloadLocalAfter && providers.local.kind === 'local') {
    await maybeUnload(providers.local);
  }

  return {
    decision,
    report: finalReport,
    finalText,
    writtenFiles,
    episode,
    status,
    usedOnlyLocal: totalCostUsd === 0,
    warnings,
  };
}

/* ------------------------------------------------------------------ */
/* Cloud attempts (with bounded retry)                                 */
/* ------------------------------------------------------------------ */

async function runCloudAttempts(input: {
  cfg: ProtoConfig;
  dataDir: string;
  tier: Tier;
  decision: RouteDecision;
  baseMessages: Message[];
  userPrompt: string;
  providers: ProviderBundle;
  runAttempt: (i: { provider: Provider; tier: Tier; source: AttemptSource; messages: Message[] }) => Promise<{
    response: ChatResponse | null;
    record: AttemptRecord;
  }>;
  verifyAttempt: (record: AttemptRecord, text: string) => Promise<VerificationReport>;
  warnings: string[];
}): Promise<{ report: VerificationReport | null; text: string; refusal: boolean; envelopeDrift: boolean }> {
  const { cfg, decision, providers } = input;
  let report: VerificationReport | null = null;
  let text = '';
  let refusal = false;
  let envelopeDrift = false;

  const maxAttempts = Math.max(1, cfg.routing.maxCloudAttempts);
  let tier: Tier = input.tier;

  for (let i = 0; i < maxAttempts; i++) {
    // Escalate the cloud tier on a retry only when the task looks hard; otherwise
    // a cheap model retry is the cost-appropriate response.
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

    const { response, record } = await input.runAttempt({
      provider,
      tier,
      source: i === 0 ? 'escalation' : 'repair',
      messages,
    });

    if (!response || response.finishReason === 'error') {
      input.warnings.push(`cloud attempt ${i + 1} failed at the transport level${record.error ? `: ${record.error}` : ''}`);
      continue;
    }
    text = response.text;
    report = await input.verifyAttempt(record, response.text);
    if (report.checks.some((c) => c.id === 'refusal' && !c.ok)) refusal = true;
    if (report.checks.some((c) => c.id === 'envelope-drift' && !c.ok)) envelopeDrift = true;
    if (report.passed) return { report, text, refusal, envelopeDrift };
  }

  if (report && !report.passed) {
    input.warnings.push('the cloud answer also failed verification; returning it as an unverified best effort');
  }
  return { report, text, refusal, envelopeDrift };
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
