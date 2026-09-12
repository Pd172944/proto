/**
 * Learning-curve simulation.
 *
 * ## What this answers
 *
 * "How many tasks do I have to do before this is meaningfully better than the
 * downloaded base model?" is the central empirical question about the project,
 * and it cannot be answered by reading code. It needs a curve.
 *
 * ## What is real and what is simulated
 *
 * Real:
 *  - the router, its 43-feature vector, the policy, the verifier plumbing, the
 *    episode schema, the reward, the dataset builder, and **the actual
 *    logistic-regression trainer** the fast loop uses, evaluated on a held-out
 *    set with AUC / Brier / routing accuracy.
 *
 * Simulated:
 *  - the *user* (a stochastic draw from the task corpus), and
 *  - the *local model's competence*.
 *
 * ## The one design decision that makes this informative
 *
 * The simulated local model does **not** simply get better at easy tasks: it has
 * **per-task-class affinities** drawn from a fixed seed. Real local models are
 * idiosyncratic — a 1.5B model can be surprisingly good at mechanical renames and
 * surprisingly bad at writing tests, in a way its *difficulty* alone does not
 * predict.
 *
 * This matters because of what it implies. If the synthetic labels were a pure
 * function of difficulty, the heuristic prior would *be* the Bayes-optimal model
 * and no amount of learning could beat it: the simulation would be rigged and
 * would answer nothing. By placing the truth partly outside the difficulty scale,
 * the simulation tests the project's actual claim — that the harness can discover
 * which tasks *your* local model handles, from observation alone.
 *
 * The affinities are printed in the report, so you can see exactly what the
 * router is being asked to discover.
 *
 * ## Isolation
 *
 * Simulated episodes are written to a scratch directory under the data dir and
 * deleted afterwards (kept only with `--write`). A simulation must never
 * contaminate the real episode log or the real router weights, because that log
 * is what trains the deployed system.
 */

import { rmdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';

import { EVAL_TASKS } from '../eval/tasks.ts';
import type { EvalTask } from '../eval/tasks.ts';
import type { ProtoConfig } from '../config/schema.ts';
import { buildDatasets } from '../memory/datasets.ts';
import { fakeEpisode } from '../memory/factory.ts';
import { EpisodeStore } from '../memory/store.ts';
import { extractFeatures, toVector } from '../router/features.ts';
import { blendWithPrior, heuristicScore } from '../router/heuristic.ts';
import { LogisticScorer, MIN_TRAINING_SAMPLES, evaluate, trainLogistic } from '../router/learned.ts';
import { decideRoute } from '../router/policy.ts';
import type { RouterEnvironment } from '../router/policy.ts';
import type { TaskClass, TaskFeatures } from '../router/types.ts';
import { ensureDir } from '../util/fsx.ts';

/* ------------------------------------------------------------------ */
/* Options and result types                                            */
/* ------------------------------------------------------------------ */

export type Competence = 'pessimistic' | 'realistic' | 'optimistic';

export interface SimulateOptions {
  tasks?: number;
  checkpoints?: number[];
  seed?: number;
  competence?: Competence;
  /** Probability the verifier catches a genuinely wrong answer. */
  verifierCatchRate?: number;
  /** Probability a correct answer is wrongly rejected. */
  verifierFalseFailureRate?: number;
  /** Probability the cloud fixes what local failed. */
  cloudSuccessRate?: number;
  /** How much per-class idiosyncrasy the simulated local model has (0 = none). */
  classAffinitySpread?: number;
  /** Let the learned router drive routing once trained (the flywheel). */
  adaptive?: boolean;
  /** Keep the scratch episode directory instead of deleting it. */
  writeEpisodes?: boolean;
}

export interface SimulateCheckpoint {
  tasks: number;
  labels: number;
  localRouted: number;
  localRoutedPct: number;
  trueLocalSuccessRate: number;
  verifiedPassRate: number;
  escalations: number;
  falsePasses: number;
  falsePassRate: number;
  cloudSpendUsd: number;
  sftSamples: number;
  dpoPairs: number;
  sftDistinctClasses: number;
  routerTrained: boolean;
  /** AUC on held-out tasks: the raw learned scorer. */
  learnedAuc: number | null;
  /** AUC of the deployed default (learned blended with the heuristic prior). */
  hybridAuc: number | null;
  /** AUC of the heuristic prior, the thing that must be beaten. */
  heuristicAuc: number;
  /** AUC of "just use difficulty", the cheapest possible baseline. */
  difficultyOnlyAuc: number;
  learnedBrier: number | null;
  learnedRoutingAccuracy: number | null;
  hybridRoutingAccuracy: number | null;
  heuristicRoutingAccuracy: number;
  /** Gain over the heuristic from the deployed (hybrid) scorer. */
  deployedGainOverHeuristic: number | null;
}

export interface SimulateResult {
  options: Required<Omit<SimulateOptions, 'checkpoints'>> & { checkpoints: number[] };
  checkpoints: SimulateCheckpoint[];
  /** First checkpoint where the deployed scorer beats the heuristic by >0.02 AUC. */
  markedImpactAt: number | null;
  firstTrainingAt: number | null;
  dpoReadyAt: number | null;
  /** Per-class multipliers the router has to discover. */
  affinities: Array<{ taskClass: string; affinity: number; note: string }>;
  heldoutSize: number;
  scratchDir: string | null;
  notes: string[];
  interpretation: string[];
}

/* ------------------------------------------------------------------ */
/* Synthetic ground truth                                              */
/* ------------------------------------------------------------------ */

const COMPETENCE_OFFSET: Record<Competence, number> = {
  pessimistic: -0.18,
  realistic: 0,
  optimistic: 0.18,
};

export interface TruthModel {
  /** Per-class multiplier on the difficulty-implied success probability. */
  affinities: Map<TaskClass, number>;
  competence: Competence;
}

export function buildTruthModel(seed: number, competence: Competence, spread: number): TruthModel {
  const rand = mulberry32(seed ^ 0xa77117);
  const affinities = new Map<TaskClass, number>();
  const classes = new Set<TaskClass>(EVAL_TASKS.map((t) => classifyOf(t)));
  for (const cls of classes) {
    // Log-normal-ish around 1.0, clipped: a local model is never 3x better than
    // its difficulty suggests, nor 3x worse.
    const raw = 1 + (rand() * 2 - 1) * spread;
    affinities.set(cls, Math.min(1.6, Math.max(0.4, raw)));
  }
  return { affinities, competence };
}

export function trueSuccessProbability(difficulty: number, affinity: number, competence: Competence): number {
  const base = 1.02 - 1.15 * difficulty + COMPETENCE_OFFSET[competence];
  const raw = base * affinity;
  return Math.min(0.97, Math.max(0.02, raw));
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ------------------------------------------------------------------ */
/* Task population                                                     */
/* ------------------------------------------------------------------ */

interface Template {
  id: string;
  features: TaskFeatures;
  taskClass: TaskClass;
  difficulty: number;
  heuristicP: number;
  affinity: number;
  /** Templates a real user hits most often (small mechanical edits). */
  common: boolean;
}

const CLASS_CACHE = new Map<string, TaskClass>();

function featuresOf(t: EvalTask): TaskFeatures {
  return extractFeatures({
    task: t.task,
    ...(t.files ? { files: t.files } : {}),
    ...(t.diff ? { diff: t.diff } : {}),
    ...(t.constraints ? { constraints: t.constraints } : {}),
  });
}

function classifyOf(t: EvalTask): TaskClass {
  const cached = CLASS_CACHE.get(t.id);
  if (cached) return cached;
  const cls = featuresOf(t).taskClass;
  CLASS_CACHE.set(t.id, cls);
  return cls;
}

function buildTemplates(truth: TruthModel): Template[] {
  return EVAL_TASKS.map((t: EvalTask) => {
    const features = featuresOf(t);
    const h = heuristicScore(features);
    return {
      id: t.id,
      features,
      taskClass: features.taskClass,
      difficulty: h.difficulty,
      heuristicP: h.pLocalSuccess,
      affinity: truth.affinities.get(features.taskClass) ?? 1,
      common: t.expected === 'local',
    };
  });
}

interface HeldoutRow {
  x: number[];
  y: 0 | 1;
  difficulty: number;
  heuristicP: number;
}

function buildHeldout(templates: Template[], truth: TruthModel, size: number, rand: () => number): HeldoutRow[] {
  return Array.from({ length: size }, () => {
    const tpl = sampleTemplate(templates, rand);
    const p = trueSuccessProbability(tpl.difficulty, tpl.affinity, truth.competence);
    return {
      x: toVector(tpl.features),
      y: (rand() < p ? 1 : 0) as 0 | 1,
      difficulty: tpl.difficulty,
      heuristicP: tpl.heuristicP,
    };
  });
}

/** Real requests skew small: most are mechanical, a minority are real work. */
function sampleTemplate(templates: Template[], rand: () => number): Template {
  const wantCommon = rand() < 0.65;
  const pool = templates.filter((t) => t.common === wantCommon);
  const use = pool.length > 0 ? pool : templates;
  return use[Math.floor(rand() * use.length)] as Template;
}

/* ------------------------------------------------------------------ */
/* Simulation                                                          */
/* ------------------------------------------------------------------ */

export function simulate(cfg: ProtoConfig, dataDir: string, opts: SimulateOptions = {}): SimulateResult {
  const tasks = opts.tasks ?? 400;
  const seed = opts.seed ?? 20250601;
  const competence = opts.competence ?? 'realistic';
  const verifierCatchRate = opts.verifierCatchRate ?? 0.85;
  const verifierFalseFailureRate = opts.verifierFalseFailureRate ?? 0.05;
  const cloudSuccessRate = opts.cloudSuccessRate ?? 0.9;
  const classAffinitySpread = opts.classAffinitySpread ?? 0.3;
  const adaptive = opts.adaptive ?? false;
  const keepEpisodes = opts.writeEpisodes ?? false;

  const checkpoints = resolveCheckpoints(opts.checkpoints, tasks);

  const truth = buildTruthModel(seed, competence, classAffinitySpread);
  const templates = buildTemplates(truth);
  const heldout = buildHeldout(templates, truth, 800, mulberry32(seed ^ 0x5eed));
  const rand = mulberry32(seed);

  // Simulated episodes live in their own directory so they can never contaminate
  // the real log or the real router weights.
  const scratchDir = ensureDir(join(dataDir, 'sim', `run-${seed}-${tasks}`));
  const store = new EpisodeStore(scratchDir);

  const env: RouterEnvironment = {
    localEnabled: true,
    localAvailable: true,
    localModelLoaded: true,
    localContextWindow: cfg.local.contextWindow,
    tinyModelAvailable: Boolean(cfg.local.tinyModel),
    cloudAvailable: true,
    verifierAvailable: true,
    cloudBudgetRemainingUsd: 1_000_000,
    price: cfg.pricing[cfg.cloud.model] ?? { in: 3, out: 15 },
    // Exploration stays off and is exercised by its own tests; mixing it in here
    // would confound the data-volume curve with the exploration rate.
    random: () => 1,
  };

  const counters = {
    labels: 0,
    localRouted: 0,
    trueSuccess: 0,
    verifiedPass: 0,
    escalations: 0,
    falsePasses: 0,
    cloudSpendUsd: 0,
  };

  let policyScorer: LogisticScorer | null = null;
  const results: SimulateCheckpoint[] = [];

  const snapshot = (atTasks: number): SimulateCheckpoint => {
    const datasets = buildDatasets(store, cfg, {
      write: false,
      maxSft: cfg.train.maxSftSamples,
      maxDpo: cfg.train.maxDpoSamples,
    });
    const samples = datasets.router.samples.map((s) => ({ x: s.x, y: s.y, w: s.w }));
    const distinctClasses = new Set(datasets.router.samples.map((s) => s.meta.taskClass)).size;

    // Judge routing with the rule the system actually deploys: send local when the
    // score clears the quality floor. Using a 0.5 cutoff here would measure a
    // decision the policy never makes.
    const floor = cfg.routing.qualityFloor;

    const heuristicAuc = aucOf(heldout.map((r) => ({ p: r.heuristicP, y: r.y })));
    const difficultyOnlyAuc = aucOf(heldout.map((r) => ({ p: 1 - r.difficulty, y: r.y })));

    let trained: ReturnType<typeof trainLogistic> | null = null;
    if (samples.length >= MIN_TRAINING_SAMPLES) {
      trained = trainLogistic(samples, { epochs: 400, learningRate: 0.08, l2: 0.02, seed: 12345 });
    }

    let learnedAuc: number | null = null;
    let hybridAuc: number | null = null;
    let learnedBrier: number | null = null;
    let learnedRoutingAccuracy: number | null = null;
    let hybridRoutingAccuracy: number | null = null;

    if (trained) {
      learnedAuc = aucOf(heldout.map((r) => ({ p: trained.scorer.predict(r.x), y: r.y })));
      learnedBrier = evaluate(trained.scorer, heldout.map((r) => ({ x: r.x, y: r.y }))).brier;

      // The deployed default is *hybrid*: the learned score shrunk toward the
      // heuristic prior. Measuring it matters, because the prior is strong and the
      // blend is what a user actually runs.
      const hybridScore = (r: HeldoutRow): number =>
        blendWithPrior(trained.scorer.predict(r.x), r.heuristicP, samples.length).p;
      hybridAuc = aucOf(heldout.map((r) => ({ p: hybridScore(r), y: r.y })));
      learnedRoutingAccuracy = routeAccuracy(heldout, (r) => trained.scorer.predict(r.x), floor);
      hybridRoutingAccuracy = routeAccuracy(heldout, hybridScore, floor);
    }

    return {
      tasks: atTasks,
      labels: counters.labels,
      localRouted: counters.localRouted,
      localRoutedPct: atTasks ? counters.localRouted / atTasks : 0,
      trueLocalSuccessRate: counters.localRouted ? counters.trueSuccess / counters.localRouted : 0,
      verifiedPassRate: counters.localRouted ? counters.verifiedPass / counters.localRouted : 0,
      escalations: counters.escalations,
      falsePasses: counters.falsePasses,
      falsePassRate: counters.localRouted ? counters.falsePasses / counters.localRouted : 0,
      cloudSpendUsd: round6(counters.cloudSpendUsd),
      sftSamples: datasets.sft.samples.length,
      dpoPairs: datasets.dpo.samples.length,
      sftDistinctClasses: distinctClasses,
      routerTrained: trained !== null,
      learnedAuc,
      hybridAuc,
      heuristicAuc,
      difficultyOnlyAuc,
      learnedBrier,
      learnedRoutingAccuracy,
      hybridRoutingAccuracy,
      heuristicRoutingAccuracy: routeAccuracy(heldout, (r) => r.heuristicP, floor),
      deployedGainOverHeuristic: hybridAuc === null ? null : round4(hybridAuc - heuristicAuc),
    };
  };

  results.push(snapshot(0));

  for (let i = 1; i <= tasks; i++) {
    const tpl = sampleTemplate(templates, rand);

    // Until the learner has enough data the behaviour policy is the heuristic.
    // `--adaptive` lets the deployed policy take over as it learns: the flywheel,
    // where better routing produces data about the policy's remaining mistakes.
    const delegationCfg: ProtoConfig = adaptive
      ? cfg
      : { ...cfg, routing: { ...cfg.routing, mode: 'heuristic' } };

    const decision = decideRoute({
      ctx: { task: '' },
      features: tpl.features,
      cfg: delegationCfg,
      env,
      scorer: adaptive ? policyScorer : null,
    });

    const trueP = trueSuccessProbability(tpl.difficulty, tpl.affinity, competence);
    const truthSucceeded = rand() < trueP;
    const isLocal = decision.tier === 'local' || decision.tier === 'local-tiny';

    if (!isLocal) {
      // Cloud-first: no local attempt, therefore no label. This is why the router
      // learns slowly about the tasks it currently avoids.
      counters.cloudSpendUsd += 0.004;
      store.appendBounded(
        fakeEpisode({
          task: instanceTask(tpl, i),
          features: tpl.features,
          tier: decision.tier,
          status: 'cloud-success',
          localSucceeded: null,
          reward: 0.9,
          cloudCostUsd: 0.004,
        }),
        cfg.memory.maxShardBytes,
      );
      if (checkpoints.includes(i)) results.push(snapshot(i));
      continue;
    }

    counters.localRouted++;

    // Verification. A correct answer usually passes; a wrong one is usually
    // caught. A wrong one that is *not* caught is a false pass — the metric that
    // represents the real risk of routing work to a cheap model.
    const verified = truthSucceeded ? rand() >= verifierFalseFailureRate : rand() < 1 - verifierCatchRate;
    if (truthSucceeded) counters.trueSuccess++;
    if (verified) counters.verifiedPass++;
    if (!truthSucceeded && verified) counters.falsePasses++;
    counters.labels++;

    if (verified) {
      store.appendBounded(
        fakeEpisode({
          task: instanceTask(tpl, i),
          features: tpl.features,
          tier: decision.tier,
          status: 'local-success',
          localSucceeded: true,
          verified: true,
          reward: 1.2,
        }),
        cfg.memory.maxShardBytes,
      );
    } else {
      counters.escalations++;
      const cloudWon = rand() < cloudSuccessRate;
      counters.cloudSpendUsd += 0.004;
      const ep = fakeEpisode({
        task: instanceTask(tpl, i),
        features: tpl.features,
        tier: decision.tier,
        status: cloudWon ? 'escalated-cloud-success' : 'failed',
        localSucceeded: false,
        verified: false,
        escalated: true,
        reward: cloudWon ? 0.4 : -0.6,
        cloudCostUsd: 0.004,
      });
      if (cloudWon) {
        // A real escalation records a second, passing cloud attempt; that pair is
        // what the DPO dataset consumes.
        ep.attempts.push({
          ...(ep.attempts[0] as (typeof ep.attempts)[number]),
          n: 2,
          tier: 'cloud-strong',
          source: 'escalation',
          providerId: 'openrouter',
          model: cfg.cloud.model,
          output: `{"summary":"cloud fix for ${tpl.id}","edits":[{"file":"prices.py","content":"for item in items:\\n    pass\\n"}]}`,
          outputHash: `cloud-${tpl.id}-${i}`,
          costUsd: 0.004,
          verification: { passed: true, score: 1, blockers: [], failedChecks: [], durationMs: 2000 },
        });
      }
      store.appendBounded(ep, cfg.memory.maxShardBytes);
    }

    if (checkpoints.includes(i)) {
      const snap = snapshot(i);
      results.push(snap);
      if (adaptive && snap.routerTrained) {
        const datasets = buildDatasets(store, cfg, { write: false });
        const samples = datasets.router.samples.map((s) => ({ x: s.x, y: s.y, w: s.w }));
        if (samples.length >= MIN_TRAINING_SAMPLES) {
          policyScorer = trainLogistic(samples, { epochs: 400, learningRate: 0.08, l2: 0.02, seed: 12345 }).scorer;
        }
      }
    }
  }

  if (!keepEpisodes) {
    try {
      rmSync(scratchDir, { recursive: true, force: true });
      // Tidy the parent too when it is empty, so repeated simulations do not
      // accumulate empty directories. `rmdirSync` is non-recursive, so it fails
      // harmlessly if another run's episodes are still there.
      rmdirSync(join(dataDir, 'sim'));
    } catch {
      /* best effort: a leftover empty directory is not worth failing a run over */
    }
  }

  const marked = results.find((c) => c.deployedGainOverHeuristic !== null && c.deployedGainOverHeuristic > 0.02);

  return {
    options: {
      tasks,
      checkpoints,
      seed,
      competence,
      verifierCatchRate,
      verifierFalseFailureRate,
      cloudSuccessRate,
      classAffinitySpread,
      adaptive,
      writeEpisodes: keepEpisodes,
    },
    checkpoints: results,
    markedImpactAt: marked?.tasks ?? null,
    firstTrainingAt: results.find((c) => c.routerTrained)?.tasks ?? null,
    dpoReadyAt: results.find((c) => c.dpoPairs >= 30)?.tasks ?? null,
    affinities: [...truth.affinities.entries()]
      .map(([taskClass, affinity]) => ({
        taskClass,
        affinity: round4(affinity),
        note:
          affinity > 1.12
            ? 'simulated model is better here than difficulty suggests'
            : affinity < 0.88
              ? 'simulated model is worse here than difficulty suggests'
              : 'roughly as difficulty suggests',
      }))
      .sort((a, b) => b.affinity - a.affinity),
    heldoutSize: heldout.length,
    scratchDir: keepEpisodes ? scratchDir : null,
    notes: [
      'The router, feature vector, policy, verifier plumbing, dataset builder and logistic-regression trainer are the real implementations; only the user and the local model are synthesised.',
      'The simulated local model has per-class idiosyncrasies (the affinity table below) that the heuristic cannot see. That is what makes learning possible at all — a model whose skill were a pure function of difficulty could never be beaten.',
      'Ground-truth labels are used for evaluation, while the system trains on verification verdicts, so the gap between "true-ok" and "verified" is the verifier error rate.',
      'This is a model of the system, not a measurement of a specific checkpoint. Read the slopes as informative and the levels as a scenario.',
    ],
    interpretation: interpret(results, truth, cfg.train.lora),
  };
}

/**
 * Accuracy of the deployed routing rule: go local when the score clears the
 * quality floor. A held-out task is "correct" when local was chosen and the
 * simulated model really could do it, or when the cloud was chosen and it could
 * not. That is the decision the user experiences, and it is what a probability
 * floor — not a 0.5 cutoff — determines.
 */
function routeAccuracy(rows: HeldoutRow[], score: (r: HeldoutRow) => number, floor: number): number {
  const correct = rows.filter((r) => (score(r) >= floor ? r.y === 1 : r.y === 0)).length;
  return correct / rows.length;
}

function interpret(rows: SimulateCheckpoint[], truth: TruthModel, lora: ProtoConfig['train']['lora']): string[] {
  const out: string[] = [];
  const last = rows[rows.length - 1];
  if (!last) return out;

  if (!last.routerTrained) {
    out.push(
      `After ${last.tasks} tasks there are ${last.labels} labelled local attempts — below the ${MIN_TRAINING_SAMPLES} ` +
        'needed before the learned scorer is used at all. The router is still the heuristic prior.',
    );
  } else {
    out.push(
      `After ${last.tasks} tasks the deployed (hybrid) scorer reaches AUC ${(last.hybridAuc ?? 0).toFixed(3)} on ` +
        `held-out tasks, versus ${last.heuristicAuc.toFixed(3)} for the heuristic prior and ` +
        `${last.difficultyOnlyAuc.toFixed(3)} for difficulty alone.`,
    );
    out.push(
      `Raw learned scorer: AUC ${(last.learnedAuc ?? 0).toFixed(3)} (Brier ${(last.learnedBrier ?? 0).toFixed(3)}). ` +
        (last.learnedAuc !== null && last.learnedAuc < last.heuristicAuc
          ? 'It is currently WORSE than the hand-written prior — the expected outcome at this data volume, and exactly ' +
            'why the deployed default shrinks the learned score toward the prior instead of trusting it outright.'
          : 'It has passed the prior, which is the point at which the learned scorer starts earning its keep.'),
    );
    out.push(
      `Routing accuracy on held-out tasks: hybrid ${((last.hybridRoutingAccuracy ?? 0) * 100).toFixed(1)}%, ` +
        `heuristic ${(last.heuristicRoutingAccuracy * 100).toFixed(1)}%.`,
    );
  }

  out.push(
    `LoRA side after ${last.tasks} tasks: ${last.sftSamples} SFT sample(s) (verified local wins plus distilled cloud ` +
      `wins) and ${last.dpoPairs} preference pair(s) across ${last.sftDistinctClasses} task class(es).`,
  );
  // Steps and epochs, not row counts, decide whether fine-tuning can learn
  // anything. A row count without them is the number people quote and the one
  // that means least.
  const batchSize = Math.max(1, lora.batchSize);
  const epochs = (iters: number, rows: number): number => (iters * batchSize) / Math.max(1, rows);
  const sftIters = Math.min(lora.maxIters, Math.ceil((lora.epochs * last.sftSamples) / batchSize));
  const dpoIters = Math.min(lora.maxIters, Math.ceil((lora.epochs * last.dpoPairs) / batchSize));
  out.push(
    `At the default ${lora.epochs}-epoch target that is ${sftIters} SFT step(s) ` +
      `(${epochs(sftIters, last.sftSamples).toFixed(1)} epoch) or ${dpoIters} DPO step(s) ` +
      `(${epochs(dpoIters, last.dpoPairs).toFixed(1)} epoch) at batch ${batchSize}.`,
  );
  if (last.dpoPairs < 200) {
    out.push(
      'That is still thin. Preference tuning generally needs several hundred consistent pairs before behaviour ' +
        'visibly changes, so expect drift toward your conventions rather than a capability jump.',
    );
  }
  if (last.falsePassRate > 0.02) {
    out.push(
      `Verifier risk: ${(last.falsePassRate * 100).toFixed(1)}% of local attempts were wrong and were accepted anyway. ` +
        'That is the true cost of cheap routing, and the argument for enabling verify.runTests.',
    );
  }
  out.push(
    `The learned scorer is chasing ${[...truth.affinities.values()].filter((a) => a > 1.12 || a < 0.88).length} ` +
      'per-class idiosyncrasies of your local model (see the affinity table) — that, rather than general coding ' +
      'ability, is what it can realistically learn.',
  );
  return out;
}

function resolveCheckpoints(explicit: number[] | undefined, tasks: number): number[] {
  const base = explicit ?? [10, 25, 50, 100, 200, 400, 800, 1600, 3200];
  const inRange = base.filter((c) => c > 0 && c <= tasks);
  if (!inRange.includes(tasks)) inRange.push(tasks);
  return [...new Set(inRange)].sort((a, b) => a - b);
}

/**
 * A distinct task string per simulated instance.
 *
 * Real users do not type the same sentence twice, and the dataset builder
 * deliberately deduplicates identical tasks so that one repeated request cannot
 * dominate training. Reusing a template id verbatim would collapse the whole
 * simulation onto ~22 rows and understate the learning curve.
 */
function instanceTask(tpl: Template, i: number): string {
  return `${tpl.id} (variation ${i})`;
}

/** AUC via the Mann-Whitney U statistic with average ranks for ties. */
export function aucOf(rows: Array<{ p: number; y: 0 | 1 }>): number {
  const pos = rows.filter((r) => r.y === 1).length;
  const neg = rows.length - pos;
  if (pos === 0 || neg === 0) return 0.5;
  const sorted = [...rows].sort((a, b) => a.p - b.p);
  const n = sorted.length;
  const ranks = new Array<number>(n).fill(0);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && (sorted[j + 1] as { p: number }).p === (sorted[i] as { p: number }).p) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[k] = avg;
    i = j + 1;
  }
  let rankSum = 0;
  for (let k = 0; k < n; k++) if ((sorted[k] as { y: number }).y === 1) rankSum += ranks[k] as number;
  return (rankSum - (pos * (pos + 1)) / 2) / (pos * neg);
}

function round4(n: number): number {
  return Math.round(n * 10_000) / 10_000;
}

function round6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}

/* ------------------------------------------------------------------ */
/* Reporting                                                           */
/* ------------------------------------------------------------------ */

export function formatSimulationReport(result: SimulateResult): string[] {
  const lines: string[] = [];
  const pct = (n: number): string => `${(n * 100).toFixed(1)}%`;
  const num = (n: number | null, digits = 3): string => (n === null ? '-' : n.toFixed(digits));

  lines.push(
    `simulated ${result.options.tasks} tasks (seed ${result.options.seed}, competence "${result.options.competence}"` +
      `${result.options.adaptive ? ', adaptive routing' : ''})`,
  );
  lines.push(`held-out evaluation: ${result.heldoutSize} tasks with ground-truth labels from the same population`);
  lines.push('');
  lines.push('   tasks  labels  local%  true-ok  verified  escal  falsePass  sft   dpo  hybridAUC  learnedAUC  heurAUC  hybridAcc');
  for (const c of result.checkpoints) {
    lines.push(
      `  ${String(c.tasks).padStart(6)}  ${String(c.labels).padStart(6)}  ${pct(c.localRoutedPct).padStart(6)}  ` +
        `${pct(c.trueLocalSuccessRate).padStart(7)}  ${pct(c.verifiedPassRate).padStart(8)}  ` +
        `${String(c.escalations).padStart(5)}  ${pct(c.falsePassRate).padStart(9)}  ` +
        `${String(c.sftSamples).padStart(3)}  ${String(c.dpoPairs).padStart(4)}  ` +
        `${num(c.hybridAuc).padStart(9)}  ${num(c.learnedAuc).padStart(10)}  ${c.heuristicAuc.toFixed(3).padStart(7)}  ` +
        `${(c.hybridRoutingAccuracy === null ? '-' : pct(c.hybridRoutingAccuracy)).padStart(9)}`,
    );
  }
  lines.push('');
  lines.push(`first training run:          ${result.firstTrainingAt === null ? 'not reached' : `after ~${result.firstTrainingAt} tasks`}`);
  lines.push(
    `deployed scorer beats prior: ${result.markedImpactAt === null ? 'not yet (>0.02 AUC) in this run' : `after ~${result.markedImpactAt} tasks`}`,
  );
  lines.push(`30+ DPO pairs available:     ${result.dpoReadyAt === null ? 'not reached' : `after ~${result.dpoReadyAt} tasks`}`);
  if (result.scratchDir) lines.push(`episodes kept at:            ${result.scratchDir}`);

  lines.push('');
  lines.push('what the router has to discover (simulated per-class affinities):');
  lines.push('  task class          affinity  meaning');
  for (const a of result.affinities) {
    lines.push(`  ${a.taskClass.padEnd(18)}  ${a.affinity.toFixed(3).padStart(8)}  ${a.note}`);
  }

  lines.push('');
  lines.push('what this means:');
  for (const line of result.interpretation) lines.push(`  - ${line}`);
  lines.push('');
  lines.push('caveats:');
  for (const line of result.notes) lines.push(`  - ${line}`);
  return lines;
}
