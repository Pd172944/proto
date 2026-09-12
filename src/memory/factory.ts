/**
 * Synthetic episode factory.
 *
 * This is **not** on the production path — `harness/loop.ts` builds episodes from
 * real model interactions. It exists so that tests and `proto simulate` can
 * produce schema-valid episodes, and it lives in `src/` rather than in `test/`
 * for one specific reason: if the Episode schema changes, both the test suite and
 * the learning-curve simulation must break loudly together rather than drifting
 * apart. It is deliberately explicit rather than "smart" for the same reason.
 *
 * Every field the schema requires is set here, so a new required field surfaces
 * as a compile error in exactly one place.
 */

import type { Episode, AttemptRecord } from './types.ts';
import { EPISODE_SCHEMA_VERSION, REWARD_VERSION } from './types.ts';
import { extractFeatures, toVector } from '../router/features.ts';
import { FEATURE_VECTOR_VERSION } from '../router/types.ts';
import { ulid } from '../util/ids.ts';
import { sha256Short } from '../util/text.ts';

export interface FakeEpisodeInput {
  task?: string;
  /**
   * Authoritative features. When supplied, `task` is used only as a label and for
   * hashing, and the vector is taken from these features. The learning-curve
   * simulation needs this: it varies the task *text* so that dataset dedup sees
   * genuinely distinct tasks (as a real user's would be) while holding the
   * *features* fixed to a known template.
   */
  features?: ReturnType<typeof extractFeatures>;
  files?: Array<{ path: string; content: string }>;
  tier?: 'local' | 'local-tiny' | 'cloud-cheap' | 'cloud-strong';
  status?: Episode['outcome']['status'];
  localSucceeded?: boolean | null;
  escalated?: boolean;
  exploration?: boolean;
  reward?: number;
  cloudCostUsd?: number;
  verified?: boolean;
  prompt?: string;
  output?: string;
  ts?: string;
  behaviorPropensity?: number;
}

/**
 * Build a minimally-valid Episode for store/dataset tests.
 *
 * Kept deliberately explicit rather than "smart": when the Episode schema grows,
 * this file fails to compile and the tests are forced to acknowledge the change.
 */
export function fakeEpisode(input: FakeEpisodeInput = {}): Episode {
  const task = input.task ?? 'Fix the off-by-one error in this loop so it does not go out of bounds';
  const files = input.files ?? [{ path: 'prices.py', content: 'for i in range(len(items) + 1):\n    pass\n' }];
  const features = input.features ?? extractFeatures({ task, files });
  const tier = input.tier ?? 'local';
  const isLocal = tier === 'local' || tier === 'local-tiny';
  const status = input.status ?? (input.localSucceeded === false ? 'failed' : 'local-success');
  const prompt = input.prompt ?? `TASK:\n${task}\n\nFILE: ${files[0]?.path}\n\`\`\`\n${files[0]?.content}\n\`\`\``;
  const output = input.output ?? '{"summary":"mock","edits":[{"file":"prices.py","content":"ok"}],"risk":"low"}';
  const verified = input.verified ?? status !== 'failed';

  const attempt: AttemptRecord = {
    n: 1,
    tier,
    source: 'initial',
    providerId: isLocal ? 'local' : 'openrouter',
    model: isLocal ? 'qwen2.5-coder:1.5b-instruct' : 'anthropic/claude-sonnet-4.5',
    prompt,
    promptHash: sha256Short(prompt),
    promptTokens: 100,
    outputTokens: 50,
    costUsd: isLocal ? 0 : (input.cloudCostUsd ?? 0.002),
    latencyMs: isLocal ? 900 : 4000,
    finishReason: 'stop',
    output,
    outputHash: sha256Short(output),
    verification: {
      passed: verified,
      score: verified ? 1 : 0.2,
      blockers: verified ? [] : ['syntax error'],
      failedChecks: verified ? [] : ['syntax:prices.py'],
      durationMs: 12,
    },
  };

  const ts = input.ts ?? new Date().toISOString();
  return {
    id: ulid(Date.parse(ts)),
    schemaVersion: EPISODE_SCHEMA_VERSION,
    ts,
    harnessVersion: '0.1.0-test',
    platform: 'test',
    task,
    taskHash: sha256Short(task),
    systemPrompt: 'You are the coding tier of a two-model harness. Reply with ONE JSON object.',
    systemPromptHash: sha256Short('system'),
    features,
    vector: toVector(features),
    vectorVersion: FEATURE_VECTOR_VERSION,
    decision: {
      tier,
      reason: 'test',
      reasons: ['test'],
      pLocalSuccess: 0.8,
      difficulty: 0.2,
      taskClass: features.taskClass,
      exploration: input.exploration ?? false,
      forced: false,
      unverified: false,
      scorer: 'heuristic',
      expectedLocalCostUsd: 0,
      expectedCloudCostUsd: 0.002,
      expectedLocalLatencyMs: 900,
      expectedCloudLatencyMs: 4000,
      vetoes: [],
    },
    environment: {
      localAvailable: true,
      localModel: 'qwen2.5-coder:1.5b-instruct',
      localContextWindow: 8192,
      localModelLoaded: true,
      cloudAvailable: true,
      cloudModel: 'anthropic/claude-sonnet-4.5',
      cloudProvider: 'openrouter',
      cloudPrice: { in: 3, out: 15 },
      configuredQualityFloor: 0.72,
      verifierAvailable: true,
      routingMode: 'hybrid',
      explorationEpsilon: 0.06,
    },
    attempts: [attempt],
    outcome: {
      status,
      finalTier: tier,
      escalated: input.escalated ?? false,
      localSucceeded: input.localSucceeded === undefined ? (isLocal ? verified : null) : input.localSucceeded,
      totalCostUsd: isLocal ? 0 : (input.cloudCostUsd ?? 0.002),
      totalLatencyMs: isLocal ? 900 : 4000,
      reward: input.reward ?? (input.localSucceeded === false ? -0.6 : 1.3),
      rewardVersion: REWARD_VERSION,
      behaviorPropensity: input.behaviorPropensity ?? 0.94,
    },
    consent: { localTraining: false, globalShare: false },
    redaction: { applied: true, counts: {}, charsRemoved: 0 },
    tags: ['v1'],
  };
}
