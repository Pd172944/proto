/**
 * The reward function.
 *
 * ## Why a hand-written shaped reward, and not a learned reward model
 *
 * A learned reward model needs (a) a lot of preference data we do not have on
 * day one, and (b) compute to train, which is exactly what this project is
 * trying to avoid. More importantly, a learned reward model on a laptop with a
 * few hundred samples is a *worse* reward than an explicit one, because its
 * failure modes are invisible. So the reward here is a documented linear
 * combination of observable facts. It is versioned (`REWARD_VERSION`) and
 * stamped into every episode, so changing it never silently invalidates old
 * data — downstream tooling can filter on the version.
 *
 * ## The formula
 *
 *   r = 0.80 * [final answer verified]
 *     + 0.30 * verification_score
 *     + 0.20 * [local attempt verified AND no escalation]     (cheap success bonus)
 *     - 0.60 * [escalated]                                     (local was wrong)
 *     - 0.30 * [refusal]
 *     - 0.20 * [envelope drift]
 *     - 0.25 * [explicit user rejection]
 *     + 0.25 * [explicit user acceptance]
 *     - 0.10 * [verifier found blockers]
 *     + 0.15 * [exploration succeeded]                         (information bonus)
 *     - 0.05 * [exploration failed]                            (cost of learning)
 *     - 1.0  * [cloud cost in USD]                             (spend discipline)
 *
 * clamped to [-1.5, 1.75].
 *
 * ## Why the weights are scaled the way they are
 *
 * The dominant term is the binary "did it verify". Everything else is a small
 * adjustment, because the RL signal we can trust is binary and the rest is
 * opinion. In particular the cheap-success bonus is smaller than the escalation
 * penalty: we want the policy to *learn when local fails* more than we want it
 * to *prefer local*. Getting that backwards produces a router that confidently
 * sends hard work to a model that cannot do it.
 *
 * The upper clamp is deliberately above the maximum achievable positive sum
 * (~1.7). An earlier revision clamped at 1.5, which meant a clean local success
 * (1.8 pre-clamp) absorbed the exploration bonus and the user-acceptance bonus
 * entirely — both became no-ops in exactly the common case they were meant to
 * reward. Saturating a reward hides the distinctions you designed it to make.
 *
 * The exploration terms are the interesting ones. A successful exploration
 * episode is worth *more* than a normal success, because it is the only source
 * of counterfactual evidence about tasks the policy would have avoided — that
 * episode is what lets the scorer discover it was being too conservative. A
 * failed exploration costs only a little, so exploration never dominates.
 */

import type { OutcomeStatus } from './types.ts';
import type { EpisodeDecision, Episode } from './types.ts';
import { REWARD_VERSION } from './types.ts';

export { REWARD_VERSION };

export interface RewardInput {
  status: OutcomeStatus;
  escalated: boolean;
  /** True when a local-tier attempt was made. */
  localAttempted: boolean;
  /** Whether the local attempt passed verification (null if never observed). */
  localPassed: boolean | null;
  /** Score of the final accepted attempt (0..1). */
  finalScore: number;
  refusal: boolean;
  envelopeDrift: boolean;
  exploration: boolean;
  feedback?: 'accept' | 'reject' | 'edit';
  /** True when the final verification had blockers. */
  hadBlockers: boolean;
  /** Total cloud spend for the episode, USD. */
  cloudCostUsd: number;
}

export interface RewardComponent {
  name: string;
  value: number;
  note: string;
}

export interface RewardResult {
  reward: number;
  version: number;
  components: RewardComponent[];
}

const WEIGHTS = {
  verified: 0.8,
  score: 0.3,
  cheapSuccess: 0.2,
  escalated: -0.6,
  refusal: -0.3,
  envelopeDrift: -0.2,
  userReject: -0.25,
  userAccept: 0.25,
  blockers: -0.1,
  explorationSuccess: 0.15,
  explorationFailure: -0.05,
  /** Cost penalty: 1.0 per USD, i.e. a $0.05 escalation costs 0.05 reward. */
  costScale: -1.0,
} as const;

/**
 * Upper clamp above the maximum positive sum (~1.7) so that no bonus is ever
 * swallowed by saturation. See the design note in the module docstring.
 */
const REWARD_MAX = 1.75;
const REWARD_MIN = -1.5;

export function computeReward(input: RewardInput): RewardResult {
  const components: RewardComponent[] = [];
  const add = (name: string, value: number, note: string): void => {
    if (value !== 0) components.push({ name, value, note });
  };

  const successStatuses: OutcomeStatus[] = ['local-success', 'cloud-success', 'escalated-cloud-success'];
  const verified = successStatuses.includes(input.status);

  add('verified', verified ? WEIGHTS.verified : 0, verified ? 'final answer passed verification' : 'no verified answer');
  if (verified) {
    add('verificationScore', WEIGHTS.score * input.finalScore, `final verification score ${input.finalScore.toFixed(2)}`);
  }

  const cheapSuccess = input.localAttempted && input.localPassed === true && !input.escalated;
  add(
    'cheapSuccessBonus',
    cheapSuccess ? WEIGHTS.cheapSuccess : 0,
    cheapSuccess ? 'solved locally without escalation' : '',
  );

  add('escalationPenalty', input.escalated ? WEIGHTS.escalated : 0, input.escalated ? 'local attempt failed; work escalated' : '');
  add('refusalPenalty', input.refusal ? WEIGHTS.refusal : 0, input.refusal ? 'model refused the task' : '');
  add('envelopeDriftPenalty', input.envelopeDrift ? WEIGHTS.envelopeDrift : 0, input.envelopeDrift ? 'model ignored the output envelope' : '');
  add('blockersPenalty', input.hadBlockers ? WEIGHTS.blockers : 0, input.hadBlockers ? 'verifier reported blockers' : '');

  if (input.feedback === 'reject') add('userReject', WEIGHTS.userReject, 'user explicitly rejected the result');
  if (input.feedback === 'accept') add('userAccept', WEIGHTS.userAccept, 'user explicitly accepted the result');

  const expSuccess = input.exploration && (input.localPassed === true);
  const expFailure = input.exploration && (input.localPassed === false);
  add('explorationBonus', expSuccess ? WEIGHTS.explorationSuccess : 0, expSuccess ? 'exploration episode succeeded: valuable counterfactual' : '');
  add('explorationCost', expFailure ? WEIGHTS.explorationFailure : 0, expFailure ? 'exploration episode failed: small learning cost' : '');

  add('cost', WEIGHTS.costScale * input.cloudCostUsd, `cloud spend $${input.cloudCostUsd.toFixed(4)}`);

  const reward = components.reduce((a, c) => a + c.value, 0);
  return {
    reward: Math.round(Math.min(REWARD_MAX, Math.max(REWARD_MIN, reward)) * 1e6) / 1e6,
    version: REWARD_VERSION,
    components,
  };
}

/**
 * The router's supervised label, derived from an episode.
 *
 * Returns null when local was attempted but never verified (nothing to learn),
 * or when local was not attempted at all outside of exploration. Undefined
 * labels are dropped rather than imputed: imputing "local would have failed"
 * for tasks we routed away is exactly the bias that makes routers overconfident.
 *
 * `w` is the **inverse-propensity weight**, not the propensity itself. See
 * `ipsWeight()` for why the direction of this number matters so much.
 */
export function routerLabel(episode: Episode): { y: 0 | 1; x: number[]; w: number } | null {
  const localAttempts = episode.attempts.filter((a) => a.tier === 'local' || a.tier === 'local-tiny');
  if (localAttempts.length === 0) return null;
  const verified = localAttempts.find((a) => a.verification !== null);
  if (!verified || !verified.verification) return null;
  return {
    y: verified.verification.passed ? 1 : 0,
    x: episode.vector,
    w: ipsWeight(episode.outcome.behaviorPropensity),
  };
}

/**
 * Propensity: the probability that the behaviour policy would choose the action
 * it actually chose, `π(a|x)`.
 *
 * This is the quantity stored on every episode, and it is *not* the training
 * weight — `ipsWeight()` inverts it. Getting that direction wrong is a subtle
 * and damaging bug, which is why the two are separate, named functions.
 *
 * The behaviour policy modelled here is: take the greedy decision, then with
 * probability `epsilon` send a cloud-bound task to local instead (exploration
 * only ever pushes toward local, and only when a verifier can catch failure).
 * Therefore:
 *
 *   forced decision      -> π = 1     (the policy had no choice)
 *   greedy local         -> π = 1     (nothing was going to flip it away)
 *   greedy cloud         -> π = 1 - ε
 *   exploration to local -> π = ε
 *
 * Two honest limitations:
 *  - `epsilon` is used even when exploration was *not* permitted (no verifier,
 *    or above the difficulty/token bounds). That slightly understates π for
 *    cloud decisions, i.e. over-weights them by ~1/(1-ε) ≈ 1.06. Negligible,
 *    but it is an approximation, not an exact propensity.
 *  - The greedy decision itself is deterministic given the features, so the
 *    randomness in the behaviour policy comes only from the exploration coin.
 *    A rigorous treatment would log the actual stochastic policy; this is the
 *    cheap, documented approximation.
 */
export function behaviorPropensity(decision: EpisodeDecision, epsilon: number): number {
  if (decision.forced) return 1;
  if (decision.exploration) return Math.max(1e-6, Math.min(1, epsilon));
  const choseLocal = decision.tier === 'local' || decision.tier === 'local-tiny';
  if (choseLocal) return 1;
  return Math.max(1e-6, Math.min(1, 1 - epsilon));
}

/**
 * Largest IPS weight we will apply. Without a cap, a small `epsilon` (say
 * 0.001) would give a single exploration episode a weight of 1000 and let one
 * noisy label dominate the fit. The cap trades a little bias for a lot of
 * variance reduction, which is the right trade at this data scale.
 */
export const MAX_IPS_WEIGHT = 25;

/**
 * Turn a propensity into an inverse-propensity weight.
 *
 * Why this matters: exploration episodes are the *only* episodes that carry
 * counterfactual information — they are the cases where the policy would have
 * sent the task to the cloud but sent it local anyway, and we therefore learn
 * what local would have done. Those labels are rare (probability `epsilon`),
 * so they must be **up-weighted** by ~1/ε to represent the population they stand
 * in for. Using the propensity itself as the weight does the exact opposite,
 * down-weighting the only informative samples by ~16x. That was a real bug in
 * an earlier revision; there is a test asserting the direction.
 */
export function ipsWeight(propensity: number): number {
  const pi = Math.max(1e-6, Math.min(1, propensity));
  return Math.min(MAX_IPS_WEIGHT, 1 / pi);
}
