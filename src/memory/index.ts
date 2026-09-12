/**
 * Memory layer public surface.
 *
 * The RL data pipeline is: `Episodes -> datasets -> (router weights | LoRA job)`.
 * Everything in this folder is about the first arrow; nothing here trains.
 */

export * from './types.ts';
export { EpisodeStore, newEpisodeId, writeMarker } from './store.ts';
export type { FeedbackRecord, StoreStats } from './store.ts';
export { redact, redactForHash, totalRedactions, REDACTION_RULES } from './redact.ts';
export type { RedactResult, RedactOptions, RedactionRule } from './redact.ts';
export { computeReward, routerLabel, behaviorPropensity, ipsWeight, MAX_IPS_WEIGHT, REWARD_VERSION } from './reward.ts';
export type { RewardInput, RewardResult, RewardComponent } from './reward.ts';
export { buildDatasets, describeDatasets, DATASET_DIR } from './datasets.ts';
export type {
  BuiltDatasets,
  BuildOptions,
  DpoSample,
  RouterSample,
  SftSample,
  DatasetBuild,
  BuildStats,
} from './datasets.ts';
