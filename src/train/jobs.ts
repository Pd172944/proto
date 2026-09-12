/**
 * Durable job queue for training sessions.
 *
 * Why a queue at all for something that runs synchronously? Because the thing
 * that actually invokes a session is a `launchd`/`cron` tick that may fire while
 * the machine is busy, and the user needs to be able to see — days later — what
 * was attempted, what was skipped, why, and what came out. A queue file turns
 * "the RL thing sometimes does something" into an auditable log.
 *
 * It also makes the pipeline resumable: an interrupted session is marked
 * `interrupted` rather than lost, and the adapter directory it wrote can be kept
 * or discarded.
 */

import { join } from 'node:path';

import type { SessionPlan } from './scheduler.ts';
import { readJsonOrNull, writeJsonAtomic, ensureDir } from '../util/fsx.ts';
import { localDayKey, shiftDayKey } from '../util/clock.ts';
import { shortId } from '../util/ids.ts';

export type JobStatus = 'planned' | 'running' | 'done' | 'failed' | 'skipped' | 'interrupted';

export interface TrainJob {
  id: string;
  createdAt: string;
  kind: 'sft' | 'dpo';
  status: JobStatus;
  plan: SessionPlan;
  /** Why the job was created or skipped. */
  note: string;
  datasetPath: string;
  mlxDataDir: string;
  adapterPath: string;
  baseModel: string;
  command?: string;
  logPath?: string;
  startedAt?: string;
  finishedAt?: string;
  exitCode?: number | null;
  durationMs?: number;
  /** Final training loss parsed from the log, when available. */
  trainLoss?: number;
  validLoss?: number;
  /** Evaluation of the adapter against the previous one, when measured. */
  evaluation?: { beforeReward: number; afterReward: number; samples: number };
  error?: string;
}

export class JobQueue {
  readonly path: string;
  private readonly maxJobs: number;
  private jobs: TrainJob[];

  constructor(dataDir: string, maxJobs = 100) {
    this.maxJobs = maxJobs;
    this.path = join(dataDir, 'train', 'jobs.json');
    ensureDir(join(dataDir, 'train'));
    const existing = readJsonOrNull<{ jobs?: TrainJob[] }>(this.path);
    this.jobs = existing?.jobs ?? [];
  }

  all(): TrainJob[] {
    return [...this.jobs];
  }

  recent(n = 10): TrainJob[] {
    return this.jobs.slice(-n).reverse();
  }

  pending(): TrainJob[] {
    return this.jobs.filter((j) => j.status === 'planned');
  }

  add(job: Omit<TrainJob, 'id' | 'createdAt'> & { id?: string; createdAt?: string }): TrainJob {
    const full: TrainJob = {
      ...job,
      id: job.id ?? shortId('job'),
      createdAt: job.createdAt ?? new Date().toISOString(),
    };
    this.jobs.push(full);
    this.trim();
    this.save();
    return full;
  }

  update(id: string, patch: Partial<TrainJob>): TrainJob | null {
    const idx = this.jobs.findIndex((j) => j.id === id);
    if (idx < 0) return null;
    const merged = { ...(this.jobs[idx] as TrainJob), ...patch };
    this.jobs[idx] = merged;
    this.save();
    return merged;
  }

  latestOfKind(kind: 'sft' | 'dpo'): TrainJob | null {
    for (let i = this.jobs.length - 1; i >= 0; i--) {
      const j = this.jobs[i] as TrainJob;
      if (j.kind === kind) return j;
    }
    return null;
  }

  private trim(): void {
    if (this.jobs.length <= this.maxJobs) return;
    // Keep the most recent jobs; older ones are of no operational interest.
    this.jobs = this.jobs.slice(-this.maxJobs);
  }

  private save(): void {
    writeJsonAtomic(this.path, { jobs: this.jobs });
  }
}

/** Marker marking the state a session left behind, used by `proto train status`. */
export interface TrainState {
  lastTickAt?: string;
  lastRunAt?: string;
  /** Id of the newest episode included in the last run (ULIDs sort by time). */
  highWaterEpisodeId?: string;
  /** Minutes of training per local day. */
  minutesByDay: Record<string, number>;
  consecutiveSkips: number;
  totalRuns: number;
  totalMinutes: number;
  activeAdapter?: string;
  /** Set when training is enabled but has been failing; surfaced loudly. */
  lastError?: string;
}

export const EMPTY_TRAIN_STATE: TrainState = {
  minutesByDay: {},
  consecutiveSkips: 0,
  totalRuns: 0,
  totalMinutes: 0,
};

export function loadTrainState(dataDir: string): TrainState {
  const raw = readJsonOrNull<Partial<TrainState>>(join(dataDir, 'train', 'state.json'));
  if (!raw) return { ...EMPTY_TRAIN_STATE, minutesByDay: {} };
  return {
    ...EMPTY_TRAIN_STATE,
    ...raw,
    minutesByDay: raw.minutesByDay ?? {},
  };
}

export function saveTrainState(dataDir: string, state: TrainState): void {
  ensureDir(join(dataDir, 'train'));
  writeJsonAtomic(join(dataDir, 'train', 'state.json'), state);
}

export function minutesUsedToday(state: TrainState, now: Date = new Date()): number {
  return state.minutesByDay[localDayKey(now)] ?? 0;
}

export function addMinutes(state: TrainState, minutes: number, now: Date = new Date()): TrainState {
  const day = localDayKey(now);
  const next: TrainState = {
    ...state,
    minutesByDay: { ...state.minutesByDay, [day]: (state.minutesByDay[day] ?? 0) + minutes },
    totalMinutes: state.totalMinutes + minutes,
  };
  // Keep only the last 30 days of bookkeeping.
  const cutoff = shiftDayKey(localDayKey(now), -30);
  for (const key of Object.keys(next.minutesByDay)) {
    if (key < cutoff) delete next.minutesByDay[key];
  }
  return next;
}
