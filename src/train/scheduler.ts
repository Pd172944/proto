/**
 * The training scheduler: decide whether it is acceptable to touch the user's CPU.
 *
 * This is the module that makes "RL that costs basically 0 compute" honest. The
 * claim is not that training is free — it is that training happens only in
 * conditions where the compute is genuinely idle and wall-powered, at the lowest
 * OS priority, in bounded slices, and with a watchdog that aborts the moment
 * the machine gets busy again.
 *
 * The gate list is deliberately explicit and every gate reports a reason
 * whether it passes or fails, so `proto train status` can always answer "why
 * isn't this running?" — the single most common failure of background ML
 * systems is that they silently do nothing and the user assumes they are broken
 * (or, worse, assume they are running).
 */

import { cpus, loadavg } from 'node:os';

import { localHour } from '../util/clock.ts';

import type { ProtoConfig } from '../config/schema.ts';

export interface SystemState {
  platform: string;
  /** true = on wall power, false = on battery, null = unknown (non-macOS). */
  onAC: boolean | null;
  batteryPct: number | null;
  /** True when the OS reports thermal or performance throttling. */
  thermalWarning: boolean;
  /** Raw 1/5/15-minute load averages. */
  load: [number, number, number];
  cpuCount: number;
  /** Normalised load: load1m / cpuCount. 1.0 means fully busy. */
  normalizedLoad: number;
  now: Date;
}

/**
 * Read machine state.
 *
 * macOS specifics:
 *  - `pmset -g batt` reports the power source and charge.
 *  - `pmset -g therm` reports thermal pressure; a non-zero
 *    `CPU_Scheduler_Limit` means the OS is already throttling us, in which case
 *    training would be both slow and unpleasant for the user.
 *
 * On other platforms everything degrades to `unknown` rather than guessing, and
 * the gates that depend on unknown values fail *closed* (they block training),
 * because the whole point is to be conservative with someone else's machine.
 */
export async function readSystemState(now: Date = new Date()): Promise<SystemState> {
  const { exec } = await import('../util/proc.ts');
  const cpuCount = cpus().length;
  const load = loadavg() as [number, number, number];

  let onAC: boolean | null = null;
  let batteryPct: number | null = null;
  let thermalWarning = false;

  if (process.platform === 'darwin') {
    const batt = await exec('pmset', ['-g', 'batt'], { timeoutMs: 5000 });
    if (batt.code === 0) {
      const text = `${batt.stdout}${batt.stderr}`;
      if (/AC Power/i.test(text)) onAC = true;
      else if (/Battery Power/i.test(text)) onAC = false;
      const pct = text.match(/(\d{1,3})%/);
      if (pct?.[1]) batteryPct = Number(pct[1]);
    }
    const therm = await exec('pmset', ['-g', 'therm'], { timeoutMs: 5000 });
    if (therm.code === 0) {
      const text = `${therm.stdout}${therm.stderr}`;
      // Any reported limit below 100 means the OS is throttling.
      const limit = text.match(/CPU_Scheduler_Limit\s*=\s*(\d+)/);
      if (limit?.[1] && Number(limit[1]) < 100) thermalWarning = true;
      if (/CPU_Speed_Limit\s*=\s*(\d+)/.test(text)) {
        const speed = text.match(/CPU_Speed_Limit\s*=\s*(\d+)/);
        if (speed?.[1] && Number(speed[1]) < 100) thermalWarning = true;
      }
    }
  }

  return {
    platform: process.platform,
    onAC,
    batteryPct,
    thermalWarning,
    load,
    cpuCount,
    normalizedLoad: load[0] / Math.max(1, cpuCount),
    now,
  };
}

export interface GateInput {
  cfg: ProtoConfig;
  state: SystemState;
  /** New, reward-labelled episodes since the last training run. */
  newEpisodes: number;
  /** Minutes of training already spent today. */
  minutesUsedToday: number;
  /** Whether any dataset has usable samples. */
  hasTrainingData: boolean;
}

export interface GateResult {
  allowed: boolean;
  /** Every gate, with its verdict, for `proto train status`. */
  checks: Array<{ id: string; ok: boolean; detail: string }>;
  /** Failing check details, in priority order. */
  blockers: string[];
}

export function evaluateGates(input: GateInput): GateResult {
  const { cfg, state, newEpisodes, minutesUsedToday, hasTrainingData } = input;
  const t = cfg.train;
  const checks: GateResult['checks'] = [];
  const blockers: string[] = [];

  const add = (id: string, ok: boolean, detail: string, blocking = true): void => {
    checks.push({ id, ok, detail });
    if (!ok && blocking) blockers.push(detail);
  };

  add(
    'opt-in',
    t.enabled,
    t.enabled
      ? 'training is enabled (user opted in)'
      : 'training is disabled; enable with `proto train enable` (opt-in by design)',
  );

  const hour = localHour(state.now);
  const inWindow = t.windowStartHour <= t.windowEndHour
    ? hour >= t.windowStartHour && hour < t.windowEndHour
    : hour >= t.windowStartHour || hour < t.windowEndHour;
  add(
    'window',
    inWindow,
    inWindow
      ? `inside the allowed window (${t.windowStartHour}:00-${t.windowEndHour}:00, now ${hour}:00)`
      : `outside the allowed window (${t.windowStartHour}:00-${t.windowEndHour}:00, now ${hour}:00)`,
  );

  if (t.requireAC) {
    const ok = state.onAC === true;
    add(
      'power',
      ok,
      state.onAC === null
        ? 'power source unknown on this platform and requireAC is set; refusing to train'
        : ok
          ? `on wall power${state.batteryPct !== null ? ` (${state.batteryPct}%)` : ''}`
          : 'on battery power and train.requireAC is set',
    );
  } else {
    add('power', true, 'requireAC is off; power source is not a gate', false);
  }

  if (!t.requireAC && state.onAC === false && state.batteryPct !== null) {
    add(
      'battery-level',
      state.batteryPct >= t.minBatteryPct,
      state.batteryPct >= t.minBatteryPct
        ? `battery ${state.batteryPct}% is above the ${t.minBatteryPct}% floor`
        : `battery ${state.batteryPct}% is below the ${t.minBatteryPct}% floor`,
    );
  }

  if (t.respectThermalState) {
    add(
      'thermal',
      !state.thermalWarning,
      state.thermalWarning
        ? 'the CPU is already thermally throttled; not adding load'
        : `no thermal throttling reported (platform ${state.platform})`,
    );
  } else {
    add('thermal', true, 'respectThermalState is off', false);
  }

  add(
    'idle',
    state.load[0] <= t.maxLoadAverage,
    state.load[0] <= t.maxLoadAverage
      ? `1-minute load average ${state.load[0].toFixed(2)} is at or below the ${t.maxLoadAverage} limit`
      : `1-minute load average ${state.load[0].toFixed(2)} exceeds the ${t.maxLoadAverage} limit; the machine is busy`,
  );

  add(
    'data',
    hasTrainingData,
    hasTrainingData
      ? 'there are usable training samples'
      : 'no usable training samples yet; run some tasks first (SFT needs a verified attempt)',
  );

  add(
    'min-episodes',
    newEpisodes >= t.minNewEpisodes,
    newEpisodes >= t.minNewEpisodes
      ? `${newEpisodes} new labelled episode(s) since the last run (min ${t.minNewEpisodes})`
      : `only ${newEpisodes} new labelled episode(s) since the last run (min ${t.minNewEpisodes})`,
  );

  add(
    'daily-budget',
    minutesUsedToday < t.dailyBudgetMin,
    minutesUsedToday < t.dailyBudgetMin
      ? `${minutesUsedToday} of ${t.dailyBudgetMin} daily training minutes used`
      : `daily budget of ${t.dailyBudgetMin} minutes is exhausted (${minutesUsedToday} used)`,
  );

  return { allowed: blockers.length === 0, checks, blockers };
}

export interface SessionPlan {
  /** Wall-clock cap for this session. */
  maxRuntimeMin: number;
  /** Iterations to run, scaled down if the session is short. */
  iters: number;
  mode: 'sft' | 'dpo';
  reason: string;
}

/**
 * Plan a session.
 *
 * Iterations are scaled by how much time is actually available, because `iters`
 * tuned for a 20-minute window is wrong for a 5-minute one. Under-shooting costs
 * nothing (the next night continues); over-shooting means a killed process and a
 * wasted night.
 */
export function planSession(cfg: ProtoConfig, minutesUsedToday: number, mode?: 'sft' | 'dpo'): SessionPlan {
  const remaining = Math.max(1, cfg.train.dailyBudgetMin - minutesUsedToday);
  const maxRuntimeMin = Math.min(cfg.train.maxRuntimeMin, remaining);
  const configured = cfg.train.lora.iters;
  // Assume roughly 6 seconds per iteration at this LoRA size on Apple Silicon
  // (measured on an M-series laptop with a 1.5B 4-bit model). Deliberately
  // pessimistic: better to stop early than to be killed mid-run.
  const affordable = Math.floor((maxRuntimeMin * 60) / 6);
  const iters = Math.max(5, Math.min(configured, affordable));
  const chosenMode = mode ?? cfg.train.lora.mode;
  return {
    maxRuntimeMin,
    iters,
    mode: chosenMode,
    reason:
      iters < configured
        ? `reduced iterations from ${configured} to ${iters} to fit ${maxRuntimeMin} available minute(s)`
        : `${iters} iteration(s) fit comfortably in ${maxRuntimeMin} minute(s)`,
  };
}

/** Nice level and QoS wrapper description, used in the plan output. */
export function priorityPlan(): { wrap: string; description: string } {
  if (process.platform === 'darwin') {
    return {
      wrap: 'taskpolicy -b nice -n 19',
      description:
        'run under macOS background QoS (taskpolicy -b) at nice 19, so the process yields to everything interactive',
    };
  }
  return { wrap: 'nice -n 19', description: 'run at nice 19' };
}
