/**
 * Training subsystem tests.
 *
 * The most important assertions here are about *refusing* to run: the scheduler
 * must block on battery, on an already-busy machine, outside the window, when
 * there is not enough data, and when the daily budget is spent. A bug that lets
 * training run when it should not is exactly the failure the project promises
 * not to have.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { evaluateGates, planSession, priorityPlan } from '../src/train/scheduler.ts';
import type { SystemState } from '../src/train/scheduler.ts';
import {
  buildLoraCommand,
  prepareMlxDataDir,
  prepareMlxDataDirFromSamples,
  preflight,
  pythonCandidate,
  ollamaModelName,
  serveCommands,
} from '../src/train/mlx.ts';
import { JobQueue, addMinutes, loadTrainState, minutesUsedToday, saveTrainState } from '../src/train/jobs.ts';
import { getActiveAdapter, listAdapters, setActiveAdapter, writeAdapterMetrics, writeOllamaModelfile } from '../src/train/adapter.ts';
import { measuredSecondsPerStep, parseLosses, refreshRouter } from '../src/train/index.ts';
import { EpisodeStore } from '../src/memory/store.ts';
import { writeTextAtomic, readTextOrNull } from '../src/util/fsx.ts';
import { makeEpisode, tempDir, testConfig } from './helpers.ts';

function state(overrides: Partial<SystemState> = {}): SystemState {
  return {
    platform: 'darwin',
    onAC: true,
    batteryPct: 80,
    thermalWarning: false,
    load: [0.4, 0.5, 0.6],
    cpuCount: 10,
    normalizedLoad: 0.04,
    now: new Date('2025-06-01T03:00:00'),
    ...overrides,
  };
}

function gateInput(cfg: ReturnType<typeof testConfig>, overrides: Partial<Parameters<typeof evaluateGates>[0]> = {}) {
  return {
    cfg,
    state: state(),
    newEpisodes: 50,
    minutesUsedToday: 0,
    hasTrainingData: true,
    ...overrides,
  };
}

describe('training gates', () => {
  it('blocks everything when training has not been opted into', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, enabled: false } });
    const result = evaluateGates(gateInput(cfg));
    assert.equal(result.allowed, false);
    assert.ok(result.blockers.some((b) => /disabled/.test(b)));
  });

  it('allows a run when every condition is satisfied', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, enabled: true } });
    const result = evaluateGates(gateInput(cfg));
    assert.equal(result.allowed, true, result.blockers.join('; '));
    assert.ok(result.checks.every((c) => c.detail.length > 0), 'every gate must explain itself');
  });

  it('blocks on battery power when requireAC is set', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, enabled: true, requireAC: true } });
    const result = evaluateGates(gateInput(cfg, { state: state({ onAC: false, batteryPct: 90 }) }));
    assert.equal(result.allowed, false);
    assert.ok(result.blockers.some((b) => /battery power/.test(b)));
  });

  it('blocks when the power source cannot be determined on a non-macOS host', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, enabled: true, requireAC: true } });
    const result = evaluateGates(gateInput(cfg, { state: state({ onAC: null, platform: 'linux' }) }));
    assert.equal(result.allowed, false);
    assert.ok(result.blockers.some((b) => /power source unknown/.test(b)));
  });

  it('blocks when the machine is busy', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, enabled: true, maxLoadAverage: 4 } });
    const result = evaluateGates(gateInput(cfg, { state: state({ load: [7.5, 6, 4] }) }));
    assert.equal(result.allowed, false);
    assert.ok(result.blockers.some((b) => /load average/.test(b)));
  });

  it('blocks when the CPU is already thermally throttled', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, enabled: true, respectThermalState: true } });
    const result = evaluateGates(gateInput(cfg, { state: state({ thermalWarning: true }) }));
    assert.equal(result.allowed, false);
    assert.ok(result.blockers.some((b) => /thermally throttled/.test(b)));
  });

  it('blocks outside the allowed window, including a window that wraps midnight', () => {
    const dir = tempDir();
    const train = { ...testConfig(dir).train, enabled: true, windowStartHour: 1, windowEndHour: 6 };
    const cfg = testConfig(dir, { train });

    assert.equal(evaluateGates(gateInput(cfg, { state: state({ now: new Date('2025-06-01T03:00:00') }) })).allowed, true);
    const daytime = evaluateGates(gateInput(cfg, { state: state({ now: new Date('2025-06-01T14:00:00') }) }));
    assert.equal(daytime.allowed, false);
    assert.ok(daytime.blockers.some((b) => /outside the allowed window/.test(b)));

    const wrapping = testConfig(dir, { train: { ...train, windowStartHour: 23, windowEndHour: 5 } });
    assert.equal(evaluateGates(gateInput(wrapping, { state: state({ now: new Date('2025-06-01T02:00:00') }) })).allowed, true);
    assert.equal(evaluateGates(gateInput(wrapping, { state: state({ now: new Date('2025-06-01T23:30:00') }) })).allowed, true);
    assert.equal(evaluateGates(gateInput(wrapping, { state: state({ now: new Date('2025-06-01T12:00:00') }) })).allowed, false);
  });

  it('requires a minimum amount of new data', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, enabled: true, minNewEpisodes: 25 } });
    const result = evaluateGates(gateInput(cfg, { newEpisodes: 3 }));
    assert.equal(result.allowed, false);
    assert.ok(result.blockers.some((b) => /new labelled episode/.test(b)));
  });

  it('blocks when there is nothing usable to train on', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, enabled: true } });
    const result = evaluateGates(gateInput(cfg, { hasTrainingData: false }));
    assert.equal(result.allowed, false);
    assert.ok(result.blockers.some((b) => /no usable training samples/.test(b)));
  });

  it('enforces the daily compute budget', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, enabled: true, dailyBudgetMin: 30 } });
    const result = evaluateGates(gateInput(cfg, { minutesUsedToday: 30 }));
    assert.equal(result.allowed, false);
    assert.ok(result.blockers.some((b) => /budget of 30 minutes is exhausted/.test(b)));
  });
});

describe('session planning', () => {
  it('never exceeds the remaining daily budget', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, maxRuntimeMin: 20, dailyBudgetMin: 30 } });
    assert.equal(planSession(cfg, 0).maxRuntimeMin, 20);
    assert.equal(planSession(cfg, 25).maxRuntimeMin, 5);
  });

  it('scales steps to the available time', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, {
      train: { ...testConfig(dir).train, maxRuntimeMin: 1, dailyBudgetMin: 1 },
    });
    // 60 seconds at 1.5s/step allows 40 steps.
    const plan = planSession(cfg, 0, 'sft', 2000, 1.5);
    assert.ok(plan.iters <= 40, `time budget should cap steps, got ${plan.iters}`);
    assert.ok(plan.iters >= 1);
  });

  it('derives steps from the dataset so the model actually sees the data', () => {
    // The old design hard-coded 60 iterations. At batch 1 that is 60 examples —
    // 15% of one epoch on a 400-row dataset, which cannot teach anything. Steps
    // now come from the epoch target over the real row count.
    const dir = tempDir();
    const cfg = testConfig(dir, {
      train: {
        ...testConfig(dir).train,
        maxRuntimeMin: 600,
        dailyBudgetMin: 600,
        lora: { ...testConfig(dir).train.lora, batchSize: 4, epochs: 3, maxIters: 2000 },
      },
    });
    const plan = planSession(cfg, 0, 'sft', 400, 1.5);
    // 3 epochs x 400 rows / batch 4 = 300 steps, and the time budget allows it.
    assert.equal(plan.iters, 300);
    assert.ok(plan.effectiveEpochs >= 2.9 && plan.effectiveEpochs <= 3.1, `got ${plan.effectiveEpochs}`);
    assert.equal(plan.sampleCount, 400);
    assert.match(plan.reason, /epoch/);
  });

  it('warns plainly when the budget cannot even cover one epoch', () => {
    // The honest failure mode for laptop LoRA: minutes of compute over thousands
    // of rows skims the data instead of learning from it.
    const dir = tempDir();
    const cfg = testConfig(dir, {
      train: { ...testConfig(dir).train, maxRuntimeMin: 20, dailyBudgetMin: 20 },
    });
    const plan = planSession(cfg, 0, 'sft', 5000, 1.5);
    assert.ok(plan.effectiveEpochs < 1);
    assert.match(plan.reason, /under one epoch/);
    assert.match(plan.reason, /Raise train.maxRuntimeMin/);
  });

  it('respects the hard step ceiling', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, {
      train: {
        ...testConfig(dir).train,
        maxRuntimeMin: 100_000,
        dailyBudgetMin: 100_000,
        lora: { ...testConfig(dir).train.lora, epochs: 50, maxIters: 500, batchSize: 1 },
      },
    });
    assert.equal(planSession(cfg, 0, 'sft', 10_000, 0.5).iters, 500);
  });

  it('runs at the lowest reasonable priority on macOS', () => {
    const plan = priorityPlan();
    if (process.platform === 'darwin') assert.match(plan.wrap, /taskpolicy -b/);
    assert.match(plan.description, /nice 19/);
  });
});

describe('mlx driver', () => {
  it('builds an SFT command with the tiny LoRA hyperparameters', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, {
      train: {
        ...testConfig(dir).train,
        baseModel: 'mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit',
        lora: { layers: 8, rank: 8, scale: 16, dropout: 0.05, learningRate: 1e-5, batchSize: 4, epochs: 3, maxIters: 2000, maxSeqLen: 1024, mode: 'sft' },
      },
    });
    const cmd = buildLoraCommand({
      cfg,
      python: '/tmp/venv/bin/python',
      baseModel: cfg.train.baseModel,
      dataDir: '/tmp/data',
      adapterPath: '/tmp/adapter',
      mode: 'sft',
      iters: 60,
    });
    assert.equal(cmd.bin, '/tmp/venv/bin/python');
    assert.ok(cmd.args.includes('mlx_lm.lora'));
    assert.ok(cmd.args.includes('--train'));
    assert.ok(cmd.args.includes('8'), 'rank/layers should be passed through');
    assert.ok(!cmd.args.includes('dpo'), 'SFT must not enable DPO mode');
    assert.match(cmd.display, /mlx_lm\.lora/);
  });

  it('switches to DPO mode when asked', () => {
    const dir = tempDir();
    const cfg = testConfig(dir);
    const cmd = buildLoraCommand({
      cfg,
      python: 'python3',
      baseModel: 'base',
      dataDir: '/tmp/data',
      adapterPath: '/tmp/adapter',
      mode: 'dpo',
      iters: 10,
    });
    assert.ok(cmd.args.includes('--train-mode'));
    assert.ok(cmd.args.includes('dpo'));
  });

  it('splits a validation set out of the training data when none is supplied', () => {
    const dir = tempDir();
    const trainPath = `${dir}/sft.jsonl`;
    const rows = Array.from({ length: 30 }, (_, i) => JSON.stringify({ messages: [{ role: 'user', content: `q${i}` }] }));
    writeTextAtomic(trainPath, rows.join('\n') + '\n');

    const prepared = prepareMlxDataDir({ targetDir: `${dir}/mlx`, trainPath, mode: 'sft' });
    assert.ok(prepared.trainRows > 0);
    assert.ok(prepared.validRows > 0, 'a validation split is required to know whether training helped');
    assert.equal(prepared.trainRows + prepared.validRows, 30);
    assert.ok(readTextOrNull(`${dir}/mlx/valid.jsonl`));
  });

  it('refuses to prepare an empty dataset', () => {
    const dir = tempDir();
    writeTextAtomic(`${dir}/empty.jsonl`, '');
    assert.throws(
      () => prepareMlxDataDir({ targetDir: `${dir}/mlx`, trainPath: `${dir}/empty.jsonl`, mode: 'sft' }),
      /no training rows/,
    );
  });

  it('never installs anything: preflight reports the manual commands instead', async () => {
    const dir = tempDir();
    const pf = await preflight(dir);
    // mlx-lm is not installed in the test environment, which is the interesting path.
    assert.equal(typeof pf.ok, 'boolean');
    assert.ok(pf.installInstructions.length >= 4);
    assert.ok(pf.installInstructions.some((c) => c.includes('pip install mlx-lm')));
    assert.ok(pf.installInstructions.every((c) => !/sudo/.test(c)));
    assert.match(pf.detail, /mlx-lm|python/);
  });

  it('prefers a project venv interpreter when one exists', () => {
    const dir = tempDir();
    assert.equal(pythonCandidate(dir), 'python3');
    const venvPython = `${dir}/venv/bin/python3`;
    writeTextAtomic(venvPython, '#!/bin/sh\n');
    assert.equal(pythonCandidate(dir), venvPython);
  });

  it('parses the loss lines that mlx-lm prints', () => {
    const output = [
      'Iter 10: Train loss 2.345, Learning Rate 1.000e-05, It/sec 3.2',
      'Iter 20: Train loss 1.234, Learning Rate 1.000e-05, It/sec 3.1',
      'Iter 20: Val loss 1.456, Val took 1.2s',
    ].join('\n');
    const parsed = parseLosses(output);
    assert.equal(parsed.trainLoss, 1.234);
    assert.equal(parsed.validLoss, 1.456);
  });

  it('produces sane serving commands and model names', () => {
    const dir = tempDir();
    const cmds = serveCommands({ dataDir: dir, baseModel: 'mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit', adapterPath: '/tmp/adapter' });
    assert.match(cmds.mlx, /mlx_lm\.server/);
    assert.match(cmds.mlx, /--adapter-path/);
    assert.match(cmds.note, /recommended/);
    assert.equal(ollamaModelName('mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit'), 'qwen2.5-coder-1.5b-instruct-4bit');
  });
});

describe('job queue and state', () => {
  it('persists jobs across instances and updates them', () => {
    const dir = tempDir();
    const q = new JobQueue(dir);
    const job = q.add({
      kind: 'sft',
      status: 'planned',
      plan: { maxRuntimeMin: 10, iters: 20, sampleCount: 80, effectiveEpochs: 1, mode: 'sft', reason: 'test' },
      note: 'test',
      datasetPath: '/tmp/sft.jsonl',
      mlxDataDir: '/tmp/data',
      adapterPath: '/tmp/adapter',
      baseModel: 'base',
    });
    q.update(job.id, { status: 'done', durationMs: 1234 });

    const reloaded = new JobQueue(dir);
    assert.equal(reloaded.all().length, 1);
    assert.equal(reloaded.all()[0]?.status, 'done');
    assert.equal(reloaded.recent(5)[0]?.durationMs, 1234);
    assert.equal(reloaded.latestOfKind('sft')?.id, job.id);
  });

  it('tracks daily minutes and rolls old days off', () => {
    const dir = tempDir();
    let state = loadTrainState(dir);
    assert.equal(minutesUsedToday(state), 0);

    state = addMinutes(state, 12.5, new Date('2025-06-01T03:00:00'));
    assert.equal(minutesUsedToday(state, new Date('2025-06-01T04:00:00')), 12.5);
    assert.equal(minutesUsedToday(state, new Date('2025-06-02T04:00:00')), 0);

    saveTrainState(dir, state);
    const reloaded = loadTrainState(dir);
    assert.equal(reloaded.minutesByDay['2025-06-01'], 12.5);
    assert.equal(reloaded.totalMinutes, 12.5);
  });
});

describe('training data preparation', () => {
  it('writes rows straight from memory, with a validation split', () => {
    // Regression guard for the bug that made local fine-tuning impossible on a
    // fresh install: the scheduler read a dataset file it had never written.
    const dir = tempDir();
    const samples = Array.from({ length: 50 }, (_, i) => ({ messages: [{ role: 'user', content: `q${i}` }] }));
    const prepared = prepareMlxDataDirFromSamples({ targetDir: `${dir}/mlx`, samples });

    assert.equal(prepared.trainRows + prepared.validRows, 50);
    assert.ok(prepared.validRows > 0, 'a validation split is needed to know whether training helped');
    const train = readTextOrNull(`${dir}/mlx/train.jsonl`) ?? '';
    const valid = readTextOrNull(`${dir}/mlx/valid.jsonl`) ?? '';
    assert.equal(train.trim().split('\n').length, prepared.trainRows);
    assert.equal(valid.trim().split('\n').length, prepared.validRows);
    assert.ok(!train.includes('_meta'), 'internal provenance keys must not reach the trainer');
  });

  it('trains on everything when there are too few rows to split', () => {
    const dir = tempDir();
    const prepared = prepareMlxDataDirFromSamples({
      targetDir: `${dir}/mlx`,
      samples: [{ messages: [] }, { messages: [] }],
    });
    assert.equal(prepared.trainRows, 2);
    assert.equal(prepared.validRows, 2);
  });

  it('refuses an empty sample set instead of writing a file the trainer cannot use', () => {
    const dir = tempDir();
    assert.throws(
      () => prepareMlxDataDirFromSamples({ targetDir: `${dir}/mlx`, samples: [] }),
      /no training rows/,
    );
  });
});

describe('measured step cost', () => {
  it('falls back to the configured estimate before any history exists', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, secondsPerStep: 2.5 } });
    assert.equal(measuredSecondsPerStep(cfg, new JobQueue(dir)), 2.5);
  });

  it('uses the median of what actually happened once jobs have run', () => {
    // A guessed constant decides how many steps fit in the night, so guessing
    // forever would make every session plan fictional.
    const dir = tempDir();
    const q = new JobQueue(dir);
    const plan = { maxRuntimeMin: 10, iters: 100, sampleCount: 400, effectiveEpochs: 1, mode: 'sft' as const, reason: '' };
    for (const durationMs of [100_000, 200_000, 300_000]) {
      const job = q.add({
        kind: 'sft',
        status: 'done',
        plan,
        note: 'test',
        datasetPath: '/tmp/d',
        mlxDataDir: '/tmp/m',
        adapterPath: '/tmp/a',
        baseModel: 'base',
        durationMs,
      });
      q.update(job.id, { durationMs });
    }
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, secondsPerStep: 99 } });
    // 100s, 200s, 300s over 100 steps => 1, 2, 3 s/step; median 2.
    assert.equal(measuredSecondsPerStep(cfg, new JobQueue(dir)), 2);
  });

  it('ignores failed and implausibly short jobs', () => {
    const dir = tempDir();
    const q = new JobQueue(dir);
    const plan = { maxRuntimeMin: 10, iters: 100, sampleCount: 10, effectiveEpochs: 1, mode: 'sft' as const, reason: '' };
    const failed = q.add({
      kind: 'sft',
      status: 'done',
      plan,
      note: '',
      datasetPath: '',
      mlxDataDir: '',
      adapterPath: '',
      baseModel: '',
      durationMs: 10,
    });
    q.update(failed.id, { status: 'failed', durationMs: 10 });
    const cfg = testConfig(dir, { train: { ...testConfig(dir).train, secondsPerStep: 4 } });
    assert.equal(measuredSecondsPerStep(cfg, new JobQueue(dir)), 4);
  });
});

describe('adapters', () => {
  it('lists, activates and clears adapters', () => {
    const dir = tempDir();
    assert.deepEqual(listAdapters(dir), []);

    const adapterDir = `${dir}/models/adapters/sft-abc`;
    writeTextAtomic(`${adapterDir}/adapter_config.json`, '{}');
    writeTextAtomic(`${adapterDir}/adapters.safetensors`, 'x');
    writeAdapterMetrics(adapterDir, { trainLoss: 1.1, iters: 60, mode: 'sft', datasetRows: 40, jobId: 'job-1' });

    const adapters = listAdapters(dir);
    assert.equal(adapters.length, 1);
    assert.equal(adapters[0]?.complete, true);
    assert.equal(adapters[0]?.mode, 'sft');
    assert.equal(adapters[0]?.trainLoss, 1.1);

    const active = setActiveAdapter(dir, adapters[0]!, 'base-model');
    assert.equal(getActiveAdapter(dir)?.name, active.name);
    assert.match(active.runtimeHint, /mlx_lm\.server/);
  });

  it('generates an Ollama Modelfile with the conversion steps spelled out', () => {
    const dir = tempDir();
    const result = writeOllamaModelfile({
      dataDir: dir,
      baseModel: 'mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit',
      ggufAdapterPath: `${dir}/models/lora.gguf`,
    });
    const content = readTextOrNull(result.modelfilePath) ?? '';
    assert.match(content, /^FROM /m);
    assert.match(content, /^ADAPTER /m);
    assert.match(content, /PARAMETER temperature/);
    assert.ok(result.conversionCommands.length >= 3);
    assert.ok(result.conversionCommands.some((c) => c.includes('mlx_lm.fuse')));
  });
});

describe('router refresh', () => {
  it('declines to train with too few episodes and explains why', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    store.append(makeEpisode({ localSucceeded: true }));
    const result = refreshRouter(testConfig(dir), dir, { write: true });
    assert.equal(result.trained, false);
    assert.match(result.reason ?? '', /need 40/);
  });

  it('trains from enough labelled episodes and writes weights once learnable', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    // Easy tasks succeed locally, hard tasks fail: a signal the scorer can learn.
    for (let i = 0; i < 30; i++) {
      store.append(
        makeEpisode({
          task: `Rename the variable a${i} to count${i} in this file.`,
          localSucceeded: true,
          reward: 1.2,
        }),
      );
      store.append(
        makeEpisode({
          task: `Design the module boundaries and migrate the schema for service ${i}.`,
          localSucceeded: false,
          status: 'escalated-cloud-success',
          escalated: true,
          verified: false,
          reward: 0.3,
        }),
      );
    }
    const result = refreshRouter(testConfig(dir), dir, { write: true });
    assert.equal(result.trained, true, result.reason ?? '');
    assert.ok(result.sampleCount >= 40);
    assert.ok(result.auc > 0.5, `expected better-than-chance separation, auc=${result.auc}`);
    assert.ok(readTextOrNull(result.weightsPath), 'weights must be written to disk');
  });
});
