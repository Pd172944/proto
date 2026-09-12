/**
 * Adapter management: where adapters live, which one is active, and how to serve
 * them.
 *
 * The awkward truth about local fine-tuning on a Mac is the final mile. There
 * are two viable paths and they have different trade-offs, so this module
 * documents both and generates the artifacts for each rather than pretending the
 * problem does not exist:
 *
 *  A. **MLX serving (recommended).** `mlx_lm.server --model <base> --adapter-path
 *     <adapter>` loads base + LoRA directly. No conversion. Downside: it is a
 *     second local server alongside Ollama, and MLX's server is less mature.
 *
 *  B. **Ollama.** Better ergonomics and model management, but Ollama consumes
 *     GGUF, so an MLX LoRA adapter must first be fused and converted. We generate
 *     the Modelfile and the exact conversion commands; we never run them.
 *
 * Choosing not to hide this is deliberate: a harness that claims "we fine-tuned
 * your model" but leaves it invisible to the runtime the user actually uses is
 * worse than one that explains the two options.
 */

import { join } from 'node:path';
import { readdirSync, statSync } from 'node:fs';

import { serveCommands, ollamaModelName } from './mlx.ts';
import type { ProtoConfig } from '../config/schema.ts';
import { ensureDir, readJsonOrNull, writeJsonAtomic, writeTextAtomic, listFiles } from '../util/fsx.ts';

export interface AdapterInfo {
  name: string;
  path: string;
  mode: 'sft' | 'dpo' | 'unknown';
  createdAt: string | null;
  sizeBytes: number;
  /** True when mlx-lm wrote a complete adapter (config + weights). */
  complete: boolean;
  trainLoss: number | null;
}

export function adaptersRoot(dataDir: string): string {
  return join(dataDir, 'models', 'adapters');
}

export function listAdapters(dataDir: string): AdapterInfo[] {
  const root = adaptersRoot(dataDir);
  let entries: string[];
  try {
    entries = readdirSync(root);
  } catch {
    return [];
  }
  const out: AdapterInfo[] = [];
  for (const name of entries) {
    const path = join(root, name);
    let st;
    try {
      st = statSync(path);
    } catch {
      continue;
    }
    if (!st.isDirectory()) continue;
    const files = listFiles(path, { max: 200 });
    const hasConfig = files.some((f) => f.endsWith('adapter_config.json'));
    const hasWeights = files.some((f) => f.endsWith('adapters.safetensors'));
    const sizeBytes = files.reduce((a, f) => {
      try {
        return a + statSync(f).size;
      } catch {
        return a;
      }
    }, 0);
    out.push({
      name,
      path,
      mode: name.startsWith('dpo') ? 'dpo' : name.startsWith('sft') ? 'sft' : 'unknown',
      createdAt: st.birthtime ? st.birthtime.toISOString() : null,
      sizeBytes,
      complete: hasConfig && hasWeights,
      trainLoss: readLoss(path),
    });
  }
  return out.sort((a, b) => (b.createdAt ?? '').localeCompare(a.createdAt ?? ''));
}

function readLoss(adapterPath: string): number | null {
  const meta = readJsonOrNull<{ trainLoss?: number }>(join(adapterPath, 'proto-metrics.json'));
  return typeof meta?.trainLoss === 'number' ? meta.trainLoss : null;
}

/** Record training metrics next to the adapter so provenance survives restarts. */
export function writeAdapterMetrics(
  adapterPath: string,
  metrics: { trainLoss?: number; validLoss?: number; iters: number; mode: string; datasetRows: number; jobId: string },
): void {
  ensureDir(adapterPath);
  writeJsonAtomic(join(adapterPath, 'proto-metrics.json'), {
    ...metrics,
    writtenAt: new Date().toISOString(),
  });
}

/* ------------------------------------------------------------------ */
/* Active adapter                                                      */
/* ------------------------------------------------------------------ */

export interface ActiveAdapter {
  name: string;
  path: string;
  promotedAt: string;
  /** Config changes the user must make for the runtime to pick it up. */
  runtimeHint: string;
}

export function activeAdapterPath(dataDir: string): string {
  return join(dataDir, 'models', 'active.json');
}

export function getActiveAdapter(dataDir: string): ActiveAdapter | null {
  return readJsonOrNull<ActiveAdapter>(activeAdapterPath(dataDir));
}

export function setActiveAdapter(dataDir: string, info: AdapterInfo, baseModel: string): ActiveAdapter {
  const record: ActiveAdapter = {
    name: info.name,
    path: info.path,
    promotedAt: new Date().toISOString(),
    runtimeHint:
      `set local.runtime="mlx" and serve with: ` +
      serveCommands({ dataDir, baseModel, adapterPath: info.path }).mlx,
  };
  ensureDir(join(dataDir, 'models'));
  writeJsonAtomic(activeAdapterPath(dataDir), record);
  return record;
}

export function clearActiveAdapter(dataDir: string): void {
  writeJsonAtomic(activeAdapterPath(dataDir), null);
}

/* ------------------------------------------------------------------ */
/* Serving artifacts                                                   */
/* ------------------------------------------------------------------ */

/**
 * Write an Ollama Modelfile for a fused model.
 *
 * Note the `ADAPTER` line points at a GGUF LoRA file, not at the MLX directory:
 * Ollama cannot read MLX safetensors. The conversion command is emitted as a
 * comment and as a returned string so the user can run it themselves.
 */
export function writeOllamaModelfile(input: {
  dataDir: string;
  baseModel: string;
  /** GGUF LoRA path, once the user has converted the MLX adapter. */
  ggufAdapterPath: string;
  outPath?: string;
}): { modelfilePath: string; modelName: string; conversionCommands: string[] } {
  const modelName = `${ollamaModelName(input.baseModel)}-proto`;
  const outPath = input.outPath ?? join(input.dataDir, 'models', 'Modelfile');
  const baseTag = `${ollamaModelName(input.baseModel)}:latest`;
  const content = `# Generated by proto. Do not edit by hand.
#
# Ollama needs a GGUF LoRA adapter. Convert the MLX adapter first:
#   (see conversionCommands printed by \`proto train adapters\`)
FROM ${baseTag}
ADAPTER ${input.ggufAdapterPath}
PARAMETER temperature 0.1
PARAMETER num_ctx 8192
`;
  writeTextAtomic(outPath, content);
  return {
    modelfilePath: outPath,
    modelName,
    conversionCommands: [
      '# 1. fuse the MLX LoRA adapter into the base weights (mlx-lm, local, no download):',
      `${input.dataDir}/venv/bin/python -m mlx_lm.fuse --model ${input.baseModel} --adapter-path ${input.ggufAdapterPath.replace(/\/[^/]+$/, '')} --save-path ${input.dataDir}/models/fused`,
      '# 2. convert the fused model to GGUF (requires llama.cpp\'s converter):',
      `python3 llama.cpp/convert_hf_to_gguf.py ${input.dataDir}/models/fused --outfile ${input.dataDir}/models/fused.gguf --outtype q4_K_M`,
      '# 3. create the Ollama model:',
      `ollama create ${modelName} -f ${outPath}`,
    ],
  };
}
