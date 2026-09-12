/**
 * Harness (agent loop) public surface.
 */

export { runTask, applyToWorkspace } from './loop.ts';
export type { RunOptions, RunResult, RunEvent } from './loop.ts';
export {
  buildUserPrompt,
  buildMessages,
  buildRepairPrompt,
  systemPromptFor,
  repairSystemPrompt,
  PROMPT_VERSION,
} from './prompt.ts';
