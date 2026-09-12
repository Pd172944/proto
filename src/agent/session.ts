/**
 * Session state for the interactive agent.
 *
 * Kept deliberately small and serialisable. A session is: the conversation, the
 * working directory, the approval memory for this session ("always allow writes" —
 * never persisted to disk, because a remembered approval from three weeks ago is
 * not consent), running cost, and a transcript.
 *
 * Sessions are written to `<dataDir>/sessions/<id>.json` so `/resume` can work and,
 * more importantly, so that the transcript is available later as training data for
 * the trajectories that actually did something useful.
 */

import { join } from 'node:path';

import type { Message } from '../providers/types.ts';
import type { ApprovalDecision } from '../tools/types.ts';
import { ensureDir, readJsonOrNull, writeJsonAtomic, listFiles } from '../util/fsx.ts';
import { shortId } from '../util/ids.ts';

export interface SessionStats {
  turns: number;
  steps: number;
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  costUsd: number;
  filesEdited: string[];
  commandsRun: number;
}

export interface SessionRecord {
  version: 1;
  id: string;
  createdAt: string;
  updatedAt: string;
  workspace: string;
  model: string;
  provider: string;
  /** Conversation, excluding the system prompt (rebuilt per run from context). */
  messages: Message[];
  stats: SessionStats;
}

const EMPTY_STATS: SessionStats = {
  turns: 0,
  steps: 0,
  inputTokens: 0,
  outputTokens: 0,
  cachedInputTokens: 0,
  costUsd: 0,
  filesEdited: [],
  commandsRun: 0,
};

export class Session {
  readonly id: string;
  readonly workspace: string;
  readonly createdAt: string;
  model: string;
  provider: string;
  messages: Message[] = [];
  stats: SessionStats = { ...EMPTY_STATS, filesEdited: [] };

  /**
   * In-memory only. Deliberately never persisted: "always allow" means "for this
   * session, while I am watching". A remembered approval would let a later,
   * unrelated run write files the user never agreed to.
   */
  private readonly alwaysAllowed = new Set<string>();

  constructor(opts: { workspace: string; model: string; provider: string; id?: string }) {
    this.id = opts.id ?? shortId('s');
    this.workspace = opts.workspace;
    this.model = opts.model;
    this.provider = opts.provider;
    this.createdAt = new Date().toISOString();
  }

  /** Map a tool name to its approval key, so "always" can be granted per class. */
  static approvalKey(toolName: string, path?: string): string {
    return path ? `${toolName}:${path}` : toolName;
  }

  rememberApproval(key: string): void {
    this.alwaysAllowed.add(key);
  }

  isRemembered(key: string): boolean {
    return this.alwaysAllowed.has(key);
  }

  clearRemembered(): void {
    this.alwaysAllowed.clear();
  }

  /** Decide without prompting, when the user has already said "always". */
  preApproved(key: string): ApprovalDecision | null {
    return this.alwaysAllowed.has(key) ? 'allow-always' : null;
  }

  recordTurn(result: {
    steps: number;
    usage: { inputTokens: number; outputTokens: number; cachedInputTokens: number };
    costUsd: number;
    editedFiles: string[];
    commands: string[];
  }): void {
    this.stats.turns += 1;
    this.stats.steps += result.steps;
    this.stats.inputTokens += result.usage.inputTokens;
    this.stats.outputTokens += result.usage.outputTokens;
    this.stats.cachedInputTokens += result.usage.cachedInputTokens;
    this.stats.costUsd += result.costUsd;
    this.stats.commandsRun += result.commands.length;
    for (const f of result.editedFiles) {
      if (!this.stats.filesEdited.includes(f)) this.stats.filesEdited.push(f);
    }
  }

  /** Append the full turn so the next user message has the context it needs. */
  appendTurn(userText: string, agentMessages: Message[]): void {
    this.messages.push({ role: 'user', content: userText });
    // agentMessages[0] is the system prompt; the rest is the assistant/tool exchange.
    this.messages.push(...agentMessages.slice(1));
    this.trimToBudget();
  }

  /**
   * Keep the conversation inside a sane window.
   *
   * A coding session can run for hours. Dropping the *oldest* exchanges is the
   * crude but honest option: the alternative (summarising) costs another model
   * call and can silently lose the constraint the user stated at the start, so
   * instead we drop whole turns and tell the user it happened.
   */
  trimToBudget(maxChars = 400_000): { dropped: number } {
    let total = this.messages.reduce((a, m) => a + m.content.length, 0);
    let dropped = 0;
    while (total > maxChars && this.messages.length > 4) {
      const removed = this.messages.splice(0, 2);
      dropped += removed.length;
      total -= removed.reduce((a, m) => a + m.content.length, 0);
    }
    return { dropped };
  }

  clear(): void {
    this.messages = [];
    this.stats = { ...EMPTY_STATS, filesEdited: [] };
    this.alwaysAllowed.clear();
  }

  toRecord(): SessionRecord {
    return {
      version: 1,
      id: this.id,
      createdAt: this.createdAt,
      updatedAt: new Date().toISOString(),
      workspace: this.workspace,
      model: this.model,
      provider: this.provider,
      messages: this.messages,
      stats: this.stats,
    };
  }

  static fromRecord(record: SessionRecord): Session {
    const session = new Session({
      workspace: record.workspace,
      model: record.model,
      provider: record.provider,
      id: record.id,
    });
    session.messages = record.messages;
    session.stats = { ...EMPTY_STATS, ...record.stats, filesEdited: record.stats?.filesEdited ?? [] };
    return session;
  }
}

/* ------------------------------------------------------------------ */
/* Persistence                                                        */
/* ------------------------------------------------------------------ */

export function sessionsDir(dataDir: string): string {
  return join(dataDir, 'sessions');
}

export function saveSession(dataDir: string, session: Session): string {
  const dir = ensureDir(sessionsDir(dataDir));
  const path = join(dir, `${session.id}.json`);
  writeJsonAtomic(path, session.toRecord());
  return path;
}

export function loadSession(dataDir: string, id: string): Session | null {
  const record = readJsonOrNull<SessionRecord>(join(sessionsDir(dataDir), `${id}.json`));
  return record ? Session.fromRecord(record) : null;
}

export function latestSession(dataDir: string): Session | null {
  const files = listFiles(sessionsDir(dataDir), { suffix: '.json', max: 200 });
  if (files.length === 0) return null;
  const newest = files.sort().pop() as string;
  const record = readJsonOrNull<SessionRecord>(newest);
  return record ? Session.fromRecord(record) : null;
}

export function listSessions(dataDir: string, limit = 10): SessionRecord[] {
  const files = listFiles(sessionsDir(dataDir), { suffix: '.json', max: 500 });
  const records: SessionRecord[] = [];
  for (const f of files) {
    const r = readJsonOrNull<SessionRecord>(f);
    if (r) records.push(r);
  }
  return records.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)).slice(0, limit);
}
