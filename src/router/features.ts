/**
 * Feature extraction.
 *
 * This is the router's sensor layer and it is deliberately **model-free**: every
 * signal below is computed with regexes and cheap structural analysis over the
 * task text and any code in scope. Two consequences that matter:
 *
 *  - Routing costs ~0 ms and $0. Asking a model "is this easy?" would defeat the
 *    entire point of having a cheap tier.
 *  - The same features are what the learned scorer consumes. Because they are
 *    deterministic, a logged episode can be replayed offline to evaluate a new
 *    routing policy without re-running any model (see `proto replay`).
 *
 * The heuristics are intentionally transparent and unit-tested. Where a signal
 * is uncertain we prefer to under-claim: an over-confident "this is easy" sends
 * work to a weak model and costs the user more time than it saves.
 */

import {
  FEATURE_NAMES,
  FEATURE_VECTOR_VERSION,
} from './types.ts';
import type { TaskClass, TaskContext, TaskFeatures } from './types.ts';
import { countMatches, estimateTokens } from '../util/text.ts';

/* ------------------------------------------------------------------ */
/* Task classification                                                 */
/* ------------------------------------------------------------------ */

interface ClassRule {
  cls: TaskClass;
  /** Prior difficulty in [0,1] for the class, used as both a feature and a prior. */
  difficulty: number;
  /**
   * Selection priority. Higher wins when several classes match.
   *
   * The ordering encodes one deliberate principle: **when a task matches several
   * classes, classify it as the hardest one.** Undertaking is the expensive
   * mistake for a router — labelling a performance investigation as a "bugfix"
   * sends it to a model that will produce a plausible non-fix — whereas
   * overestimating difficulty costs one cloud call. So `perf`, `security` and
   * `migration` outrank the generic edit classes even when those match more
   * keywords.
   */
  priority: number;
  /** Any match contributes. More matches => stronger evidence. */
  patterns: RegExp[];
  /** Relative weight, used to break ties between equal priorities. */
  weight: number;
}

/**
 * Rationale for the difficulty priors: they encode "how often does a
 * 1.5B-class model get this right first try, given that a verifier will catch
 * hard failures" — not task prestige. Rationale for the priorities: specificity
 * and blast radius, in that order.
 */
const CLASS_RULES: ClassRule[] = [
  {
    cls: 'format',
    difficulty: 0.05,
    priority: 10,
    weight: 3,
    patterns: [
      /\b(format|reformat|prettier|black(?!list)|gofmt|rustfmt|lint|eslint|whitespace|indent(?:ation)?|trailing comma)/i,
      /\b(line length|style guide|sort imports?|organize imports?)/i,
    ],
  },
  {
    cls: 'rename',
    difficulty: 0.1,
    priority: 12,
    weight: 3,
    patterns: [/\brename\b/i, /\bchange the name of\b/i, /\bcall(?:ed|s)? it\b.*\binstead\b/i],
  },
  {
    cls: 'prompt-edit',
    difficulty: 0.12,
    priority: 34,
    weight: 2.5,
    patterns: [
      /\b(prompt|system message|instruction(?:s)?|persona|template)/i,
      /\breword\b/i,
      /\bmake (?:the )?(?:prompt|wording|tone)/i,
    ],
  },
  {
    cls: 'docs',
    difficulty: 0.12,
    priority: 18,
    weight: 2,
    patterns: [
      /\b(docstring|doc comment|jsdoc|readme|changelog|comment(?:s)? (?:for|on|the))/i,
      /\badd (?:a )?(?:doc|comment|note)/i,
    ],
  },
  {
    cls: 'explain',
    difficulty: 0.15,
    priority: 22,
    weight: 2,
    patterns: [
      /\b(explain|what does|why does|walk me through|describe what|how does .* work|understand)/i,
      /\b(document|summar(?:y|ise|ize)|clarify|describe) (?:what|how|why|the|this)/i,
    ],
  },
  {
    cls: 'config',
    difficulty: 0.3,
    priority: 33,
    weight: 1.6,
    patterns: [
      /\b(package\.json|pyproject\.toml|tsconfig|dockerfile|makefile|\.github\/workflows|ci config|env(?:ironment)? var)/i,
      /\bbump (?:the )?(?:version|dependency)/i,
    ],
  },
  {
    cls: 'add-validation',
    difficulty: 0.22,
    priority: 30,
    weight: 2.4,
    patterns: [
      /\b(validat|guard|bounds? check|out of bounds|out-of-bounds|index error|none check|null check|empty check|sanity check|clamp|assert|reject)/i,
      /\b(handle (?:the )?(?:case|input|missing|empty|null))/i,
    ],
  },
  {
    cls: 'bugfix-local',
    difficulty: 0.3,
    priority: 36,
    weight: 2.2,
    patterns: [
      // Deliberately narrow: only unambiguous defect wording. Earlier revisions
      // also matched "instead of" and "raises", which stole tasks like "add a
      // guard so an empty list returns 0 instead of raising" from
      // add-validation. Weak evidence must not outrank specific evidence.
      /\b(fix|fixes|bug|broken|off[- ]by[- ]one|typo|incorrect|regression|crash(?:es)?|fails?|failing|wrong result|wrong value)/i,
    ],
  },
  {
    cls: 'local-edit',
    difficulty: 0.25,
    priority: 26,
    weight: 1.6,
    // Only *action* verbs. An earlier revision also matched scope hints such as
    // "in this file", which made local-edit outrank the more specific `rename`
    // and `prompt-edit` classes. Scope belongs in the `locality` feature, not in
    // the class decision.
    patterns: [/\b(change|update|adjust|rewrite|modify|tweak|swap|replace|move|remove|delete|drop|simplify)/i],
  },
  {
    cls: 'write-tests',
    difficulty: 0.35,
    priority: 35,
    weight: 1.8,
    patterns: [/\b(write|add|create) (?:a |some |more )?tests?\b/i, /\btest coverage\b/i, /\bunit tests?\b/i],
  },
  {
    cls: 'feature-new',
    difficulty: 0.55,
    priority: 42,
    weight: 1.4,
    // Requires *new capability* phrasing. A bare "add" is not evidence of a new
    // feature — "add a guard", "add a docstring" and "add a test" are all
    // narrower, easier classes, and must win.
    patterns: [
      /\b(implement|introduce|support for|new (?:feature|endpoint|command|option|flag|service|module|provider|api|route))/i,
      /\b(add|build|create) (?:a |an |the )?(?:new |support for )/i,
      /\bfeature\b/i,
    ],
  },
  {
    cls: 'refactor-multi',
    difficulty: 0.6,
    priority: 55,
    weight: 2.2,
    patterns: [
      /\b(refactor|restructure|reorganiz|extract (?:a |the )?(?:module|class|interface|service)|split .* into|decouple|abstraction|design pattern|shared module|deduplicat)/i,
      /\bacross (?:the )?(?:codebase|repo|modules|files|services)/i,
    ],
  },
  {
    cls: 'debug-unknown',
    difficulty: 0.72,
    priority: 60,
    weight: 1.8,
    patterns: [
      /\b(debug|figure out why|investigate|intermittent|sometimes|flaky|works locally|not sure why|root cause|silently)/i,
    ],
  },
  {
    cls: 'algorithm',
    difficulty: 0.7,
    priority: 70,
    weight: 2,
    patterns: [
      /\b(algorithm|dynamic programming|graph traversal|dijkstra|topological|complexity|O\(n\^?\d?\)|heuristic|optimi[sz]e the algorithm|makespan|scheduler)/i,
      /\bimplement .*(?:sort|search|tree|graph|parser|tokenizer)/i,
    ],
  },
  {
    cls: 'perf',
    difficulty: 0.7,
    priority: 75,
    weight: 2.2,
    patterns: [
      /\b(performance|latency|throughput|speed (?:it )?up|slow(?:er)?|profil|benchmark|memory (?:usage|leak)|cache miss|hot path|regression in)/i,
    ],
  },
  {
    cls: 'concurrency',
    difficulty: 0.75,
    priority: 80,
    weight: 2.4,
    patterns: [
      /\b(race condition|deadlock|thread|mutex|semaphore|lock contention|goroutine|asyncio\.gather|parallel(?:ism|ise|ize)?|concurren|atomic|fan-?out)/i,
      /\bawait\b.*\binside (?:a )?(?:loop|callback)/i,
    ],
  },
  {
    cls: 'migration',
    difficulty: 0.82,
    priority: 86,
    weight: 2.4,
    patterns: [
      /\b(migrat|upgrade|port(?:ing)? (?:this|the|to)|rewrite .* in (?:rust|go|typescript|python)|breaking change|deprecat|schema change|framework upgrade|without downtime)/i,
    ],
  },
  {
    cls: 'architecture',
    difficulty: 0.85,
    priority: 88,
    weight: 2.4,
    patterns: [
      /\b(architect|design (?:a|the) (?:system|api|schema|interface|module)|module boundar|trade-?offs?|scalab|should we use|strategy for|roadmap|evolve independently)/i,
    ],
  },
  {
    cls: 'security',
    difficulty: 0.8,
    priority: 90,
    weight: 2.6,
    patterns: [
      /\b(security|vulnerab|exploit|injection|xss|csrf|sanitiz|authenticat|authoriz|encrypt|hash(?:ing)? password|secret|credential|CVE-|harden|privilege escalat|escalation of privilege|bypass|least privilege|permission|sandbox|audit log)/i,
    ],
  },
];

const EDIT_VERBS =
  /\b(fix|change|add|remove|refactor|rename|rewrite|implement|update|create|write|move|delete|replace|patch|bump|extract|inline|revert|adjust|modify)\b/i;

export function classifyTask(text: string, ctx: { fileCount: number; hasStackTrace: boolean }): {
  cls: TaskClass;
  difficulty: number;
  signals: string[];
} {
  const signals: string[] = [];
  const matched: Array<{ cls: TaskClass; difficulty: number; priority: number; hits: number; weight: number }> = [];

  for (const rule of CLASS_RULES) {
    let hits = 0;
    for (const p of rule.patterns) if (p.test(text)) hits++;
    if (hits === 0) continue;
    matched.push({ cls: rule.cls, difficulty: rule.difficulty, priority: rule.priority, hits, weight: rule.weight });
  }

  // Hardest match wins; ties break on evidence count, then on the class weight.
  matched.sort((a, b) => b.priority - a.priority || b.hits - a.hits || b.weight - a.weight);

  let best: { cls: TaskClass; difficulty: number } = { cls: 'unknown', difficulty: 0.5 };
  if (matched.length > 0) {
    const winner = matched[0] as (typeof matched)[number];
    best = { cls: winner.cls, difficulty: winner.difficulty };
    signals.push(
      `classified as ${winner.cls} (class prior difficulty ${winner.difficulty.toFixed(2)}); ` +
        `matched ${matched.length} class(es), chose the hardest`,
    );
    if (matched.length > 1) {
      signals.push(
        `other matches (ranked lower on purpose): ${matched
          .slice(1, 4)
          .map((m) => m.cls)
          .join(', ')}`,
      );
    }
  } else if (ctx.hasStackTrace) {
    // A stack trace is strong evidence of a concrete, debuggable defect rather
    // than an open-ended design question.
    best = { cls: 'bugfix-local', difficulty: 0.3 };
    signals.push('stack trace present with no other class signal -> treated as localized bugfix');
  } else {
    signals.push('no class patterns matched; difficulty treated as unknown (0.50)');
  }

  // Cross-file scope upgrades several classes to their harder cousins.
  if (
    ctx.fileCount >= 4 &&
    ['local-edit', 'bugfix-local', 'add-validation', 'docs', 'rename'].includes(best.cls)
  ) {
    signals.push(`${ctx.fileCount} files in scope: upgrading ${best.cls} -> refactor-multi`);
    return { cls: 'refactor-multi', difficulty: 0.6, signals };
  }

  return { cls: best.cls, difficulty: best.difficulty, signals };
}

/* ------------------------------------------------------------------ */
/* Code statistics                                                     */
/* ------------------------------------------------------------------ */

export interface CodeStats {
  loopCount: number;
  funcCount: number;
  maxNesting: number;
  branchCount: number;
  hasAsync: boolean;
  hasConcurrency: boolean;
  hasTypes: boolean;
  hasErrorHandling: boolean;
  hasTestsInScope: boolean;
  lineCount: number;
  languages: string[];
}

/** Strip comments and string literals so keyword counts reflect real code. */
export function stripNonCode(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/(^|[^:])\/\/[^\n]*/g, '$1 ')
    .replace(/(^|\s)#[^\n]*/g, '$1 ')
    .replace(/"""[\s\S]*?"""/g, ' ')
    .replace(/'''[\s\S]*?'''/g, ' ')
    .replace(/"(?:[^"\\\n]|\\.)*"/g, '""')
    .replace(/'(?:[^'\\\n]|\\.)*'/g, "''");
}

const EXT_LANG: Record<string, string> = {
  ts: 'typescript', tsx: 'typescript', js: 'javascript', jsx: 'javascript', mjs: 'javascript',
  cjs: 'javascript', py: 'python', rb: 'ruby', go: 'go', rs: 'rust', java: 'java', kt: 'kotlin',
  c: 'c', h: 'c', cc: 'cpp', cpp: 'cpp', hpp: 'cpp', cs: 'csharp', php: 'php', swift: 'swift',
  sh: 'shell', bash: 'shell', zsh: 'shell', sql: 'sql', yml: 'yaml', yaml: 'yaml', json: 'json',
  toml: 'toml', md: 'markdown', html: 'html', css: 'css', scss: 'scss', vue: 'vue', svelte: 'svelte',
};

export function languageOfPath(path: string): string | null {
  const m = path.match(/\.([A-Za-z0-9]+)$/);
  if (!m) return null;
  return EXT_LANG[(m[1] as string).toLowerCase()] ?? null;
}

/**
 * Nesting depth. Brace languages are measured by brace depth; indentation
 * languages (Python, YAML) by leading whitespace. We take whichever signal the
 * file actually exhibits more of, which avoids reporting depth 0 for Python.
 */
function measureNesting(code: string): number {
  let braceDepth = 0;
  let maxBrace = 0;
  for (const ch of code) {
    if (ch === '{') {
      braceDepth++;
      if (braceDepth > maxBrace) maxBrace = braceDepth;
    } else if (ch === '}') braceDepth = Math.max(0, braceDepth - 1);
  }

  let maxIndent = 0;
  for (const line of code.split('\n')) {
    if (!line.trim()) continue;
    const m = line.match(/^[ \t]*/);
    const width = (m?.[0] ?? '').replace(/\t/g, '    ').length;
    // Ignore continuation-style alignment (very deep single lines).
    if (width <= 40) maxIndent = Math.max(maxIndent, Math.floor(width / 4));
  }

  return Math.max(maxBrace, maxIndent);
}

export function analyzeCode(files: Array<{ path: string; content: string }>): CodeStats {
  const code = stripNonCode(files.map((f) => f.content).join('\n'));
  const raw = files.map((f) => f.content).join('\n');
  const languages = [...new Set(files.map((f) => languageOfPath(f.path)).filter((l): l is string => !!l))];

  const looksLikeTestFile = files.some((f) => /(^|\/)(test|tests|spec|__tests__)\b|\.(test|spec)\.|_test\.|test_/i.test(f.path));
  const hasTestCode = /\b(describe|it|expect|assert|def test_|unittest|pytest)\b/.test(code);

  return {
    loopCount: countMatches(code, /\b(for|while|do)\b\s*[({]?/g),
    funcCount: countMatches(code, /\b(function|def|fn|func|sub|method)\b\s*[A-Za-z_]*\s*\(|=>\s*\{/g),
    maxNesting: measureNesting(code),
    branchCount: countMatches(code, /\b(if|else if|elif|switch|case|catch|except|when)\b/g),
    hasAsync: /\b(async|await|Promise|asyncio|goroutine|tokio|Future|coroutine)\b/.test(code),
    hasConcurrency: /\b(thread|mutex|lock|semaphore|atomic|channel|worker|pool|parallel|race)\b/i.test(code),
    hasTypes: /:\s*(string|number|boolean|void|any|unknown|[A-Z][A-Za-z0-9_]*)\b|->\s*\w|\binterface\b|\btype\s+\w+\s*=|\bProtocol\b/.test(code),
    hasErrorHandling: /\b(try|catch|except|finally|Result|raise|throw|recover)\b/.test(code),
    hasTestsInScope: looksLikeTestFile || hasTestCode,
    lineCount: raw.split('\n').length,
    languages,
  };
}

/* ------------------------------------------------------------------ */
/* Phrasing signals                                                    */
/* ------------------------------------------------------------------ */

const VAGUE_REFERENTS = /\b(it|this|that|these|those|some|something|stuff|things?|the file|the code|the function|somewhere)\b/gi;
const HEDGES = /\b(maybe|perhaps|probably|i think|not sure|might|somehow|figure out|wherever|whatever|etc)\b/gi;
const SPECIFICITY = /`[^`]+`|"[^"]+"|'[^']+'|\b[A-Za-z_][A-Za-z0-9_]*\(\)|\bline \d+\b|:\d+:\d+|\.\w{1,5}\b/g;

const CONSTRAINT_PATTERNS = [
  /\bmust\b/gi,
  /\bmake sure\b/gi,
  /\bensure\b/gi,
  /\bdon'?t\b/gi,
  /\bdo not\b/gi,
  /\bwithout\b/gi,
  /\bonly\b/gi,
  /\bexactly\b/gi,
  /\bkeep\b/gi,
  /\bshould not\b/gi,
  /\bavoid\b/gi,
  /\bpreserve\b/gi,
  /\bunchanged\b/gi,
  /\band also\b/gi,
  /\bas well as\b/gi,
];

export function countConstraints(text: string, extra: string[] = []): number {
  let n = 0;
  for (const p of CONSTRAINT_PATTERNS) n += countMatches(text, p);
  n += extra.length;
  return n;
}

/** 0 = fully specified, 1 = maximally ambiguous. */
export function measureAmbiguity(text: string): number {
  const vague = countMatches(text, VAGUE_REFERENTS);
  const hedges = countMatches(text, HEDGES);
  const specific = countMatches(text, SPECIFICITY);
  const len = Math.max(1, text.split(/\s+/).length);
  const vagueRate = Math.min(1, (vague / len) * 12);
  const hedgeRate = Math.min(1, (hedges / len) * 20);
  const specificityRelief = Math.min(1, specific / 4);
  const raw = 0.25 + 0.45 * vagueRate + 0.3 * hedgeRate - 0.35 * specificityRelief;
  return clamp01(raw);
}

/** 0 = sprawling, 1 = tightly scoped to one small place. */
export function measureLocality(
  text: string,
  stats: { fileCount: number; lineCount: number; specificSymbols: number; classIsLocal: boolean },
): number {
  let score = 0.5;
  if (stats.fileCount === 0) score = 0.45; // no code supplied: we cannot be sure
  else if (stats.fileCount === 1) score = 0.85;
  else if (stats.fileCount === 2) score = 0.6;
  else if (stats.fileCount >= 5) score = 0.15;
  else score = 0.4;

  if (stats.lineCount > 0 && stats.lineCount <= 80) score += 0.1;
  else if (stats.lineCount > 600) score -= 0.2;

  score += Math.min(0.2, stats.specificSymbols * 0.07);
  if (stats.classIsLocal) score += 0.1;
  if (/\b(across|everywhere|all files|throughout|codebase)\b/i.test(text)) score -= 0.35;
  return clamp01(score);
}

const EXTERNAL_API = /\b(react|vue|svelte|next\.?js|django|flask|fastapi|rails|spring|numpy|pandas|torch|tensorflow|boto3|aws|gcp|azure|kubernetes|docker|graphql|grpc|kafka|redis|postgres|mysql|mongodb)\b/i;
const VERSION_PIN = /\bv?\d+\.\d+(\.\d+)?\b/;

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

/* ------------------------------------------------------------------ */
/* Main entry point                                                    */
/* ------------------------------------------------------------------ */

export function extractFeatures(ctx: TaskContext): TaskFeatures {
  const task = ctx.task ?? '';
  const files = ctx.files ?? [];
  const diff = ctx.diff ?? '';
  const signals: string[] = [];

  const stats = analyzeCode(files);
  const scoped = `${task}\n${diff}`;
  const scopedCode = stripNonCode(scoped);

  const hasStackTrace = /Traceback \(most recent call last\)|\n\s+at\s+[\w.$]+\(.*:\d+:\d+\)|panic:|Exception in thread|\bFile ".*", line \d+/.test(
    scoped,
  );
  const hasReproSteps = /\b(repro(?:duce|duction)?|steps to reproduce|when i (?:run|do|call|click|open)|fails when|to reproduce|minimal example|here'?s? how)\b/i.test(
    scoped,
  );

  const { cls, difficulty: classDifficulty, signals: classSignals } = classifyTask(task || diff, {
    fileCount: files.length,
    hasStackTrace,
  });
  signals.push(...classSignals);

  const specificSymbols = countMatches(scoped, SPECIFICITY);
  const ambiguity = measureAmbiguity(scoped);
  const locality = measureLocality(scoped, {
    fileCount: files.length,
    lineCount: stats.lineCount,
    specificSymbols,
    classIsLocal: ['format', 'rename', 'local-edit', 'bugfix-local', 'add-validation', 'docs', 'prompt-edit'].includes(cls),
  });

  const constraintCount = countConstraints(scoped, ctx.constraints ?? []);
  const isQuestion = /\?\s*$/.test(task.trim()) || /^(how|why|what|when|where|can you|could you|is it|does it)\b/i.test(task.trim());
  // Read-only intent: no edit verb anywhere in the task, plus explanation
  // vocabulary. The no-edit-verb guard matters: "review this and fix the bug"
  // contains "review" but is emphatically a mutating task.
  const isExplainOnly =
    !EDIT_VERBS.test(task) &&
    /\b(explain|describe|what does|why does|walk me through|summar(?:y|ise|ize)|review|critique|clarify|document)\b/i.test(task);

  const changedLines = estimateChangedLines(diff, task, classDifficulty);
  const estOutputTokens = estimateOutputTokens({ changedLines, isExplainOnly, cls, classDifficulty });
  const estInputTokens = estimateTokens(
    [task, ...files.map((f) => `// ${f.path}\n${f.content}`), diff].join('\n'),
  );

  const languages = [...new Set([...stats.languages, ...detectFencedLanguages(task)])];

  const features: TaskFeatures = {
    taskChars: task.length,
    estInputTokens,
    estOutputTokens,
    fileCount: files.length,
    changedLines,
    loopCount: stats.loopCount,
    funcCount: stats.funcCount,
    maxNesting: stats.maxNesting,
    branchCount: stats.branchCount,
    hasAsync: stats.hasAsync || /\b(async|await|concurren|thread)\b/i.test(scopedCode),
    hasConcurrency: stats.hasConcurrency,
    hasTypes: stats.hasTypes,
    hasErrorHandling: stats.hasErrorHandling,
    hasTestsInScope: stats.hasTestsInScope,
    hasStackTrace,
    hasReproSteps,
    hasExplicitAcceptanceCriteria: /\b(should|expect(?:ed)?|acceptance|so that|then it|verify that|test that)\b/i.test(scoped),
    hasExternalApiMention: EXTERNAL_API.test(scoped) || VERSION_PIN.test(scoped),
    hasPerfLanguage: /\b(performance|latency|throughput|faster|slower|profil|benchmark|memory usage|O\(n)\b/i.test(scoped),
    // Note: a bare "token" is NOT security wording. "Add a null check before
    // using the token" is a routine validation edit; treating it as security
    // would hard-lock it to the cloud because of one noun. Only qualified
    // credential phrases count.
    hasSecurityLanguage:
      /\b(security|auth|encrypt|injection|xss|csrf|permission|sanitiz|credential|secret|harden|vulnerab|privilege escalat|(?:access|auth|api|bearer|session|refresh|csrf)[ _-]?token)/i.test(
        scoped,
      ),
    hasMigrationLanguage: /\b(migrat|upgrade|port to|rewrite in|breaking change|deprecat|version bump)\b/i.test(scoped),
    isQuestion,
    isExplainOnly,
    constraintCount,
    ambiguity,
    locality,
    mentionsSpecificSymbol: specificSymbols > 0,
    taskClass: cls,
    classDifficulty,
    languages,
    signals: [...signals, ...describeSignals({ ambiguity, locality, constraintCount, stats, files: files.length })],
  };

  return features;
}

function describeSignals(input: {
  ambiguity: number;
  locality: number;
  constraintCount: number;
  stats: CodeStats;
  files: number;
}): string[] {
  const out: string[] = [];
  if (input.files === 0) out.push('no code supplied: difficulty estimated from wording alone');
  if (input.ambiguity > 0.6) out.push(`high ambiguity (${input.ambiguity.toFixed(2)}): wording leaves real choices open`);
  if (input.locality > 0.8) out.push('change looks tightly scoped to one place');
  if (input.locality < 0.35) out.push('change looks cross-cutting');
  if (input.constraintCount >= 3) out.push(`${input.constraintCount} explicit constraints increase the chance of a missed requirement`);
  if (input.stats.maxNesting >= 4) out.push(`deep nesting (depth ${input.stats.maxNesting}) makes a small model likelier to lose track`);
  if (input.stats.hasConcurrency) out.push('concurrency in scope raises correctness risk');
  return out;
}

function estimateChangedLines(diff: string, task: string, classDifficulty: number): number {
  if (diff) {
    const added = countMatches(diff, /^\+(?!\+\+)/gm);
    const removed = countMatches(diff, /^-(?!--)/gm);
    return added + removed;
  }
  const explicit = task.match(/\b(\d+)\s*(?:lines?|loc)\b/i);
  if (explicit?.[1]) return Math.min(2000, Number(explicit[1]));
  if (/\b(one[- ]line|single line|typo|rename|reformat)\b/i.test(task)) return 2;
  // Fall back to a class-dependent guess: bounded edits are small, new features are not.
  if (classDifficulty <= 0.15) return 4;
  if (classDifficulty <= 0.35) return 12;
  if (classDifficulty <= 0.6) return 45;
  return 120;
}

function estimateOutputTokens(input: { changedLines: number; isExplainOnly: boolean; cls: TaskClass; classDifficulty: number }): number {
  if (input.isExplainOnly || input.cls === 'explain' || input.cls === 'architecture') {
    return clamp(input.changedLines * 2 + 320, 120, 2200);
  }
  if (input.cls === 'format' || input.cls === 'rename') return clamp(input.changedLines * 8, 40, 800);
  // Code edits carry the changed lines plus surrounding context the model re-emits.
  return clamp(input.changedLines * 12 + 120 + input.classDifficulty * 200, 80, 4000);
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, Math.round(n)));
}

function detectFencedLanguages(text: string): string[] {
  const out: string[] = [];
  const re = /```([A-Za-z0-9+#]+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) out.push((m[1] as string).toLowerCase());
  return out;
}

/* ------------------------------------------------------------------ */
/* Vectorization                                                       */
/* ------------------------------------------------------------------ */

/**
 * Map features onto the fixed numeric vector consumed by the learned scorer.
 * Counts are log1p-scaled so that a 10k-line file does not dominate a 10-line one.
 */
export function toVector(f: TaskFeatures): number[] {
  const log1p = (n: number): number => Math.log1p(Math.max(0, n));
  const cls = f.taskClass;
  return [
    1, // bias
    log1p(f.estInputTokens) / 10,
    log1p(f.estOutputTokens) / 8,
    log1p(f.fileCount),
    log1p(f.changedLines) / 5,
    log1p(f.loopCount),
    log1p(f.funcCount),
    log1p(f.maxNesting) / 2,
    log1p(f.branchCount) / 3,
    f.hasAsync ? 1 : 0,
    f.hasConcurrency ? 1 : 0,
    f.hasTypes ? 1 : 0,
    f.hasErrorHandling ? 1 : 0,
    f.hasTestsInScope ? 1 : 0,
    f.hasStackTrace ? 1 : 0,
    f.hasReproSteps ? 1 : 0,
    f.hasExplicitAcceptanceCriteria ? 1 : 0,
    f.hasExternalApiMention ? 1 : 0,
    f.hasPerfLanguage ? 1 : 0,
    f.hasSecurityLanguage ? 1 : 0,
    f.hasMigrationLanguage ? 1 : 0,
    f.isQuestion ? 1 : 0,
    f.isExplainOnly ? 1 : 0,
    log1p(f.constraintCount),
    f.ambiguity,
    f.locality,
    f.mentionsSpecificSymbol ? 1 : 0,
    f.classDifficulty,
    cls === 'local-edit' ? 1 : 0,
    cls === 'bugfix-local' ? 1 : 0,
    cls === 'add-validation' ? 1 : 0,
    cls === 'rename' || cls === 'format' || cls === 'prompt-edit' ? 1 : 0,
    cls === 'write-tests' ? 1 : 0,
    cls === 'docs' || cls === 'explain' ? 1 : 0,
    cls === 'refactor-multi' ? 1 : 0,
    cls === 'feature-new' ? 1 : 0,
    cls === 'perf' ? 1 : 0,
    cls === 'debug-unknown' ? 1 : 0,
    cls === 'concurrency' ? 1 : 0,
    cls === 'security' ? 1 : 0,
    cls === 'migration' ? 1 : 0,
    cls === 'architecture' ? 1 : 0,
    cls === 'algorithm' ? 1 : 0,
  ];
}

/** Guard against silent feature/vector drift. */
export function assertVectorShape(v: number[]): void {
  if (v.length !== FEATURE_NAMES.length) {
    throw new Error(
      `feature vector length ${v.length} does not match FEATURE_NAMES (${FEATURE_NAMES.length}); ` +
        `bump FEATURE_VECTOR_VERSION and retrain`,
    );
  }
}

export { FEATURE_VECTOR_VERSION };
