/**
 * A `.gitignore` matcher, for the case where the workspace is not a git repository.
 *
 * When the workspace *is* a repository, discovery shells out to `git ls-files` and this
 * module is never used — git already knows exactly which files it tracks and excludes,
 * including nested `.gitignore` files, `.git/info/exclude`, and the user's global
 * excludes. Reimplementing that is a losing game, so the only reason this exists is the
 * case git cannot answer: a plain directory of source.
 *
 * It implements the parts of the gitignore grammar that change which files are indexed:
 * comments, negation, directory-only patterns, anchoring, `**`, `*` and `?`. It does not
 * implement character classes beyond `?`, or the `\[` escapes, because a mis-parsed
 * bracket expression costs a file that the model can still find with `search`.
 *
 * The failure direction is chosen deliberately: when a rule is ambiguous this treats the
 * path as *not ignored*. Hiding a file the model needed is a silent wrong answer; showing
 * one it did not need costs a little context.
 */

export interface IgnoreRule {
  /** Directory the rule came from, POSIX-relative to the scan root. Empty for the root. */
  base: string;
  negated: boolean;
  dirOnly: boolean;
  re: RegExp;
}

/** Translate one gitignore pattern into a regular expression over a relative path. */
function globToRegExp(pattern: string): RegExp {
  let p = pattern;
  let anchored = false;

  if (p.startsWith('/')) {
    anchored = true;
    p = p.slice(1);
  }
  // A slash anywhere but the end anchors the pattern to the .gitignore's directory.
  if (p.slice(0, -1).includes('/')) anchored = true;

  let re = '';
  for (let i = 0; i < p.length; i++) {
    const ch = p[i] as string;
    if (ch === '*') {
      const isDouble = p[i + 1] === '*';
      if (isDouble) {
        // `**/` spans zero or more directories; a trailing `**` spans everything.
        if (p[i + 2] === '/') {
          re += '(?:[^/]+/)*';
          i += 2;
        } else {
          re += '.*';
          i += 1;
        }
      } else {
        re += '[^/]*';
      }
      continue;
    }
    if (ch === '?') {
      re += '[^/]';
      continue;
    }
    if ('\\^$+?.()|{}[]'.includes(ch)) {
      re += `\\${ch}`;
      continue;
    }
    re += ch;
  }

  // An unanchored pattern matches at any depth, so `build` also excludes `src/build`.
  const prefix = anchored ? '' : '(?:.*/)?';
  return new RegExp(`^${prefix}${re}$`);
}

export function parseGitignore(text: string, base: string): IgnoreRule[] {
  const rules: IgnoreRule[] = [];
  for (const rawLine of text.split('\n')) {
    let line = rawLine.replace(/\r$/, '');
    if (line === '' || line.startsWith('#')) continue;
    // A trailing space is only significant when escaped; unescaped trailing spaces are
    // stripped by git, and editors add them by accident.
    line = line.replace(/(?<!\\)\s+$/, '');
    if (line === '') continue;

    let negated = false;
    if (line.startsWith('!')) {
      negated = true;
      line = line.slice(1);
    }
    let dirOnly = false;
    if (line.endsWith('/')) {
      dirOnly = true;
      line = line.slice(0, -1);
    }
    if (line === '') continue;
    // `build/` excludes the directory *and everything under it*, not just the directory
    // entry itself. A caller checking a file path directly (rather than pruning during a
    // walk) would otherwise be told `build/output.bin` is not ignored.
    const re = globToRegExp(line);
    rules.push({
      base,
      negated,
      dirOnly,
      re: dirOnly ? new RegExp(`${re.source.slice(0, -1)}(?:/.*)?$`) : re,
    });
  }
  return rules;
}

/**
 * Apply rules in order; the last matching rule wins, which is how git resolves a
 * negation that appears after the rule it re-includes.
 */
export function isIgnored(path: string, isDir: boolean, rules: IgnoreRule[]): boolean {
  let ignored = false;
  for (const rule of rules) {
    // `path` is root-relative; each rule matches against its own directory's view.
    let rel = path;
    if (rule.base !== '') {
      if (!path.startsWith(`${rule.base}/`)) continue;
      rel = path.slice(rule.base.length + 1);
    }
    if (rule.re.test(rel)) ignored = !rule.negated;
  }
  return ignored;
}
