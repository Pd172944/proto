/**
 * A tiny argv parser. No dependency, explicit about what it accepts.
 *
 * Grammar: `proto <command> [subcommand] [positional...] [--flag[=value]] [-abc]`
 *
 * Flags are declared per-command so that unknown flags fail loudly instead of
 * being silently ignored — a silent typo in `--aply` would otherwise look like
 * a successful dry run, which is exactly the class of bug this project cannot
 * afford (it mutates user source files).
 */

export interface FlagSpec {
  name: string;
  type: 'boolean' | 'string' | 'number';
  alias?: string;
  description: string;
  default?: string | number | boolean;
  /** Multiple occurrences accumulate into an array. */
  multiple?: boolean;
}

export interface ParsedArgs {
  positionals: string[];
  flags: Record<string, string | number | boolean | string[]>;
  errors: string[];
}

export function parseArgs(argv: string[], spec: FlagSpec[]): ParsedArgs {
  const byName = new Map<string, FlagSpec>();
  for (const s of spec) {
    byName.set(s.name, s);
    if (s.alias) byName.set(s.alias, s);
  }

  const flags: Record<string, string | number | boolean | string[]> = {};
  const positionals: string[] = [];
  const errors: string[] = [];

  const setFlag = (name: string, rawValue: string | boolean, hadExplicitValue: boolean): void => {
    const s = byName.get(name);
    if (!s) {
      errors.push(`unknown flag: ${name.startsWith('-') ? name : `--${name}`}`);
      return;
    }
    let value: string | number | boolean;
    if (s.type === 'boolean') {
      if (typeof rawValue === 'string' && hadExplicitValue) {
        value = rawValue !== 'false' && rawValue !== '0';
      } else {
        value = true;
      }
    } else if (s.type === 'number') {
      const n = Number(rawValue);
      if (!Number.isFinite(n)) {
        errors.push(`flag --${s.name} expects a number, got ${JSON.stringify(rawValue)}`);
        return;
      }
      value = n;
    } else {
      value = typeof rawValue === 'string' ? rawValue : 'true';
    }
    if (s.multiple) {
      const prev = flags[s.name];
      const arr = Array.isArray(prev) ? prev : prev === undefined ? [] : [String(prev)];
      arr.push(String(value));
      flags[s.name] = arr;
    } else {
      flags[s.name] = value;
    }
  };

  for (let i = 0; i < argv.length; i++) {
    const token = argv[i] as string;
    if (token === '--') {
      positionals.push(...argv.slice(i + 1));
      break;
    }
    if (token.startsWith('--')) {
      const eq = token.indexOf('=');
      if (eq >= 0) {
        setFlag(token.slice(2, eq), token.slice(eq + 1), true);
      } else {
        const spec = byName.get(token.slice(2));
        if (spec && spec.type !== 'boolean') {
          const next = argv[i + 1];
          if (next === undefined || next.startsWith('-')) {
            errors.push(`flag --${spec.name} requires a value`);
          } else {
            setFlag(spec.name, next, true);
            i++;
          }
        } else {
          setFlag(token.slice(2), true, false);
        }
      }
    } else if (token.startsWith('-') && token.length > 1) {
      // Short flags: -a, -abc (all boolean), or -n 5 / -n5 for valued flags.
      const chars = token.slice(1).split('');
      for (let c = 0; c < chars.length; c++) {
        const ch = chars[c] as string;
        const spec = byName.get(ch);
        if (!spec) {
          errors.push(`unknown flag: -${ch}`);
          continue;
        }
        if (spec.type === 'boolean') {
          setFlag(spec.name, true, false);
        } else {
          const rest = chars.slice(c + 1).join('');
          if (rest) {
            setFlag(spec.name, rest, true);
            c = chars.length;
          } else {
            const next = argv[i + 1];
            if (next === undefined) {
              errors.push(`flag -${ch} requires a value`);
            } else {
              setFlag(spec.name, next, true);
              i++;
            }
          }
        }
      }
    } else {
      positionals.push(token);
    }
  }

  // Apply defaults for anything not supplied.
  for (const s of spec) {
    if (flags[s.name] === undefined && s.default !== undefined) flags[s.name] = s.default;
  }

  return { positionals, flags, errors };
}

export function flagString(p: ParsedArgs, name: string): string | undefined {
  const v = p.flags[name];
  if (v === undefined) return undefined;
  if (Array.isArray(v)) return v[v.length - 1];
  return String(v);
}

export function flagBool(p: ParsedArgs, name: string): boolean {
  return p.flags[name] === true || p.flags[name] === 'true';
}

export function flagNumber(p: ParsedArgs, name: string): number | undefined {
  const v = p.flags[name];
  if (v === undefined || Array.isArray(v)) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

export function flagList(p: ParsedArgs, name: string): string[] {
  const v = p.flags[name];
  if (v === undefined) return [];
  return Array.isArray(v) ? v : [String(v)];
}

export function renderFlagHelp(spec: FlagSpec[]): string {
  const rows = spec.map((s) => {
    const left = `  ${s.alias ? `-${s.alias}, ` : '    '}--${s.name}${s.type === 'string' ? ' <value>' : s.type === 'number' ? ' <n>' : ''}`;
    return `${left.padEnd(34)}${s.description}${s.default !== undefined ? ` (default: ${String(s.default)})` : ''}`;
  });
  return rows.join('\n');
}
