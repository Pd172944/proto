/**
 * Terminal presentation layer: the "ember and deep water" theme.
 *
 * Visual inspiration comes from charmbracelet/gum (rounded borders, soft muted
 * colour, generous padding, calm spinners), but the colourway is deliberately
 * different: gum leans magenta/pink, this palette leans warm amber-gold for
 * structure and deep teal-aqua for anything the user can act on. Coral is
 * reserved for failure so that a red line always means something broke.
 *
 * Design constraints worth knowing before editing:
 *
 *  - This module is a *pure string transformer*. It never writes to stdout or
 *    stderr, never reads config, and never holds terminal state beyond a single
 *    colour on/off flag. Callers decide where the string goes.
 *  - Colour is derived at call time (not cached) so `NO_COLOR`, `PROTO_COLOR`
 *    and `process.stdout.isTTY` changes take effect immediately, and so tests
 *    can flip the flag without re-importing the module.
 *  - When colour is off, every renderer emits zero ANSI bytes. Caller-supplied
 *    ANSI inside a body/code/diff argument is stripped in that mode, so the
 *    "disabled output is plain text" guarantee holds even for pre-coloured
 *    input. Text utilities that exist to *measure* (`stripAnsi`,
 *    `visibleWidth`) always operate on their input verbatim.
 *  - Layout is measured with `visibleWidth`, never `String.length`, because
 *    this module emits ANSI and must survive CJK/emoji content.
 */

/** Default layout width in columns. */
export const DEFAULT_WIDTH = 80;

/* ------------------------------------------------------------------ */
/* Colour state                                                        */
/* ------------------------------------------------------------------ */

let forced: boolean | null = null;

/**
 * Force colour on or off, bypassing every form of detection.
 *
 * An explicit call wins over `NO_COLOR`: this is the programmatic override used
 * by tests and by `--color=always`, and an ambient environment variable must
 * not silently disable it. Auto-detection is restored with `resetColorMode`.
 */
export function setColorEnabled(on: boolean): void {
  forced = on;
}

/** Return to automatic detection (`NO_COLOR`, `PROTO_COLOR`, then isTTY). */
export function resetColorMode(): void {
  forced = null;
}

export function colorEnabled(): boolean {
  if (forced !== null) return forced;
  if (process.env['NO_COLOR']) return false;
  // PROTO_COLOR=1 is the escape hatch for piping coloured output into a pager
  // or a log viewer; PROTO_COLOR=0 mirrors it for symmetry.
  if (process.env['PROTO_COLOR'] === '1') return true;
  if (process.env['PROTO_COLOR'] === '0') return false;
  return process.stdout.isTTY === true;
}

/**
 * Does the terminal understand 256-colour SGR?
 *
 * We only need two tiers here: the curated 256-colour ember palette, and a
 * 16-colour approximation for genuinely old terminals. Unknown/empty TERM is
 * treated as capable because every terminal emulator released this decade is.
 */
function use256(): boolean {
  if (process.env['COLORTERM']) return true;
  const term = (process.env['TERM'] ?? '').toLowerCase();
  if (term === '') return true;
  if (term === 'dumb') return false;
  return /(256color|truecolor|direct|kitty|alacritty|wezterm|xterm|screen|tmux|rxvt|iterm|foot|ghostty)/.test(term);
}

function paint(s: string, x256: number, x16: number): string {
  if (!colorEnabled()) return s;
  const code = use256() ? `38;5;${x256}` : String(x16);
  return `\u001b[${code}m${s}\u001b[0m`;
}

/** Strip caller-supplied ANSI when colour is disabled, to keep the guarantee. */
function plain(s: string): string {
  return colorEnabled() ? s : stripAnsi(s);
}

/* ------------------------------------------------------------------ */
/* Palette                                                             */
/* ------------------------------------------------------------------ */

export interface Palette {
  readonly primary: (s: string) => string;
  readonly accent: (s: string) => string;
  readonly success: (s: string) => string;
  readonly warn: (s: string) => string;
  readonly error: (s: string) => string;
  readonly text: (s: string) => string;
  readonly dim: (s: string) => string;
  readonly border: (s: string) => string;
  readonly thinking: (s: string) => string;
  readonly bold: (s: string) => string;
}

/**
 * ANSI-256 approximations with 16-colour fallbacks:
 *   primary  amber-gold 179 / yellow 33      accent   deep teal 79 / cyan 36
 *   success  sage green 114 / green 32       warn     gold-orange 215 / yellow 33
 *   error    coral red 203 / red 31          text     warm off-white 252 / white 37
 *   dim      cool grey 245 / grey 90         border   slate 238 / grey 90
 *   thinking violet-grey 103 / magenta 35
 */
export const palette: Palette = {
  primary: (s) => paint(s, 179, 33),
  accent: (s) => paint(s, 79, 36),
  success: (s) => paint(s, 114, 32),
  warn: (s) => paint(s, 215, 33),
  error: (s) => paint(s, 203, 31),
  text: (s) => paint(s, 252, 37),
  dim: (s) => paint(s, 245, 90),
  border: (s) => paint(s, 238, 90),
  thinking: (s) => paint(s, 103, 35),
  bold: (s) => (colorEnabled() ? `\u001b[1m${s}\u001b[0m` : s),
};

/* ------------------------------------------------------------------ */
/* Glyphs                                                              */
/* ------------------------------------------------------------------ */

export interface GlyphSet {
  readonly cornerTL: string;
  readonly cornerTR: string;
  readonly cornerBL: string;
  readonly cornerBR: string;
  readonly vertical: string;
  readonly horizontal: string;
  readonly bullet: string;
  readonly arrow: string;
  readonly check: string;
  readonly cross: string;
  readonly warn: string;
  readonly ellipsis: string;
  readonly dot: string;
}

export const glyph: GlyphSet = {
  cornerTL: '\u256d', // ╭
  cornerTR: '\u256e', // ╮
  cornerBL: '\u2570', // ╰
  cornerBR: '\u256f', // ╯
  vertical: '\u2502', // │
  horizontal: '\u2500', // ─
  bullet: '\u2022', // •
  arrow: '\u2192', // →
  check: '\u2713', // ✓
  cross: '\u2717', // ✗
  warn: '\u26a0', // ⚠
  ellipsis: '\u2026', // …
  dot: '\u00b7', // ·
};

/* ------------------------------------------------------------------ */
/* ANSI-aware measurement                                              */
/* ------------------------------------------------------------------ */

/**
 * Matches one ANSI escape: CSI (`ESC [ ... final`), OSC (`ESC ] ... BEL/ST`),
 * or a two-byte `ESC x` sequence. Covers everything this module emits plus the
 * common sequences third-party code mixes into a body.
 */
const ANSI_ALL =
  /\u001b(?:\[[0-9;:?]*[ -/]*[@-~]|\][^\u0007\u001b]*(?:\u0007|\u001b\\)|[@-Z\\-_])/g;

/** Sticky variant used to test for an escape at a specific index. */
const ANSI_AT =
  /\u001b(?:\[[0-9;:?]*[ -/]*[@-~]|\][^\u0007\u001b]*(?:\u0007|\u001b\\)|[@-Z\\-_])/y;

/** Return the ANSI sequence starting exactly at `i`, or null. */
function ansiAt(s: string, i: number): string | null {
  ANSI_AT.lastIndex = i;
  const m = ANSI_AT.exec(s);
  if (!m || m.index !== i) return null;
  return m[0];
}

export function stripAnsi(s: string): string {
  ANSI_ALL.lastIndex = 0;
  return s.replace(ANSI_ALL, '');
}

/** Zero-width code points: combining marks, joiners, variation selectors. */
function isZeroWidth(cp: number): boolean {
  return (
    (cp >= 0x0300 && cp <= 0x036f) || // combining diacritics
    (cp >= 0x0483 && cp <= 0x0489) ||
    (cp >= 0x200b && cp <= 0x200f) || // ZWSP/ZWNJ/ZWJ/LRM/RLM
    (cp >= 0x2060 && cp <= 0x2064) ||
    cp === 0x200d ||
    (cp >= 0xfe00 && cp <= 0xfe0f) || // variation selectors
    cp === 0xfeff ||
    (cp >= 0x1f3fb && cp <= 0x1f3ff) || // emoji skin-tone modifiers
    (cp >= 0xe0100 && cp <= 0xe01ef) // variation selectors supplement
  );
}

/**
 * Conservative East-Asian-wide check.
 *
 * "Conservative" here means: when a range is genuinely ambiguous (East Asian
 * Ambiguous width), we prefer the double-width reading, because under-counting
 * a wide glyph is what breaks border alignment. Regional-indicator pairs are
 * deliberately *not* counted double each, so a flag emoji totals 2, not 4.
 */
function isWide(cp: number): boolean {
  return (
    (cp >= 0x1100 && cp <= 0x115f) || // Hangul Jamo
    (cp >= 0x2e80 && cp <= 0x303e) || // CJK radicals, Kangxi, CJK punctuation
    (cp >= 0x3041 && cp <= 0x33ff) || // kana, Hangul compat, CJK symbols
    (cp >= 0x3400 && cp <= 0x4dbf) || // CJK ext A
    (cp >= 0x4e00 && cp <= 0x9fff) || // CJK unified
    (cp >= 0xa000 && cp <= 0xa4cf) || // Yi
    (cp >= 0xa960 && cp <= 0xa97f) ||
    (cp >= 0xac00 && cp <= 0xd7a3) || // Hangul syllables
    (cp >= 0xf900 && cp <= 0xfaff) || // CJK compat ideographs
    (cp >= 0xfe10 && cp <= 0xfe19) ||
    (cp >= 0xfe30 && cp <= 0xfe6f) ||
    (cp >= 0xff00 && cp <= 0xff60) || // fullwidth forms
    (cp >= 0xffe0 && cp <= 0xffe6) ||
    (cp >= 0x1b000 && cp <= 0x1b2ff) ||
    (cp >= 0x1f300 && cp <= 0x1f64f) || // misc symbols + emoticons
    (cp >= 0x1f680 && cp <= 0x1f6ff) || // transport
    (cp >= 0x1f7e0 && cp <= 0x1f7eb) ||
    (cp >= 0x1f90c && cp <= 0x1f9ff) || // supplemental symbols
    (cp >= 0x1fa70 && cp <= 0x1faff) ||
    (cp >= 0x20000 && cp <= 0x2fffd) || // CJK ext B+
    (cp >= 0x30000 && cp <= 0x3fffd)
  );
}

/** Columns occupied by a single code point (0 for controls/zero-width). */
function charWidth(cp: number): number {
  if (cp < 32) return 0;
  if (cp === 0x7f) return 0;
  if (isZeroWidth(cp)) return 0;
  if (cp >= 0x2600 && cp <= 0x26ff) return 2; // misc symbols (⚠, ⛔, …)
  if (isWide(cp)) return 2;
  return 1;
}

/**
 * One terminal "unit" starting at `i`: either an ANSI escape (zero columns) or
 * a base code point plus any ZWJ-joined continuations.
 *
 * Folding `ZWJ + base` into the *preceding* unit is what makes emoji sequences
 * like 👩‍💻 measure 2 instead of 4, which is how every modern terminal renders
 * them — and therefore what keeps box borders aligned.
 */
function graphemeAt(s: string, i: number): { width: number; next: number } {
  const esc = ansiAt(s, i);
  if (esc) return { width: 0, next: i + esc.length };
  const cp = s.codePointAt(i);
  if (cp === undefined) return { width: 0, next: s.length };
  const next = i + (cp > 0xffff ? 2 : 1);
  if (cp !== 0x200d) return { width: charWidth(cp), next };

  // Zero-width joiner: absorb the joined base (and any modifiers) at no cost.
  let j = next;
  let consumedBase = false;
  while (j < s.length) {
    const inner = ansiAt(s, j);
    if (inner) {
      j += inner.length;
      continue;
    }
    const cp2 = s.codePointAt(j);
    if (cp2 === undefined) break;
    const step = cp2 > 0xffff ? 2 : 1;
    if (isZeroWidth(cp2)) {
      j += step;
      continue;
    }
    if (consumedBase) break;
    consumedBase = true;
    j += step;
  }
  return { width: 0, next: j };
}

export function visibleWidth(s: string): number {
  let w = 0;
  let i = 0;
  while (i < s.length) {
    const unit = graphemeAt(s, i);
    if (unit.next <= i) break;
    w += unit.width;
    i = unit.next;
  }
  return w;
}

/** Slice a possibly-coloured string to a visible width, adding an ellipsis. */
function truncVisible(s: string, max: number): string {
  if (max <= 0) return '';
  if (visibleWidth(s) <= max) return s;
  const keep = Math.max(0, max - 1);
  let w = 0;
  let i = 0;
  while (i < s.length) {
    const unit = graphemeAt(s, i);
    if (unit.next <= i) break;
    if (w + unit.width > keep) break;
    w += unit.width;
    i = unit.next;
  }
  return s.slice(0, i) + glyph.ellipsis;
}

/**
 * Split a string at the last point that fits `width` visible columns, never
 * cutting through an ANSI escape or a joined emoji sequence. Used to
 * hard-break over-long tokens.
 */
function splitByWidth(s: string, width: number): [string, string] {
  let w = 0;
  let i = 0;
  while (i < s.length) {
    const unit = graphemeAt(s, i);
    if (unit.next <= i) break;
    if (w + unit.width > width) break;
    w += unit.width;
    i = unit.next;
  }
  return [s.slice(0, i), s.slice(i)];
}

/* ------------------------------------------------------------------ */
/* Small shared helpers                                                */
/* ------------------------------------------------------------------ */

/** Clamp a layout width option into the supported [40, 200] band. */
export function clampWidth(width?: number): number {
  if (width === undefined || !Number.isFinite(width)) return DEFAULT_WIDTH;
  return Math.max(40, Math.min(200, Math.floor(width)));
}

function clampInt(n: number, lo: number, hi: number): number {
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, Math.floor(n)));
}

function padRightVisible(s: string, width: number): string {
  const pad = width - visibleWidth(s);
  return pad > 0 ? s + ' '.repeat(pad) : s;
}

function trimEndSpaces(s: string): string {
  return s.replace(/ +$/, '');
}

/* ------------------------------------------------------------------ */
/* Spinner                                                             */
/* ------------------------------------------------------------------ */

/** Ten-frame braille spinner — calm at ~10 fps, reads as motion not noise. */
const SPINNER_FRAMES: readonly string[] = ['\u280b', '\u2819', '\u2839', '\u2838', '\u283c', '\u2834', '\u2826', '\u2827', '\u2807', '\u280f'];

export function spinnerFrame(index: number): string {
  const i = Number.isFinite(index) ? Math.floor(index) : 0;
  const n = SPINNER_FRAMES.length;
  return SPINNER_FRAMES[((i % n) + n) % n] ?? SPINNER_FRAMES[0] ?? '';
}

/* ------------------------------------------------------------------ */
/* Wrapping                                                            */
/* ------------------------------------------------------------------ */

interface Token {
  text: string;
  width: number;
  space: boolean;
}

/**
 * Split a line into word/space tokens, keeping ANSI escapes glued to the
 * neighbouring text. Escapes never become their own break opportunity, so a
 * wrap can never land in the middle of one.
 */
function tokenize(line: string): Token[] {
  const toks: Token[] = [];
  let buf = '';
  let bufW = 0;
  let bufSpace: boolean | null = null;

  const push = (): void => {
    if (buf === '') return;
    toks.push({ text: buf, width: bufW, space: bufSpace ?? false });
    buf = '';
    bufW = 0;
    bufSpace = null;
  };

  let i = 0;
  while (i < line.length) {
    const unit = graphemeAt(line, i);
    if (unit.next <= i) break;
    const chunk = line.slice(i, unit.next);
    i = unit.next;

    if (chunk.charCodeAt(0) === 0x1b) {
      // Glue escapes to whatever is being accumulated so a wrap can never
      // separate a colour from the text it colours.
      buf += chunk;
      continue;
    }

    const isSpace = chunk === ' ' || chunk === '\t';
    if (bufSpace !== null && isSpace !== bufSpace) push();
    bufSpace = isSpace;
    buf += chunk;
    bufW += isSpace ? 1 : unit.width;
  }
  push();
  return toks;
}

function wrapLine(line: string, width: number): string[] {
  if (line === '') return [''];
  if (visibleWidth(line) <= width) return [line];

  const out: string[] = [];
  let cur = '';
  let curW = 0;
  let atLineStart = true;
  const flush = (): void => {
    out.push(trimEndSpaces(cur));
    cur = '';
    curW = 0;
    atLineStart = true;
  };

  for (const tok of tokenize(line)) {
    // Zero-width token: pure ANSI run, or a trailing reset. Attach in place.
    if (!tok.space && tok.width === 0 && tok.text.length > 0) {
      cur += tok.text;
      continue;
    }

    if (tok.space) {
      // Preserve leading indentation on the first output line only; spaces at
      // the start of a continuation line are dropped by the wrap itself.
      if (atLineStart) {
        if (out.length === 0 && curW + tok.width <= width) {
          cur += tok.text;
          curW += tok.width;
        }
        continue;
      }
      if (curW + tok.width <= width) {
        cur += tok.text;
        curW += tok.width;
      } else {
        flush();
      }
      continue;
    }

    if (curW + tok.width <= width) {
      cur += tok.text;
      curW += tok.width;
      atLineStart = false;
      continue;
    }

    // The word does not fit on the current line.
    if (curW > 0 || !atLineStart) flush();

    let rest = tok.text;
    let restW = tok.width;
    while (restW > width) {
      const [head, tail] = splitByWidth(rest, width);
      if (head === '') {
        // A single wide glyph wider than the line: emit it alone rather than
        // loop forever. Only reachable at width 1.
        const cp = rest.codePointAt(0);
        if (cp === undefined) break;
        const ch = String.fromCodePoint(cp);
        out.push(ch);
        rest = rest.slice(ch.length);
        restW = visibleWidth(rest);
        continue;
      }
      out.push(trimEndSpaces(head));
      rest = tail;
      restW = visibleWidth(rest);
    }
    cur = rest;
    curW = restW;
    atLineStart = curW === 0;
  }

  if (cur !== '' || out.length === 0) out.push(trimEndSpaces(cur));
  return out;
}

/**
 * Word-wrap `text` to `width` visible columns.
 *
 * Unlike the panel renderers, this honours the exact width requested (floor of
 * 1) instead of clamping to [40, 200]: callers use it for measurement-sensitive
 * layout, including nested indentation inside boxes.
 *
 * A word wider than the line is hard-broken at the width boundary; ANSI escapes
 * are treated as zero-width and are never split.
 */
export function wrap(text: string, width: number): string[] {
  const w = Math.max(1, Math.floor(Number.isFinite(width) ? width : DEFAULT_WIDTH));
  const src = plain(text);
  const out: string[] = [];
  for (const para of src.split('\n')) out.push(...wrapLine(para, w));
  return out.length > 0 ? out : [''];
}

/* ------------------------------------------------------------------ */
/* Boxes, rules, headers                                               */
/* ------------------------------------------------------------------ */

export interface BoxOptions {
  accent?: keyof Palette;
  width?: number;
  padding?: number;
  subtitle?: string;
}

function oneLineVisible(s: string): string {
  return s.replace(/\s*\n\s*/g, ' ').trim();
}

/**
 * Rounded box with an inline title and an optional bottom-border subtitle.
 *
 * The frame is coloured with `border` (or `accent` when supplied, which is the
 * "focused" look), the title with `primary` (or `accent`). Interior width is
 * `width - 2 - 2*padding`, so padding is a horizontal inset; a padding of 1 or
 * more also adds one blank row above and below the body, which is what gives
 * the box its unhurried gum-like feel.
 */
export function box(title: string, body: string, opts: BoxOptions = {}): string {
  const t = oneLineVisible(plain(title));
  const b = plain(body);
  const subtitle = opts.subtitle === undefined ? '' : oneLineVisible(plain(opts.subtitle));
  const width = clampWidth(opts.width);
  const pad = clampInt(opts.padding ?? 1, 0, 8);
  const vpad = pad > 0 ? 1 : 0;
  const colored = colorEnabled();

  const frame = opts.accent !== undefined ? palette[opts.accent] : palette.border;
  const label = opts.accent !== undefined ? palette[opts.accent] : palette.primary;
  const tl = colored ? glyph.cornerTL : '+';
  const tr = colored ? glyph.cornerTR : '+';
  const bl = colored ? glyph.cornerBL : '+';
  const br = colored ? glyph.cornerBR : '+';
  const hz = colored ? glyph.horizontal : '-';
  const vt = colored ? glyph.vertical : '|';

  const inner = Math.max(1, width - 2);
  const contentW = Math.max(1, inner - pad * 2);
  const lines: string[] = [];

  // Top border: `╭─ title ─────╮`.
  const titleText = truncVisible(t, Math.max(0, inner - 4));
  if (titleText === '') {
    lines.push(frame(tl + hz.repeat(inner) + tr));
  } else {
    const fill = Math.max(0, inner - visibleWidth(titleText) - 3);
    lines.push(frame(tl + hz) + ' ' + label(titleText) + ' ' + frame(hz.repeat(fill) + tr));
  }

  const side = (s: string): string =>
    frame(vt) + ' '.repeat(pad) + padRightVisible(s, contentW) + ' '.repeat(pad) + frame(vt);
  const blank = side('');

  for (let i = 0; i < vpad; i++) lines.push(blank);
  for (const raw of b.split('\n')) {
    for (const piece of wrap(raw, contentW)) lines.push(side(piece));
  }
  for (let i = 0; i < vpad; i++) lines.push(blank);

  // Bottom border: `╰──── subtitle ────╯` (subtitle centred, min one dash).
  if (subtitle === '') {
    lines.push(frame(bl + hz.repeat(inner) + br));
  } else {
    const sub = truncVisible(subtitle, Math.max(0, inner - 4));
    const labelW = visibleWidth(sub) + 2;
    const remaining = inner - labelW;
    const left = remaining <= 0 ? 0 : Math.max(1, Math.floor(remaining / 2));
    const right = Math.max(0, remaining - left);
    lines.push(frame(bl + hz.repeat(left)) + palette.dim(' ' + sub + ' ') + frame(hz.repeat(right) + br));
  }

  return lines.join('\n');
}

/** App header: an amber title line over a thin dim rule, then dim sublines. */
export function banner(lines: string[], width = DEFAULT_WIDTH): string {
  const w = clampWidth(width);
  const src = lines.map((l) => plain(l));
  const out: string[] = [];
  const head = src[0];
  if (head !== undefined && head !== '') out.push('  ' + palette.primary(palette.bold(head)));
  out.push('  ' + palette.border(glyph.horizontal.repeat(Math.max(0, w - 4))));
  for (const l of src.slice(1)) {
    for (const piece of wrap(l, Math.max(1, w - 4))) out.push('  ' + palette.dim(piece));
  }
  return out.join('\n');
}

/** Thin horizontal rule, optionally labelled: `── label ──────`. */
export function rule(label?: string, width = DEFAULT_WIDTH): string {
  const w = clampWidth(width);
  if (label === undefined || label === '') return palette.border(glyph.horizontal.repeat(w));
  const text = oneLineVisible(plain(label));
  const textW = visibleWidth(text);
  if (textW + 3 > w) return palette.border(truncVisible(text, w));
  const right = Math.max(0, w - textW - 3);
  return palette.border(glyph.horizontal) + ' ' + palette.primary(text) + ' ' + palette.border(glyph.horizontal.repeat(right));
}

/* ------------------------------------------------------------------ */
/* Key/value and status lines                                          */
/* ------------------------------------------------------------------ */

export interface KeyValue {
  key: string;
  value: string;
  note?: string;
}

/** Aligned key/value block; keys dim, values plain, notes parenthesised/dim. */
export function kv(pairs: KeyValue[], opts: { indent?: number; keyWidth?: number } = {}): string {
  const indent = clampInt(opts.indent ?? 0, 0, 64);
  const pad = ' '.repeat(indent);
  const keys = pairs.map((p) => visibleWidth(plain(p.key)));
  const keyW = Math.max(1, clampInt(opts.keyWidth ?? (keys.length > 0 ? Math.max(...keys) : 1), 1, 64));
  return pairs
    .map((p) => {
      const key = plain(p.key);
      const value = plain(p.value);
      // Pad to a fixed key column, then always two spaces: the value column
      // must line up whether or not a key happens to fill the column exactly.
      const gap = Math.max(0, keyW - visibleWidth(key));
      let line = `${pad}${palette.dim(key)}${' '.repeat(gap + 2)}${palette.text(value)}`;
      if (p.note !== undefined && p.note !== '') line += `  ${palette.dim('(' + plain(p.note) + ')')}`;
      return line;
    })
    .join('\n');
}

/** Single-line status strip: `model x · tokens 1,204 · cost $0.0031`. */
export function statusLine(parts: KeyValue[], width = DEFAULT_WIDTH): string {
  const w = clampWidth(width);
  const sep = ' ' + palette.border(glyph.dot) + ' ';
  const cells = parts.map(
    (p) => `${palette.dim(plain(p.key))} ${palette.text(plain(p.value))}`,
  );
  while (cells.length > 1 && visibleWidth(cells.join(sep)) > w) cells.pop();
  const line = cells.join(sep);
  return visibleWidth(line) > w ? truncVisible(line, w) : line;
}

/* ------------------------------------------------------------------ */
/* Approval panel                                                      */
/* ------------------------------------------------------------------ */

const KIND_ACCENT: Record<'read' | 'write' | 'exec', keyof Palette> = {
  read: 'accent',
  write: 'warn',
  exec: 'error',
};

const KIND_VERB: Record<'read' | 'write' | 'exec', string> = {
  read: 'read access',
  write: 'write access',
  exec: 'command execution',
};

/**
 * Consent prompt for a tool call. The kind is always spelled out in the body
 * (so it survives a log or a transcript) and drives the frame colour: teal for
 * reads, gold for writes, coral for execution.
 */
export function approvalPanel(
  req: { title: string; detail: string; kind: 'read' | 'write' | 'exec'; warning?: string },
  width = DEFAULT_WIDTH,
): string {
  const accent = KIND_ACCENT[req.kind];
  const kindColor = palette[accent];
  const body: string[] = [];
  for (const line of plain(req.detail).split('\n')) body.push(line);
  body.push('');
  body.push(`${palette.dim('kind')}     ${kindColor(req.kind)}   ${palette.dim('answer')} approve / deny`);
  if (req.warning !== undefined && req.warning !== '') {
    body.push('');
    body.push(palette.warn(`${glyph.warn} ${plain(req.warning)}`));
  }
  return box(plain(req.title), body.join('\n'), { width, subtitle: KIND_VERB[req.kind], accent });
}

/* ------------------------------------------------------------------ */
/* Code blocks and lightweight highlighting                            */
/* ------------------------------------------------------------------ */

interface LangSpec {
  keywords: readonly string[];
  comments: 'slash' | 'hash' | 'both' | 'none';
}

const TS_KEYWORDS = [
  'const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'do', 'import', 'export', 'from',
  'default', 'interface', 'type', 'class', 'extends', 'implements', 'new', 'await', 'async', 'try', 'catch',
  'finally', 'throw', 'switch', 'case', 'break', 'continue', 'typeof', 'instanceof', 'null', 'undefined', 'true',
  'false', 'this', 'void', 'readonly', 'in', 'of', 'as', 'satisfies', 'yield', 'static', 'public', 'private',
  'protected', 'delete', 'never', 'unknown', 'string', 'number', 'boolean',
];
const PY_KEYWORDS = [
  'def', 'return', 'if', 'elif', 'else', 'for', 'while', 'import', 'from', 'as', 'class', 'try', 'except',
  'finally', 'raise', 'with', 'lambda', 'yield', 'async', 'await', 'pass', 'break', 'continue', 'in', 'is',
  'not', 'and', 'or', 'None', 'True', 'False', 'global', 'nonlocal', 'assert', 'del', 'self',
];
const RUST_KEYWORDS = [
  'fn', 'let', 'mut', 'const', 'struct', 'enum', 'impl', 'trait', 'pub', 'use', 'mod', 'match', 'if', 'else',
  'for', 'while', 'loop', 'return', 'self', 'Self', 'crate', 'super', 'async', 'await', 'move', 'ref', 'where',
  'dyn', 'static', 'unsafe', 'true', 'false',
];
const GO_KEYWORDS = [
  'func', 'var', 'const', 'type', 'struct', 'interface', 'map', 'chan', 'go', 'defer', 'return', 'if', 'else',
  'for', 'range', 'switch', 'case', 'default', 'break', 'continue', 'package', 'import', 'select', 'nil', 'true',
  'false', 'string', 'int', 'error',
];
const SH_KEYWORDS = [
  'if', 'then', 'else', 'elif', 'fi', 'for', 'in', 'do', 'done', 'while', 'case', 'esac', 'function', 'return',
  'export', 'local', 'echo', 'exit', 'set', 'source', 'cd',
];
const C_KEYWORDS = [
  'int', 'char', 'long', 'short', 'unsigned', 'signed', 'float', 'double', 'void', 'struct', 'union', 'enum',
  'typedef', 'static', 'const', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break',
  'continue', 'sizeof', 'include', 'define', 'class', 'public', 'private', 'protected', 'namespace', 'template',
  'new', 'delete', 'nullptr', 'true', 'false', 'auto', 'using', 'try', 'catch', 'throw',
];
const SQL_KEYWORDS = [
  'select', 'from', 'where', 'insert', 'into', 'values', 'update', 'set', 'delete', 'create', 'table', 'drop',
  'alter', 'join', 'left', 'right', 'inner', 'outer', 'on', 'group', 'by', 'order', 'having', 'limit', 'as',
  'and', 'or', 'not', 'null', 'primary', 'key', 'index',
];

const LANG_SPECS: Record<string, LangSpec> = {
  ts: { keywords: TS_KEYWORDS, comments: 'slash' },
  typescript: { keywords: TS_KEYWORDS, comments: 'slash' },
  tsx: { keywords: TS_KEYWORDS, comments: 'slash' },
  js: { keywords: TS_KEYWORDS, comments: 'slash' },
  javascript: { keywords: TS_KEYWORDS, comments: 'slash' },
  jsx: { keywords: TS_KEYWORDS, comments: 'slash' },
  py: { keywords: PY_KEYWORDS, comments: 'hash' },
  python: { keywords: PY_KEYWORDS, comments: 'hash' },
  rs: { keywords: RUST_KEYWORDS, comments: 'slash' },
  rust: { keywords: RUST_KEYWORDS, comments: 'slash' },
  go: { keywords: GO_KEYWORDS, comments: 'slash' },
  golang: { keywords: GO_KEYWORDS, comments: 'slash' },
  sh: { keywords: SH_KEYWORDS, comments: 'hash' },
  bash: { keywords: SH_KEYWORDS, comments: 'hash' },
  zsh: { keywords: SH_KEYWORDS, comments: 'hash' },
  shell: { keywords: SH_KEYWORDS, comments: 'hash' },
  c: { keywords: C_KEYWORDS, comments: 'slash' },
  h: { keywords: C_KEYWORDS, comments: 'slash' },
  cpp: { keywords: C_KEYWORDS, comments: 'slash' },
  'c++': { keywords: C_KEYWORDS, comments: 'slash' },
  java: { keywords: C_KEYWORDS, comments: 'slash' },
  sql: { keywords: SQL_KEYWORDS, comments: 'slash' },
  json: { keywords: [], comments: 'none' },
  yaml: { keywords: [], comments: 'hash' },
  yml: { keywords: [], comments: 'hash' },
  toml: { keywords: [], comments: 'hash' },
};

const HL_CACHE = new Map<string, RegExp>();

function highlightRe(spec: LangSpec): RegExp {
  const cacheKey = `${spec.comments}:${spec.keywords.join(',')}`;
  const cached = HL_CACHE.get(cacheKey);
  if (cached) return cached;

  const commentParts: string[] = [];
  if (spec.comments === 'slash' || spec.comments === 'both') commentParts.push('\\/\\/[^\\n]*', '\\/\\*[\\s\\S]*?\\*\\/');
  if (spec.comments === 'hash' || spec.comments === 'both') commentParts.push('#[^\\n]*');
  const commentAlt = commentParts.length > 0 ? commentParts.join('|') : '(?!x)x';

  const stringAlt = '"(?:\\\\.|[^"\\\\\\n])*"|\'(?:\\\\.|[^\'\\\\\\n])*\'|`(?:\\\\.|[^`\\\\])*`';
  const keywordAlt = spec.keywords.length > 0 ? `\\b(?:${spec.keywords.join('|')})\\b` : '(?!x)x';

  const re = new RegExp(`(${commentAlt})|(${stringAlt})|(${keywordAlt})`, 'g');
  HL_CACHE.set(cacheKey, re);
  return re;
}

/**
 * Tiny keyword/string/comment highlighter.
 *
 * Unknown languages return the input untouched, which is the important
 * property: this must never mangle plain text, so it only ever runs for a
 * allow-listed language and never rewrites anything it does not match.
 */
function highlight(code: string, lang?: string): string {
  if (!colorEnabled() || code === '') return code;
  const spec = lang === undefined ? undefined : LANG_SPECS[lang.toLowerCase()];
  if (spec === undefined) return code;
  const re = highlightRe(spec);
  re.lastIndex = 0;
  return code.replace(re, (match: string, comment?: string, str?: string, keyword?: string) => {
    if (comment) return palette.dim(comment);
    if (str) return palette.success(str);
    if (keyword) return palette.accent(keyword);
    return match;
  });
}

export interface CodeBlockOptions {
  width?: number;
  maxLines?: number;
}

/**
 * Code with a subtle left gutter naming the language.
 *
 * Lines are wrapped to the remaining width; when `maxLines` is exceeded the
 * remainder is replaced by an explicit `… N more lines` note so a caller can
 * never silently lose content.
 */
export function codeBlock(code: string, lang?: string, opts: CodeBlockOptions = {}): string {
  const src = plain(code);
  const width = clampWidth(opts.width);
  const maxLines = Math.max(1, clampInt(opts.maxLines ?? 40, 1, 5000));
  const tag = (lang ?? '').trim().slice(0, 12);
  const gutterW = Math.max(4, Math.min(12, tag.length));
  const avail = Math.max(8, width - gutterW - 3);

  let raw = src.split('\n');
  if (raw.length > 1 && raw[raw.length - 1] === '') raw.pop();
  if (raw.length === 0) raw = [''];

  const shown = raw.slice(0, maxLines);
  const dropped = raw.length - shown.length;
  const out: string[] = [];

  shown.forEach((line) => {
    const pieces = wrap(highlight(line, lang), avail);
    pieces.forEach((piece, i) => {
      const gut = i === 0 ? palette.dim(padRightVisible(tag, gutterW)) : ' '.repeat(gutterW);
      out.push(`${gut} ${palette.border(glyph.vertical)} ${piece}`);
    });
  });

  if (dropped > 0) {
    out.push(palette.dim(`${' '.repeat(gutterW)} ${glyph.ellipsis} ${dropped} more lines`));
  }
  return out.join('\n');
}

/* ------------------------------------------------------------------ */
/* Diff                                                                */
/* ------------------------------------------------------------------ */

export interface DiffOptions {
  width?: number;
  context?: number;
  maxLines?: number;
}

interface Op {
  kind: 'equal' | 'del' | 'ins';
  text: string;
}

interface Hunk {
  aStart: number;
  aCount: number;
  bStart: number;
  bCount: number;
  ops: Op[];
}

/** LCS table size above which we refuse the quadratic path. */
const LCS_MAX_CELLS = 2000 * 2000;
/** Line count above which either input switches to the positional fallback. */
const LCS_MAX_LINES = 2000;

function splitDiffLines(s: string): string[] {
  if (s === '') return [];
  const lines = s.split('\n');
  if (lines.length > 1 && lines[lines.length - 1] === '') lines.pop();
  return lines;
}

/**
 * Straightforward dynamic-programming LCS. Myers would use less memory, but
 * this is a terminal renderer for human-sized diffs and the code is far easier
 * to verify; the cell guard keeps a pathological input from allocating GBs.
 */
function lcsOps(a: string[], b: string[]): Op[] {
  const n = a.length;
  const m = b.length;
  if (n === 0) return b.map((text) => ({ kind: 'ins' as const, text }));
  if (m === 0) return a.map((text) => ({ kind: 'del' as const, text }));
  if (n * m > LCS_MAX_CELLS) return positionalOps(a, b);

  const w = m + 1;
  const dp = new Int32Array((n + 1) * w);
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      if (a[i] === b[j]) dp[i * w + j] = (dp[(i + 1) * w + (j + 1)] ?? 0) + 1;
      else dp[i * w + j] = Math.max(dp[(i + 1) * w + j] ?? 0, dp[i * w + (j + 1)] ?? 0);
    }
  }

  const ops: Op[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      ops.push({ kind: 'equal', text: a[i] ?? '' });
      i++;
      j++;
    } else if ((dp[(i + 1) * w + j] ?? 0) >= (dp[i * w + (j + 1)] ?? 0)) {
      ops.push({ kind: 'del', text: a[i] ?? '' });
      i++;
    } else {
      ops.push({ kind: 'ins', text: b[j] ?? '' });
      j++;
    }
  }
  while (i < n) {
    ops.push({ kind: 'del', text: a[i] ?? '' });
    i++;
  }
  while (j < m) {
    ops.push({ kind: 'ins', text: b[j] ?? '' });
    j++;
  }
  return ops;
}

/** Positional fallback: cheap, linear, and honest about being approximate. */
function positionalOps(a: string[], b: string[]): Op[] {
  const ops: Op[] = [];
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const av = a[i];
    const bv = b[i];
    if (av !== undefined && bv !== undefined) {
      if (av === bv) ops.push({ kind: 'equal', text: av });
      else {
        ops.push({ kind: 'del', text: av });
        ops.push({ kind: 'ins', text: bv });
      }
    } else if (av !== undefined) {
      ops.push({ kind: 'del', text: av });
    } else if (bv !== undefined) {
      ops.push({ kind: 'ins', text: bv });
    }
  }
  return ops;
}

/** Group changed runs into hunks, padding each with `context` equal lines. */
function buildHunks(ops: Op[], context: number): Hunk[] {
  const changed: number[] = [];
  ops.forEach((op, i) => {
    if (op.kind !== 'equal') changed.push(i);
  });
  if (changed.length === 0) return [];

  const groups: Array<[number, number]> = [];
  let start = changed[0] ?? 0;
  let prev = start;
  for (const idx of changed.slice(1)) {
    if (idx - prev <= context * 2 + 1) {
      prev = idx;
      continue;
    }
    groups.push([start, prev]);
    start = idx;
    prev = idx;
  }
  groups.push([start, prev]);

  return groups.map(([s, e]) => {
    const from = Math.max(0, s - context);
    const to = Math.min(ops.length - 1, e + context);
    let aStart = 1;
    let bStart = 1;
    for (let k = 0; k < from; k++) {
      const op = ops[k];
      if (op === undefined) continue;
      if (op.kind !== 'ins') aStart++;
      if (op.kind !== 'del') bStart++;
    }
    let aCount = 0;
    let bCount = 0;
    const slice: Op[] = [];
    for (let k = from; k <= to; k++) {
      const op = ops[k];
      if (op === undefined) continue;
      slice.push(op);
      if (op.kind !== 'ins') aCount++;
      if (op.kind !== 'del') bCount++;
    }
    return { aStart, aCount, bStart, bCount, ops: slice };
  });
}

function renderOp(op: Op): string {
  if (op.kind === 'equal') return palette.dim(' ' + op.text);
  if (op.kind === 'del') return palette.error('-' + op.text);
  return palette.success('+' + op.text);
}

/**
 * Unified-style line diff.
 *
 * Real LCS diffing up to 2000 lines per side; beyond that (or beyond ~4M DP
 * cells) it degrades to a positional line-by-line comparison and says so in
 * the header, because a quadratic table is not worth the pause on an input no
 * human will read line by line anyway.
 */
export function diff(before: string, after: string, path: string, opts: DiffOptions = {}): string {
  const width = clampWidth(opts.width);
  const context = clampInt(opts.context ?? 2, 0, 50);
  const maxLines = Math.max(4, clampInt(opts.maxLines ?? 200, 4, 10000));
  const a = splitDiffLines(plain(before));
  const b = splitDiffLines(plain(after));
  const large = a.length > LCS_MAX_LINES || b.length > LCS_MAX_LINES;
  const ops = large ? positionalOps(a, b) : lcsOps(a, b);
  const hunks = buildHunks(ops, context);

  const out: string[] = [];
  const header = large ? `diff ${plain(path)}  (large input: line-by-line fallback)` : `diff ${plain(path)}`;
  out.push(rule(header, width));

  if (hunks.length === 0) {
    out.push(palette.dim(`  ${glyph.check} no changes`));
  } else {
    for (const h of hunks) {
      out.push(palette.accent(`@@ -${h.aStart},${h.aCount} +${h.bStart},${h.bCount} @@`));
      for (const op of h.ops) out.push(renderOp(op));
    }
  }

  if (out.length > maxLines) {
    const dropped = out.length - maxLines;
    return out
      .slice(0, maxLines)
      .concat(palette.dim(`  ${glyph.ellipsis} ${dropped} more lines (truncated)`))
      .join('\n');
  }
  return out.join('\n');
}

/* ------------------------------------------------------------------ */
/* Markdown-lite                                                       */
/* ------------------------------------------------------------------ */

/**
 * Inline spans for one already-wrapped segment.
 *
 * Inline code is lifted into placeholders first so that a `**` inside a code
 * span can never be re-read as bold. Unmatched markers are simply left alone,
 * which is what keeps malformed markdown from throwing or eating text.
 */
function inlineSpans(s: string): string {
  const codes: string[] = [];
  const marked = s.replace(/`([^`\n]*)`/g, (_m: string, code: string) => {
    const idx = codes.push(palette.accent(code)) - 1;
    return `\u0000${idx}\u0000`;
  });
  const bolded = marked.replace(/\*\*([^*\n]+)\*\*/g, (_m: string, body: string) => palette.bold(body));
  return bolded.replace(/\u0000(\d+)\u0000/g, (_m: string, i: string) => codes[Number(i)] ?? '');
}

function flushFence(out: string[], code: string[], lang: string, width: number): void {
  out.push(codeBlock(code.join('\n'), lang, { width }));
}

/**
 * A deliberately small markdown subset: headings, bullets, numbered lists,
 * fenced code, bold and inline code. Anything else passes through untouched.
 *
 * Formatting is applied *after* wrapping each logical line, so a wrap can never
 * split a `**bold**` pair and leave an unbalanced escape in the output.
 */
export function markdownLite(text: string, width: number): string {
  const w = clampWidth(width);
  try {
    const src = plain(text);
    const out: string[] = [];
    let inFence = false;
    let fenceLang = '';
    let fenceLines: string[] = [];

    for (const line of src.split('\n')) {
      if (/^\s*```/.test(line)) {
        if (inFence) {
          flushFence(out, fenceLines, fenceLang, w);
          inFence = false;
          fenceLang = '';
          fenceLines = [];
        } else {
          inFence = true;
          fenceLang = (line.match(/^\s*```+\s*([^\s`]*)/)?.[1] ?? '').trim();
        }
        continue;
      }
      if (inFence) {
        fenceLines.push(line);
        continue;
      }

      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        const level = (heading[1] ?? '').length;
        const body = (heading[2] ?? '').trim();
        const prefix = level <= 2 ? palette.primary('\u258d ') : '';
        const prefixW = visibleWidth(prefix);
        wrap(body, Math.max(1, w - prefixW)).forEach((piece) => {
          out.push(prefix + palette.primary(palette.bold(piece)));
        });
        continue;
      }

      const bullet = line.match(/^(\s*)[-*+]\s+(.*)$/);
      if (bullet) {
        const prefix = palette.accent(glyph.bullet) + ' ';
        const prefixW = visibleWidth(prefix);
        wrap(bullet[2] ?? '', Math.max(1, w - prefixW)).forEach((piece, i) => {
          out.push((i === 0 ? prefix : ' '.repeat(prefixW)) + inlineSpans(piece));
        });
        continue;
      }

      const numbered = line.match(/^(\s*)(\d+)[.)]\s+(.*)$/);
      if (numbered) {
        const prefix = palette.accent(`${numbered[2] ?? ''}.`) + ' ';
        const prefixW = visibleWidth(prefix);
        wrap(numbered[3] ?? '', Math.max(1, w - prefixW)).forEach((piece, i) => {
          out.push((i === 0 ? prefix : ' '.repeat(prefixW)) + inlineSpans(piece));
        });
        continue;
      }

      const quote = line.match(/^>\s?(.*)$/);
      if (quote) {
        const prefix = palette.thinking(glyph.vertical) + ' ';
        const prefixW = visibleWidth(prefix);
        wrap(quote[1] ?? '', Math.max(1, w - prefixW)).forEach((piece) => {
          out.push(prefix + palette.thinking(inlineSpans(piece)));
        });
        continue;
      }

      for (const piece of wrap(line, w)) out.push(inlineSpans(piece));
    }

    if (inFence) flushFence(out, fenceLines, fenceLang, w);
    return out.join('\n');
  } catch {
    // Layout is best-effort; malformed input must never take the CLI down.
    return text;
  }
}
