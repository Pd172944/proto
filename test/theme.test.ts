/**
 * Tests for the terminal theme: colour gating, ANSI-aware measurement, box
 * alignment, the LCS diff, wrapping, markdown-lite, the spinner, code-block
 * capping and the approval panel.
 *
 * The colour tests deliberately drive the module-level flag rather than
 * spawning a TTY: every renderer must degrade to plain text the moment colour
 * is off, and that is the property the CLI depends on when output is piped.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import {
  DEFAULT_WIDTH,
  activityLine,
  approvalPanel,
  banner,
  box,
  chip,
  clampWidth,
  codeBlock,
  colorEnabled,
  createRailWriter,
  diff,
  footerHints,
  gradient,
  glyph,
  kv,
  markdownLite,
  palette,
  rail,
  resetColorMode,
  rule,
  setColorEnabled,
  spinnerFrame,
  statusLine,
  stripAnsi,
  turnDivider,
  visibleWidth,
  welcomePanel,
  wordmark,
  wrap,
} from '../src/tui/theme.ts';

/** Strip trailing spaces and assert every line has the same visible width. */
function assertUniformBox(out: string): void {
  const lines = out.split('\n');
  assert.ok(lines.length >= 2, 'a box has at least a top and bottom border');
  const widths = lines.map((l) => visibleWidth(l.replace(/ +$/, '')));
  const expected = widths[0];
  assert.equal(typeof expected, 'number');
  widths.forEach((w, i) => {
    assert.equal(w, expected, `line ${i} width ${w} != ${String(expected)}\n${out}`);
  });
}

/** Run `fn` with colour forced off, restoring the previous mode afterwards. */
function withColorOff(fn: () => void): void {
  setColorEnabled(false);
  try {
    fn();
  } finally {
    resetColorMode();
  }
}

/**
 * Pin `process.stdout.columns` for the duration of `fn`.
 *
 * `wordmark` is the one primitive sized by the terminal rather than by an
 * argument, so its narrow-terminal fallback can only be tested by shadowing the
 * column count. On a non-TTY `columns` is absent, so the own property is removed
 * again afterwards to restore the original prototype getter (or absence).
 */
function withColumns(cols: number, fn: () => void): void {
  const own = Object.getOwnPropertyDescriptor(process.stdout, 'columns');
  Object.defineProperty(process.stdout, 'columns', { value: cols, configurable: true, writable: true });
  try {
    fn();
  } finally {
    if (own !== undefined) Object.defineProperty(process.stdout, 'columns', own);
    else Reflect.deleteProperty(process.stdout, 'columns');
  }
}

/**
 * SGR parameter bodies in a string, excluding the bare reset.
 *
 * Colour on a 256-colour terminal lands as `38;5;N` and on a 16-colour terminal
 * as a bare code, so tests that count "distinct colours" read the parameters
 * rather than assuming a tier.
 */
function sgrCodes(s: string): string[] {
  return [...s.matchAll(/\u001b\[([0-9;]*)m/g)]
    .map((m) => m[1] ?? '')
    .filter((code) => code !== '' && code !== '0');
}

describe('colour gating', () => {
  it('emits zero ANSI from every renderer when colour is disabled', () => {
    withColorOff(() => {
      assert.equal(colorEnabled(), false);

      const outputs: Array<[string, string]> = [
        ['primary', palette.primary('x')],
        ['accent', palette.accent('x')],
        ['success', palette.success('x')],
        ['warn', palette.warn('x')],
        ['error', palette.error('x')],
        ['text', palette.text('x')],
        ['dim', palette.dim('x')],
        ['border', palette.border('x')],
        ['thinking', palette.thinking('x')],
        ['bold', palette.bold('x')],
        ['box', box('title', 'body\nsecond line', { subtitle: 'sub' })],
        ['banner', banner(['head', 'sub line'])],
        ['rule', rule('label')],
        ['kv', kv([{ key: 'model', value: 'local', note: 'tiny' }])],
        ['statusLine', statusLine([{ key: 'a', value: '1' }, { key: 'b', value: '2' }])],
        [
          'approvalPanel',
          approvalPanel({ title: 'Write', detail: 'src/a.ts', kind: 'write', warning: 'overwrites 3 lines' }),
        ],
        ['wrap', wrap('a paragraph that is long enough to wrap across lines', 20).join('\n')],
        ['codeBlock', codeBlock('const x = 1 // hi', 'ts', { maxLines: 3 })],
        ['diff', diff('a\nb\nc\n', 'a\nB\nc\n', 'src/a.ts')],
        ['markdownLite', markdownLite('# H\n**b** and `c`\n- item\n```ts\nconst a = 1\n```', 60)],
        ['spinnerFrame', spinnerFrame(3)],
      ];

      for (const [name, out] of outputs) {
        assert.equal(stripAnsi(out), out, `${name} leaked ANSI when colour is off`);
        assert.doesNotMatch(out, /\u001b\[/, `${name} emitted an escape when colour is off`);
      }
    });
  });

  it('falls back to ASCII box drawing when colour is disabled', () => {
    withColorOff(() => {
      const out = box('t', 'body');
      assert.ok(out.includes('+'), 'ASCII corner expected');
      assert.ok(out.includes('|'), 'ASCII vertical expected');
      assert.ok(!out.includes(glyph.cornerTL), 'rounded corner must not appear');
      assert.ok(!out.includes(glyph.vertical), 'unicode vertical must not appear');
    });
  });

  it('honours NO_COLOR in automatic mode', () => {
    const saved = process.env['NO_COLOR'];
    try {
      process.env['NO_COLOR'] = '1';
      resetColorMode();
      assert.equal(colorEnabled(), false);
      assert.equal(palette.primary('x'), 'x');
      assert.equal(stripAnsi(box('t', 'b')), box('t', 'b'));
    } finally {
      if (saved === undefined) delete process.env['NO_COLOR'];
      else process.env['NO_COLOR'] = saved;
      resetColorMode();
    }
  });

  it('lets an explicit setColorEnabled(true) override the ambient environment', () => {
    const saved = process.env['NO_COLOR'];
    try {
      process.env['NO_COLOR'] = '1'; // as set by many CI and harness environments
      setColorEnabled(true);
      assert.equal(colorEnabled(), true);
      assert.match(box('t', 'b'), /\u001b\[/);
    } finally {
      if (saved === undefined) delete process.env['NO_COLOR'];
      else process.env['NO_COLOR'] = saved;
      resetColorMode();
    }
  });

  it('produces ANSI colour when explicitly enabled', () => {
    setColorEnabled(true);
    try {
      const out = box('title', `body ${palette.error('bad')}`, { subtitle: 'tool' });
      assert.match(out, /\u001b\[/);
      assert.ok(out.includes(glyph.cornerTL), 'rounded corners when coloured');
      assert.ok(stripAnsi(out).includes('body bad'));
    } finally {
      resetColorMode();
    }
  });
});

describe('visibleWidth', () => {
  it('measures plain text', () => {
    assert.equal(visibleWidth(''), 0);
    assert.equal(visibleWidth('abc'), 3);
    assert.equal(visibleWidth('a b'), 3);
  });

  it('ignores ANSI escapes', () => {
    assert.equal(visibleWidth('\u001b[38;5;179mabc\u001b[0m'), 3);
    assert.equal(visibleWidth('\u001b[1m\u001b[31mhi\u001b[0m\u001b[0m'), 2);
    assert.equal(visibleWidth('a\u001b[2mb\u001b[0m c'), 4);
  });

  it('counts wide CJK characters and emoji as width 2', () => {
    assert.equal(visibleWidth('\u4e2d'), 2); // 中
    assert.equal(visibleWidth('\u4e2da'), 3);
    assert.equal(visibleWidth('\ud83d\ude00'), 2); // 😀
    assert.equal(visibleWidth('\u4e2da\ud83d\ude00'), 5);
    assert.equal(visibleWidth('\ud83d\udc69\u200d\ud83d\udcbb'), 2); // 👩‍💻 ZWJ sequence
  });
});

describe('box', () => {
  it('keeps every border line the same width for several body shapes', () => {
    withColorOff(() => {
      assertUniformBox(box('short', 'one line'));
      assertUniformBox(box('wrap', 'a fairly long body line that must wrap across the interior of the box cleanly'));
      assertUniformBox(box('empty', ''));
      assertUniformBox(box('multi', 'first\nsecond\nthird', { width: 50, padding: 2 }));
    });

    setColorEnabled(true);
    try {
      const colouredBody = `before ${palette.error('red')} and ${palette.accent('teal')} after`;
      assertUniformBox(box('coloured', colouredBody));
      assertUniformBox(box('wide', '\u4e2d\u6587\u5b57\u7b26\u4e32 mixed with ascii and \ud83d\ude00 emoji'));
    } finally {
      resetColorMode();
    }
  });

  it('aligns borders when a subtitle is present', () => {
    const out = box('read_file', 'src/app/main.ts  (128 lines)', { subtitle: 'tool' });
    assert.ok(stripAnsi(out).includes('tool'), 'subtitle is rendered');
    assertUniformBox(out);

    const longSub = box('t', 'b', { subtitle: 'a subtitle long enough to be truncated by the border' });
    assertUniformBox(longSub);

    const emptySub = box('t', 'b', { subtitle: '' });
    assertUniformBox(emptySub);
  });

  it('clamps width into [40, 200] and defaults to DEFAULT_WIDTH', () => {
    assert.equal(DEFAULT_WIDTH, 80);
    assert.equal(clampWidth(undefined), 80);
    assert.equal(clampWidth(10), 40);
    assert.equal(clampWidth(5000), 200);
    assert.equal(visibleWidth(box('t', 'b', { width: 10 }).split('\n')[0] ?? ''), 40);
  });
});

describe('banner and rule', () => {
  it('rule honours the requested width', () => {
    assert.equal(visibleWidth(rule(undefined, 60)), 60);
    assert.equal(visibleWidth(rule('label', 60)), 60);
    assert.ok(stripAnsi(rule('label', 60)).includes('label'));
  });

  it('banner keeps its title and rule inside the width', () => {
    withColorOff(() => {
      const out = banner(['proto harness v0.1.0', 'local: qwen2.5'], 60);
      const lines = out.split('\n');
      assert.ok(stripAnsi(lines[0] ?? '').includes('proto harness'));
      assert.ok(visibleWidth(lines[1] ?? '') <= 60);
      assert.ok(out.includes('\u2500'), 'thin rule present');
    });
  });
});

describe('diff', () => {
  const base = Array.from({ length: 20 }, (_v, i) => `line ${i + 1}`).join('\n') + '\n';

  it('finds a single changed line in a 20-line file', () => {
    withColorOff(() => {
      const after = base.replace('line 10', 'line ten');
      const out = diff(base, after, 'src/a.ts');
      const lines = out.split('\n');
      assert.ok(lines.includes('-line 10'), 'deletion line present');
      assert.ok(lines.includes('+line ten'), 'insertion line present');
      assert.ok(lines.some((l) => l.startsWith('@@')), 'hunk header present');
      assert.ok(!lines.includes('-line 9'), 'unchanged line not removed');
    });
  });

  it('detects an insertion', () => {
    withColorOff(() => {
      const after = base.replace('line 5\n', 'line 5\ninserted line\n');
      const out = diff(base, after, 'src/a.ts');
      assert.ok(out.split('\n').includes('+inserted line'));
    });
  });

  it('detects a deletion', () => {
    withColorOff(() => {
      const after = base.replace('line 7\n', '');
      const out = diff(base, after, 'src/a.ts');
      assert.ok(out.split('\n').includes('-line 7'));
      assert.ok(!out.split('\n').includes('+line 7'));
    });
  });

  it('reports no changes for identical input without -/+ lines', () => {
    withColorOff(() => {
      const out = diff(base, base, 'src/a.ts');
      assert.match(out, /no changes/);
      for (const line of out.split('\n')) {
        assert.ok(!line.startsWith('-'), `unexpected deletion: ${line}`);
        assert.ok(!line.startsWith('+'), `unexpected insertion: ${line}`);
      }
    });
  });

  it('handles empty inputs', () => {
    withColorOff(() => {
      assert.match(diff('', '', 'empty.ts'), /no changes/);
      assert.match(diff('', 'a\n', 'new.ts'), /\+a/);
      assert.match(diff('a\n', '', 'gone.ts'), /-a/);
    });
  });

  it('does not throw on large inputs and says it fell back', () => {
    withColorOff(() => {
      const big = Array.from({ length: 2500 }, (_v, i) => `line ${i}`).join('\n');
      const changed = big.replace('line 1200', 'line changed');
      let out = '';
      assert.doesNotThrow(() => {
        out = diff(big, changed, 'huge.ts');
      });
      const header = out.split('\n')[0] ?? '';
      assert.match(header, /fallback/);
      assert.ok(out.split('\n').some((l) => l === '+line changed'));
    });
  });

  it('truncates over maxLines with an explicit note', () => {
    withColorOff(() => {
      const before = Array.from({ length: 60 }, (_v, i) => `old ${i}`).join('\n');
      const after = Array.from({ length: 60 }, (_v, i) => `new ${i}`).join('\n');
      const out = diff(before, after, 'big.ts', { maxLines: 10, context: 0 });
      assert.match(out, /more lines \(truncated\)/);
      assert.ok(out.split('\n').length <= 11);
    });
  });

  it('colours removals and additions distinctly when enabled', () => {
    setColorEnabled(true);
    try {
      const out = diff('a\nb\n', 'a\nc\n', 'x.ts', { context: 0 });
      assert.ok(out.includes(palette.error('-b')));
      assert.ok(out.includes(palette.success('+c')));
    } finally {
      resetColorMode();
    }
  });
});

describe('wrap', () => {
  it('never exceeds the requested width, even with a long token', () => {
    const text = 'short words and averyveryverylongtokenthatkeepsgoingandgoingandgoing beyond the end';
    for (const width of [10, 20, 33, 80]) {
      for (const line of wrap(text, width)) {
        assert.ok(visibleWidth(line) <= width, `"${line}" is wider than ${width}`);
      }
    }
  });

  it('breaks a long token but keeps normal words intact', () => {
    assert.deepEqual(wrap('alpha beta gamma', 20), ['alpha beta gamma']);
    assert.deepEqual(wrap('alpha beta gamma', 11), ['alpha beta', 'gamma']);
    const hard = wrap('x'.repeat(25), 10);
    assert.deepEqual(hard, ['x'.repeat(10), 'x'.repeat(10), 'x'.repeat(5)]);
  });

  it('does not corrupt ANSI escapes', () => {
    setColorEnabled(true);
    try {
      const coloured = `\u001b[31m${'word '.repeat(8).trim()}\u001b[0m`;
      const lines = wrap(coloured, 12);
      const joined = lines.join('\n');
      assert.ok(joined.includes('\u001b[31m'), 'opening escape survived');
      assert.ok(joined.includes('\u001b[0m'), 'closing escape survived');
      for (const line of lines) {
        assert.ok(visibleWidth(line) <= 12, `coloured line too wide: ${JSON.stringify(line)}`);
      }
      assert.ok(stripAnsi(joined).startsWith('word word'));
    } finally {
      resetColorMode();
    }
  });

  it('preserves explicit newlines and empty input', () => {
    assert.deepEqual(wrap('', 20), ['']);
    assert.deepEqual(wrap('a\n\nb', 20), ['a', '', 'b']);
  });
});

describe('markdownLite', () => {
  it('renders headings, bold, inline code, bullets, numbers and fences', () => {
    setColorEnabled(true);
    try {
      const src = [
        '## Summary',
        'Changed **the parser** to use `tokens`.',
        '- one',
        '* two',
        '1. first',
        '',
        '```ts',
        'const answer = 42 // comment',
        '```',
      ].join('\n');
      const out = markdownLite(src, 60);
      const visible = stripAnsi(out);
      assert.ok(visible.includes('Summary'));
      assert.ok(visible.includes('the parser'));
      assert.ok(!out.includes('**'), 'bold markers consumed');
      assert.ok(out.includes('\u001b[1m'), 'bold escape emitted');
      assert.ok(visible.includes('tokens'), 'inline code text kept');
      assert.ok(visible.includes(glyph.bullet + ' one'));
      assert.ok(visible.includes('one') && visible.includes('two'));
      assert.ok(visible.includes('first'));
      assert.ok(visible.includes('const answer = 42'), 'fenced code passed to codeBlock');
      for (const line of out.split('\n')) assert.ok(visibleWidth(line) <= 60);
    } finally {
      resetColorMode();
    }
  });

  it('passes unknown syntax through untouched', () => {
    withColorOff(() => {
      assert.equal(markdownLite('just a plain sentence', 60), 'just a plain sentence');
    });
  });

  it('never throws on malformed input', () => {
    withColorOff(() => {
      for (const bad of ['**unclosed', '`unclosed', '```unclosed', '#', '-', '1.', '***', '> quote', '']) {
        assert.doesNotThrow(() => markdownLite(bad, 60), `threw on: ${bad}`);
        assert.equal(typeof markdownLite(bad, 60), 'string');
      }
    });
  });

  it('does not throw when colour is enabled either', () => {
    setColorEnabled(true);
    try {
      assert.equal(typeof markdownLite('**unclosed and `partial', 60), 'string');
    } finally {
      resetColorMode();
    }
  });
});

describe('spinnerFrame', () => {
  it('is stable for negative and huge indices', () => {
    assert.equal(spinnerFrame(-1), spinnerFrame(9));
    assert.equal(spinnerFrame(-11), spinnerFrame(9));
    assert.equal(spinnerFrame(1000), spinnerFrame(0));
    assert.equal(spinnerFrame(999999), spinnerFrame(9));
    assert.equal(spinnerFrame(3.7), spinnerFrame(3));
    assert.equal(spinnerFrame(-3.7), spinnerFrame(6));
  });

  it('returns 10 distinct single-column frames', () => {
    const frames = new Set<string>();
    for (let i = 0; i < 10; i++) {
      const f = spinnerFrame(i);
      assert.equal(visibleWidth(f), 1);
      frames.add(f);
    }
    assert.equal(frames.size, 10);
  });
});

describe('codeBlock', () => {
  it('caps at maxLines and reports how many lines were dropped', () => {
    withColorOff(() => {
      const code = Array.from({ length: 10 }, (_v, i) => `const line${i} = ${i}`).join('\n');
      const out = codeBlock(code, 'ts', { maxLines: 3 });
      const lines = out.split('\n');
      assert.equal(lines.length, 4, out);
      assert.ok(out.includes('7 more lines'), out);
      assert.ok(out.includes('line0') && out.includes('line2'));
      assert.ok(!out.includes('line3'), 'capped lines are not rendered');
    });
  });

  it('does not cap short blocks and keeps the language label', () => {
    withColorOff(() => {
      const out = codeBlock('const a = 1\nconst b = 2', 'ts');
      assert.ok(out.includes('ts'));
      assert.ok(!out.includes('more lines'));
      assert.equal(out.split('\n').length, 2);
    });
  });

  it('leaves unknown languages unhighlighted but still renders them', () => {
    setColorEnabled(true);
    try {
      const out = codeBlock('some **plain** text', 'brainfuck');
      assert.ok(out.includes('**plain**'), 'unknown language must not be rewritten');
      assert.ok(stripAnsi(out).includes('some **plain** text'));
    } finally {
      resetColorMode();
    }
  });

  it('highlights a known language without dropping text', () => {
    setColorEnabled(true);
    try {
      const out = codeBlock('const msg = "hi" // note', 'ts');
      assert.ok(out.includes(palette.accent('const')));
      assert.ok(out.includes(palette.success('"hi"')));
      assert.ok(out.includes(palette.dim('// note')));
      assert.ok(stripAnsi(out).includes('const msg = "hi" // note'));
    } finally {
      resetColorMode();
    }
  });
});

describe('kv and statusLine', () => {
  it('aligns values and renders notes', () => {
    withColorOff(() => {
      const out = kv([
        { key: 'a', value: '1' },
        { key: 'longer', value: '2', note: 'note' },
      ]);
      const lines = out.split('\n');
      const first = lines[0] ?? '';
      const second = lines[1] ?? '';
      assert.equal(lines.length, 2);
      assert.ok(first.startsWith('a'));
      assert.ok(second.startsWith('longer'));
      assert.ok(second.includes('(note)'));
      // Values start in the same column even though one key fills the column.
      assert.equal(first.indexOf('1'), second.indexOf('2'));
    });
  });

  it('renders a status strip and truncates when narrow', () => {
    withColorOff(() => {
      const out = statusLine([
        { key: 'model', value: 'ornith-1.5:9b' },
        { key: 'tokens', value: '1,204' },
        { key: 'cost', value: '$0.0031' },
      ]);
      assert.ok(out.includes('model'));
      assert.ok(out.includes('$0.0031'));
      assert.ok(visibleWidth(out) <= 80);

      const narrow = statusLine(
        [
          { key: 'model', value: 'a-very-long-model-name-here' },
          { key: 'tokens', value: '123456' },
        ],
        40,
      );
      assert.ok(visibleWidth(narrow) <= 40);
    });
  });
});

describe('approvalPanel', () => {
  it('names the title, kind and warning without colour', () => {
    withColorOff(() => {
      const out = approvalPanel(
        { title: 'Write src/app/main.ts', detail: 'Replaces 12 lines', kind: 'write', warning: 'this overwrites existing code' },
        70,
      );
      assert.equal(stripAnsi(out), out);
      assert.doesNotMatch(out, /\u001b\[/);
      assert.ok(out.includes('Write src/app/main.ts'), 'title present');
      assert.ok(out.includes('write'), 'kind present');
      assert.ok(out.includes('this overwrites existing code'), 'warning present');
      assertUniformBox(out);
    });
  });

  it('renders each kind and omits the warning when absent', () => {
    withColorOff(() => {
      for (const kind of ['read', 'write', 'exec'] as const) {
        const out = approvalPanel({ title: 'Tool call', detail: 'detail text', kind }, 70);
        assert.ok(out.includes(kind), `${kind} present`);
        assertUniformBox(out);
      }
      const noWarning = approvalPanel({ title: 'Read file', detail: 'src/a.ts', kind: 'read' }, 70);
      assert.ok(!stripAnsi(noWarning).includes('overwrites'));
    });
  });
});

describe('gradient', () => {
  it('ramps through distinct colours and preserves the visible text', () => {
    setColorEnabled(true);
    try {
      const out = gradient('ember to deep water', 215, 79);
      assert.equal(stripAnsi(out), 'ember to deep water');
      assert.ok(new Set(sgrCodes(out)).size >= 4, `expected a ramp: ${JSON.stringify(out)}`);
    } finally {
      resetColorMode();
    }
  });

  it('returns the input exactly when colour is off', () => {
    withColorOff(() => {
      assert.equal(gradient('plain text', 79, 215), 'plain text');
      assert.equal(gradient('', 79, 215), '');
      assert.equal(gradient('same', 79, 79), 'same');
      assert.doesNotMatch(gradient('plain text', 79, 215), /\u001b/);
    });
  });

  it('resets at a newline but carries the ramp across it', () => {
    setColorEnabled(true);
    try {
      const out = gradient('ab\ncd', 79, 215);
      assert.equal(stripAnsi(out), 'ab\ncd');
      assert.match(out, /\u001b\[0m\n/);
      const codes = sgrCodes(out);
      assert.equal(codes.length, 4, out);
      assert.notEqual(codes[0], codes[codes.length - 1]);
    } finally {
      resetColorMode();
    }
  });

  it('treats the endpoint order as meaningful', () => {
    setColorEnabled(true);
    try {
      assert.notDeepEqual(sgrCodes(gradient('abcd', 79, 215)), sgrCodes(gradient('abcd', 215, 79)));
    } finally {
      resetColorMode();
    }
  });

  it('advances a wide character by two ramp positions', () => {
    setColorEnabled(true);
    try {
      const short = sgrCodes(gradient('ab', 0, 255));
      const wide = sgrCodes(gradient('a中b', 0, 255));
      // '中' occupies columns 1-2, so 'b' sits at column 3 of 4 either way.
      assert.deepEqual(wide[2], short[1]);
    } finally {
      resetColorMode();
    }
  });
});

describe('wordmark', () => {
  it('draws five equal-width block rows under 40 columns', () => {
    setColorEnabled(true);
    try {
      withColumns(120, () => {
        const art = wordmark();
        const lines = art.split('\n');
        assert.equal(lines.length, 5, art);
        const widths = lines.map((line) => visibleWidth(line));
        assert.equal(new Set(widths).size, 1, `uneven art rows: ${JSON.stringify(widths)}`);
        assert.ok((widths[0] ?? 0) < 40, `art too wide: ${String(widths[0])}`);
        assert.doesNotMatch(stripAnsi(art), /[A-Za-z]/, 'coloured wordmark must be art, not text');
      });
    } finally {
      resetColorMode();
    }
  });

  it('falls back to plain title-cased text without colour', () => {
    withColorOff(() => {
      const out = wordmark();
      assert.equal(out, 'Proto');
      assert.doesNotMatch(out, /\u001b/);
      assert.ok(!out.includes('#'), 'no block glyphs without colour');
    });
  });

  it('falls back when the terminal is narrower than the art plus padding', () => {
    setColorEnabled(true);
    try {
      withColumns(20, () => {
        const out = wordmark();
        assert.equal(stripAnsi(out), 'Proto');
        assert.ok(!out.includes('#'));
      });
    } finally {
      resetColorMode();
    }
  });
});

describe('welcomePanel', () => {
  it('renders the wordmark, subtitle and every key inside a uniform frame', () => {
    setColorEnabled(true);
    try {
      withColumns(120, () => {
        const out = welcomePanel(
          [
            { key: 'model', value: 'anthropic · claude-sonnet-4-5', note: 'cloud' },
            { key: 'git', value: 'main', note: 'clean' },
          ],
          { subtitle: 'a coding agent that reads, edits and runs commands', width: 92 },
        );
        const visible = stripAnsi(out);
        for (const needle of ['model', 'anthropic · claude-sonnet-4-5', 'cloud', 'git', 'main', 'clean']) {
          assert.ok(visible.includes(needle), `missing ${needle}\n${visible}`);
        }
        assert.ok(visible.includes('a coding agent that reads, edits and runs commands'));
        assertUniformBox(out);
      });
    } finally {
      resetColorMode();
    }
  });

  it('is fully plain text when colour is off', () => {
    withColorOff(() => {
      const out = welcomePanel([{ key: 'model', value: 'local', note: 'tiny' }], {
        subtitle: 'a coding agent',
        width: 70,
      });
      assert.equal(stripAnsi(out), out);
      assert.doesNotMatch(out, /\u001b\[/);
      assert.doesNotMatch(out, /[\u2500-\u257f\u2800-\u28ff]/, 'no box drawing or braille without colour');
      assert.ok(out.includes('Proto'), out);
      assert.ok(out.includes('a coding agent'), out);
      assertUniformBox(out);
    });
  });
});

describe('rail', () => {
  const cases: Array<[string, string]> = [
    ['short', 'I will read the file first.'],
    ['long', 'I will read the file first, then make a single edit that adds the missing loop bound.'],
    ['empty', ''],
    ['blank line', 'first line\n\nsecond line'],
    ['wide chars', '\u4f60\u597d \ud83d\ude00 \u4e16\u754c mixed with ascii'],
  ];

  it('gives every line of the block the same visible width', () => {
    withColorOff(() => {
      for (const [name, text] of cases) {
        const out = rail(text, { width: 40 });
        const widths = out.split('\n').map((line) => visibleWidth(line));
        assert.equal(new Set(widths).size, 1, `${name} produced uneven widths\n${out}`);
        assert.equal(widths[0], 40, name);
      }
    });
  });

  it('wraps text wider than the rail budget and keeps it uniform', () => {
    withColorOff(() => {
      const out = rail('word '.repeat(40).trim(), { width: 40 });
      assert.ok(out.split('\n').length > 4, out);
      assert.equal(new Set(out.split('\n').map((line) => visibleWidth(line))).size, 1, out);
      assert.ok(out.split('\n').every((line) => line.startsWith('| ')), out);
    });
  });

  it('takes its colour from the accent option', () => {
    setColorEnabled(true);
    try {
      assert.notEqual(
        rail('body', { width: 40, accent: 'error' }),
        rail('body', { width: 40, accent: 'success' }),
      );
    } finally {
      resetColorMode();
    }
  });

  it('uses an ASCII rail without colour and honours a custom marker', () => {
    withColorOff(() => {
      const out = rail('hello', { width: 20 });
      assert.ok(out.startsWith('| '), out);
      assert.equal(stripAnsi(out), out);
    });
    setColorEnabled(true);
    try {
      assert.ok(stripAnsi(rail('hello', { width: 20, marker: '>>' })).startsWith('>> '));
    } finally {
      resetColorMode();
    }
  });
});

describe('chip', () => {
  it('never exceeds the requested width and always keeps the label', () => {
    withColorOff(() => {
      for (let width = 30; width <= 100; width++) {
        const out = chip('ok', 'read_file', 'src/app.ts  12ms  (128 lines)', { width });
        assert.ok(visibleWidth(out) <= width, `width ${width} overflowed: ${out}`);
        assert.ok(stripAnsi(out).includes('read_file'), `width ${width} lost the label`);
      }
    });
  });

  it('right-aligns the detail, truncating it rather than dropping it', () => {
    withColorOff(() => {
      const wide = chip('ok', 'read_file', 'src/app.ts  12ms', { width: 80 });
      assert.ok(wide.includes('src/app.ts'), wide);
      assert.ok(wide.includes('12ms'), wide);
      assert.equal(visibleWidth(wide), 80);

      // A chip that says only `read_file` hides the one thing the reader wants — what
      // came back, or why it failed — so the detail is ellipsised, never silently lost.
      const narrow = chip('ok', 'read_file', 'src/app.ts  12ms', { width: 24 });
      assert.ok(narrow.includes('read_file'), narrow);
      assert.match(narrow, /src|…/, `detail should survive as a truncation: ${narrow}`);
      assert.ok(visibleWidth(narrow) <= 24, String(visibleWidth(narrow)));
    });
  });

  it('renders each state and stays plain ASCII without colour', () => {
    withColorOff(() => {
      for (const state of ['ok', 'fail', 'run'] as const) {
        const out = chip(state, 'edit_file', 'anchor not unique', { width: 60 });
        assert.equal(stripAnsi(out), out);
        assert.doesNotMatch(out, /\u001b/);
        assert.doesNotMatch(out, /[\u2800-\u28ff\u2500-\u257f]/, `${state} leaked a non-ASCII glyph`);
      }
    });
    setColorEnabled(true);
    try {
      assert.ok(chip('ok', 'a', undefined, { width: 40 }).includes(glyph.check));
      assert.ok(chip('fail', 'a', undefined, { width: 40 }).includes(glyph.cross));
    } finally {
      resetColorMode();
    }
  });
});

describe('activityLine', () => {
  const base = { frame: 3, label: 'thinking', elapsedMs: 4200, tokensOut: 412, costUsd: 0.0031 };

  it('keeps the spinner and label at every width from 20 to 120', () => {
    setColorEnabled(true);
    try {
      for (let width = 20; width <= 120; width++) {
        const out = activityLine({ ...base, width });
        assert.ok(out.includes(spinnerFrame(3)), `width ${width} lost the spinner: ${out}`);
        assert.ok(out.includes('thinking'), `width ${width} lost the label: ${out}`);
        assert.ok(visibleWidth(out) <= width, `width ${width} overflowed: ${out}`);
      }
    } finally {
      resetColorMode();
    }
  });

  it('drops cost, then tokens, then elapsed as width shrinks', () => {
    withColorOff(() => {
      const all = activityLine({ ...base, width: 100 });
      assert.ok(all.includes('$0.0031'), all);
      assert.ok(all.includes('412 tok'), all);
      assert.ok(all.includes('0:04'), all);

      const noCost = activityLine({ ...base, width: 30 });
      assert.ok(!noCost.includes('$'), `cost should go first: ${noCost}`);
      assert.ok(noCost.includes('412 tok') && noCost.includes('0:04'), noCost);

      const noTokens = activityLine({ ...base, width: 20 });
      assert.ok(!noTokens.includes('tok'), `tokens should go second: ${noTokens}`);
      assert.ok(noTokens.includes('0:04'), noTokens);

      const coreOnly = activityLine({ ...base, width: 12 });
      assert.ok(!coreOnly.includes('0:04'), `elapsed should go last: ${coreOnly}`);
      assert.ok(coreOnly.includes('thinking'), coreOnly);
    });
  });

  it('formats elapsed as m:ss and uses an ASCII spinner without colour', () => {
    withColorOff(() => {
      const out = activityLine({ frame: 1, label: 'thinking', elapsedMs: 65000, width: 60 });
      assert.ok(out.includes('1:05'), out);
      assert.doesNotMatch(out, /\u001b/);
      assert.doesNotMatch(out, /[\u2800-\u28ff]/, 'braille must not survive without colour');
      assert.doesNotMatch(out, /\u00b7/, 'middle dot must not survive without colour');
    });
  });
});

describe('footerHints', () => {
  const items = [
    { key: '\u23ce', label: 'send' },
    { key: '^C', label: 'stop turn' },
    { key: '/help', label: 'commands' },
  ];

  it('lists every hint at a wide width', () => {
    setColorEnabled(true);
    try {
      const out = footerHints(items, 120);
      const visible = stripAnsi(out);
      for (const item of items) {
        assert.ok(visible.includes(item.key), `missing key ${item.key}: ${visible}`);
        assert.ok(visible.includes(item.label), `missing label ${item.label}: ${visible}`);
      }
      assert.ok(visible.includes(glyph.dot), 'coloured hints join with a middle dot');
      assert.ok(visibleWidth(out) <= 120);
    } finally {
      resetColorMode();
    }
  });

  it('truncates whole hints rather than cutting one in half', () => {
    withColorOff(() => {
      const one = footerHints(items, 20);
      assert.ok(one.includes('send'), one);
      assert.ok(!one.includes('commands'), one);
      assert.ok(!one.includes('stop'), `whole hints only: ${one}`);
      assert.ok(visibleWidth(one) <= 20, one);

      const two = footerHints(items, 30);
      assert.ok(two.includes(' | '), `plain hints use an ASCII separator: ${two}`);
      assert.ok(!two.includes('commands'), `whole hints only: ${two}`);
      assert.ok(visibleWidth(two) <= 30, two);
    });
  });

  it('contains no ANSI when colour is off', () => {
    withColorOff(() => {
      const out = footerHints(items, 100);
      assert.equal(stripAnsi(out), out);
      assert.doesNotMatch(out, /\u001b/);
    });
  });
});

describe('turnDivider', () => {
  it('centres a label in a full-width rule', () => {
    setColorEnabled(true);
    try {
      const out = turnDivider('turn 1', 60);
      assert.ok(stripAnsi(out).includes('turn 1'), out);
      assert.equal(visibleWidth(out), 60);
    } finally {
      resetColorMode();
    }
  });

  it('renders an unlabelled rule at the same width, plain without colour', () => {
    withColorOff(() => {
      const bare = turnDivider(undefined, 60);
      const labelled = turnDivider('turn 2', 60);
      assert.equal(visibleWidth(bare), 60);
      assert.equal(visibleWidth(labelled), 60);
      assert.ok(labelled.includes('turn 2'), labelled);
      assert.doesNotMatch(bare, /[^\-\n]/, `only ASCII dashes expected: ${bare}`);
      assert.doesNotMatch(labelled, /\u001b/);
    });
  });
});

describe('colour detection precedence', () => {
  /** Run `fn` with exactly the given colour env vars set. */
  function withEnv(vars: Record<string, string | undefined>, fn: () => void): void {
    const saved: Record<string, string | undefined> = {};
    for (const key of ['NO_COLOR', 'PROTO_COLOR']) {
      saved[key] = process.env[key];
      delete process.env[key];
    }
    for (const [key, value] of Object.entries(vars)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    resetColorMode();
    try {
      fn();
    } finally {
      for (const key of ['NO_COLOR', 'PROTO_COLOR']) {
        if (saved[key] === undefined) delete process.env[key];
        else process.env[key] = saved[key];
      }
      resetColorMode();
    }
  }

  it('lets PROTO_COLOR=1 override an inherited NO_COLOR', () => {
    // The escape hatch exists for piping into a pager from a shell that exports
    // NO_COLOR to every child; if NO_COLOR won, the hatch would be dead code.
    withEnv({ NO_COLOR: '1', PROTO_COLOR: '1' }, () => {
      assert.equal(colorEnabled(), true);
    });
  });

  it('still honours NO_COLOR and PROTO_COLOR on their own', () => {
    withEnv({ NO_COLOR: '1' }, () => assert.equal(colorEnabled(), false));
    withEnv({ PROTO_COLOR: '1' }, () => assert.equal(colorEnabled(), true));
    withEnv({ PROTO_COLOR: '0' }, () => assert.equal(colorEnabled(), false));
  });

  it('lets an explicit setColorEnabled call override both variables', () => {
    withEnv({ NO_COLOR: '1', PROTO_COLOR: '0' }, () => {
      setColorEnabled(true);
      try {
        assert.equal(colorEnabled(), true);
      } finally {
        resetColorMode();
      }
    });
  });
});

describe('new primitives, colour gating', () => {
  it('emits zero ANSI and no box drawing from every new renderer when colour is off', () => {
    withColorOff(() => {
      const outputs: Array<[string, string]> = [
        ['gradient', gradient('ember to deep water', 215, 79)],
        ['wordmark', wordmark()],
        ['welcomePanel', welcomePanel([{ key: 'model', value: 'local', note: 'tiny' }], { subtitle: 'sub', width: 70 })],
        ['rail', rail('a line of body text', { width: 40 })],
        ['chip', chip('ok', 'read_file', 'src/app.ts  12ms', { width: 60 })],
        ['chip fail', chip('fail', 'edit_file', 'anchor not unique', { width: 60 })],
        ['chip run', chip('run', 'exec', 'npm test', { width: 60 })],
        ['activityLine', activityLine({ frame: 4, label: 'thinking…', elapsedMs: 4200, tokensOut: 412, costUsd: 0.0031 })],
        ['footerHints', footerHints([{ key: '^C', label: 'stop' }, { key: '/help', label: 'commands' }], 60)],
        ['turnDivider', turnDivider('turn 1', 60)],
      ];

      for (const [name, out] of outputs) {
        assert.equal(stripAnsi(out), out, `${name} leaked ANSI when colour is off`);
        assert.doesNotMatch(out, /\u001b\[/, `${name} emitted an escape when colour is off`);
        assert.doesNotMatch(out, /[\u2800-\u28ff]/, `${name} emitted braille when colour is off`);
      }
      // Box drawing is allowed only from `box`'s existing ASCII frame; the new
      // furniture must be `+ - |` shaped.
      assert.doesNotMatch(outputs.map(([, out]) => out).join('\n'), /[\u2500-\u257f]/);
    });
  });
});

describe('createRailWriter', () => {
  /** Collect a rail writer's output for `chunks`, flushing at the end. */
  function run(chunks: string[], width: number, color = false): string[] {
    const lines: string[] = [];
    let buf = '';
    const sink = (t: string): void => {
      buf += t;
      let i: number;
      while ((i = buf.indexOf('\n')) >= 0) {
        lines.push(buf.slice(0, i));
        buf = buf.slice(i + 1);
      }
    };
    if (color) setColorEnabled(true);
    try {
      const w = createRailWriter(width, sink);
      for (const c of chunks) w.write(c);
      w.end();
    } finally {
      resetColorMode();
    }
    if (buf !== '') lines.push(buf);
    return lines;
  }

  it('marks every line of a single long paragraph, not just the first', () => {
    const text = 'word '.repeat(60).trim();
    const lines = run([text], 40);
    assert.ok(lines.length >= 6, `expected a wrapped paragraph, got ${lines.length} lines`);
    for (const line of lines) assert.ok(line.startsWith('| '), `unmarked line: ${JSON.stringify(line)}`);
    for (const line of lines) assert.ok(visibleWidth(line) <= 40, `too wide: ${JSON.stringify(line)}`);
  });

  it('preserves hard newlines as paragraph breaks without a dangling marker', () => {
    const lines = run(['first para\n\nsecond para\n'], 60);
    assert.deepEqual(lines, ['| first para', '', '| second para']);
  });

  it('handles a chunk boundary that splits a word, and never emits a partial line early', () => {
    // Each chunk is one character: the writer must not emit a line until it is complete.
    const lines = run([...'the quick brown fox jumps over the lazy dog again and again'], 20);
    const joined = lines.map((l) => l.replace(/^\| /, '')).join(' ');
    assert.equal(joined, 'the quick brown fox jumps over the lazy dog again and again');
    for (const line of lines) assert.ok(visibleWidth(line) <= 20);
  });

  it('hard-cuts a token that is wider than the content area', () => {
    const url = 'https://example.com/' + 'x'.repeat(120);
    const lines = run([url], 30);
    assert.ok(lines.length >= 4);
    for (const line of lines) assert.ok(visibleWidth(line) <= 30, `too wide: ${visibleWidth(line)}`);
    assert.equal(lines.join('').replace(/^\| /gm, '').replace(/\| /g, ''), url);
  });

  it('flushes a trailing line with no newline at the end', () => {
    assert.deepEqual(run(['no trailing newline'], 40), ['| no trailing newline']);
  });

  it('emits nothing for empty input', () => {
    assert.deepEqual(run([], 40), []);
    assert.deepEqual(run([''], 40), []);
  });

  it('stays inside the width when colour is on', () => {
    const lines = run(['word '.repeat(40).trim()], 40, true);
    for (const line of lines) assert.ok(visibleWidth(line) <= 40, `too wide: ${JSON.stringify(line)}`);
    assert.ok(lines.every((l) => l.includes('\u2503')), 'coloured rail should use the heavy bar');
    assert.ok(lines.every((l) => visibleWidth(stripAnsi(l)) <= 40));
  });
});
