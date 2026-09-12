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
  approvalPanel,
  banner,
  box,
  clampWidth,
  codeBlock,
  colorEnabled,
  diff,
  glyph,
  kv,
  markdownLite,
  palette,
  resetColorMode,
  rule,
  setColorEnabled,
  spinnerFrame,
  statusLine,
  stripAnsi,
  visibleWidth,
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
