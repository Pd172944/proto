/**
 * Tests for the codebase index: masking, ignore matching, the symbol cache, the
 * reference graph and the ranked map.
 *
 * The heaviest tests are on masking, because every symbol the index reports is only as
 * trustworthy as it. A masker that leaks a comment produces a phantom definition; one
 * that eats real code produces a missing definition that no later stage can recover.
 * Both failures are silent, which is what makes them worth this much attention.
 */

import { strict as assert } from 'node:assert';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { collectIdentifiers, maskCode } from '../src/index/lex.ts';
import type { LexSpec } from '../src/index/lex.ts';
import { isIgnored, parseGitignore } from '../src/index/ignore.ts';
import { buildGraph, rankFiles, referringFiles, searchSymbols, ImportResolver } from '../src/index/graph.ts';
import { renderRepoMap, renderFileOutline, estimateTokens } from '../src/index/repomap.ts';
import { cachePath, loadIndex, saveIndex, workspaceKey } from '../src/index/store.ts';
import type { FileSymbols, SymbolDef } from '../src/index/types.ts';

/* ------------------------------------------------------------------ helpers */

function file(path: string, opts: { defs?: Array<[string, number]>; refs?: string[]; imports?: string[] } = {}): FileSymbols {
  return {
    path,
    lang: 'typescript',
    bytes: 100,
    lines: 10,
    mtimeMs: 1_700_000_000_000,
    defs: (opts.defs ?? []).map(([name, line]): SymbolDef => ({ name, kind: 'function', line, signature: '()' })),
    refs: opts.refs ?? [],
    imports: opts.imports ?? [],
  };
}

function tempDir(): string {
  return mkdtempSync(join(tmpdir(), 'proto-index-test-'));
}

const TS_SPEC: LexSpec = {
  lineComments: ['//'],
  blockComments: [['/*', '*/']],
  strings: [
    { open: '`', close: '`', escape: '\\', multiline: true },
    { open: '"', close: '"', escape: '\\' },
    { open: "'", close: "'", escape: '\\' },
  ],
};

/* ------------------------------------------------------------------ masking */

describe('maskCode', () => {
  it('preserves length and newlines exactly, so line numbers survive', () => {
    const src = 'const a = 1;\n/* a block\n   spanning lines */\nconst b = "text";\n';
    const masked = maskCode(src, TS_SPEC);
    assert.equal(masked.length, src.length);
    assert.equal(masked.split('\n').length, src.split('\n').length);
    // Line 4 must still be the line that held `const b`, at the same index.
    // Only the string *contents* are blanked: the delimiters stay so that import and
    // string-shaped syntax is still visible to the extractors.
    assert.equal(masked.split('\n')[3], 'const b = "    ";');
  });

  it('blanks a line comment but keeps the code before it', () => {
    const masked = maskCode('const x = 1; // fn hidden()\n', TS_SPEC);
    assert.ok(masked.includes('const x = 1;'));
    assert.ok(!masked.includes('fn hidden'), 'comment text leaked');
  });

  it('blanks a block comment including a fake definition on its own line', () => {
    const src = '// fn a()\n/*\nfunction b() {}\n*/\nfunction real() {}\n';
    const masked = maskCode(src, TS_SPEC);
    assert.ok(!masked.includes('fn a'));
    assert.ok(!masked.includes('function b'));
    assert.ok(masked.includes('function real'));
    // The real definition is still on line 5.
    assert.equal(masked.split('\n')[4], 'function real() {}');
  });

  it('blanks string contents but keeps the delimiters and the line intact', () => {
    const masked = maskCode('const s = "function fake() {}";\n', TS_SPEC);
    assert.ok(!masked.includes('function fake'));
    assert.equal(masked.split('\n')[0]?.length, 'const s = "function fake() {}";'.length);
  });

  it('handles an escaped quote without ending the string early', () => {
    const masked = maskCode('const s = "a \\" function fake() {} b"; const t = 1;\n', TS_SPEC);
    assert.ok(!masked.includes('function fake'), 'escaped quote ended the string early');
    assert.ok(masked.includes('const t = 1;'), 'masking ran past the string');
  });

  it('keeps a multi-line template literal from leaking code, and keeps newlines', () => {
    const src = 'const q = `line one\nfunction fake() {}\nline three`;\nconst after = 1;\n';
    const masked = maskCode(src, TS_SPEC);
    assert.ok(!masked.includes('function fake'));
    assert.equal(masked.split('\n').length, src.split('\n').length);
    assert.ok(masked.includes('const after = 1;'));
  });

  it('does not throw on an unterminated block comment or string', () => {
    assert.doesNotThrow(() => maskCode('const a = 1;\n/* never closed', TS_SPEC));
    assert.doesNotThrow(() => maskCode('const a = "never closed', TS_SPEC));
    assert.doesNotThrow(() => maskCode('', TS_SPEC));
  });

  it('leaves a bare apostrophe in a comment from derailing the rest of the file', () => {
    // `// don't` used to open a string in naive maskers and blank the whole file.
    const src = "// don't do this\nconst real = 1;\n";
    const masked = maskCode(src, TS_SPEC);
    assert.ok(masked.includes('const real = 1;'), 'an apostrophe in a comment ate the file');
  });
});

describe('collectIdentifiers', () => {
  it('finds identifiers and drops keywords that carry no signal', () => {
    const ids = collectIdentifiers(maskCode('function handleLogin(user, options) { return new Session(user); }', TS_SPEC), 50);
    assert.ok(ids.includes('handleLogin'));
    assert.ok(ids.includes('Session'));
    assert.ok(ids.includes('options'));
    assert.ok(!ids.includes('function'), 'keyword leaked into references');
    assert.ok(!ids.includes('return'));
  });

  it('respects the cap', () => {
    const src = Array.from({ length: 200 }, (_, i) => `value${i}`).join(' + ');
    assert.equal(collectIdentifiers(src, 10).length, 10);
  });

  it('returns each identifier once', () => {
    assert.deepEqual(collectIdentifiers('alpha alpha alpha beta', 50), ['alpha', 'beta']);
  });
});

/* ------------------------------------------------------------------ ignore */

describe('gitignore matching', () => {
  // [pattern, path, isDir, expectIgnored, why]
  const cases: Array<[string, string, boolean, boolean, string]> = [
    ['*.log', 'app.log', false, true, 'extension match at the root'],
    ['*.log', 'logs/app.log', false, true, 'extension match at depth'],
    ['build/', 'build', true, true, 'directory-only pattern matches the directory'],
    ['build/', 'build/output.bin', false, true, 'a dir rule still covers its contents'],
    ['/dist', 'dist', true, true, 'anchored at the root'],
    ['/dist', 'src/dist', true, false, 'anchored does not match deeper'],
    ['node_modules', 'node_modules', true, true, 'bare name matches a directory'],
    ['node_modules', 'packages/a/node_modules', true, true, 'bare name matches at any depth'],
    ['**/generated', 'a/b/generated', true, true, 'double-star prefix'],
    ['docs/**', 'docs/a/b.md', false, true, 'double-star suffix'],
    ['a/**/b', 'a/x/y/b', true, true, 'double-star in the middle'],
    ['?at', 'cat', false, true, 'question mark matches one character'],
    ['?at', 'chat', false, false, 'question mark does not match two'],
    ['*.log', 'app.log', false, true, 'plain'],
  ];

  for (const [pattern, path, isDir, expectIgnored, why] of cases) {
    it(`${pattern} vs ${path} (${why})`, () => {
      const rules = parseGitignore(pattern, '');
      assert.equal(isIgnored(path, isDir, rules), expectIgnored);
    });
  }

  it('lets a later negation re-include a file', () => {
    const rules = parseGitignore('*.log\n!keep.log\n', '');
    assert.equal(isIgnored('app.log', false, rules), true);
    assert.equal(isIgnored('keep.log', false, rules), false);
  });

  it('applies rules from a nested .gitignore only below that directory', () => {
    const rules = parseGitignore('*.tmp', 'sub');
    assert.equal(isIgnored('sub/x.tmp', false, rules), true);
    assert.equal(isIgnored('other/x.tmp', false, rules), false);
  });

  it('skips comments and blank lines', () => {
    assert.deepEqual(parseGitignore('# a comment\n\n  \n', ''), []);
  });
});

/* ------------------------------------------------------------------ store */

describe('symbol cache', () => {
  it('round-trips definitions, references and imports', () => {
    const dir = tempDir();
    try {
      const root = '/tmp/some/project';
      const entries = [
        file('src/a.ts', { defs: [['alpha', 3], ['beta', 9]], refs: ['beta', 'gamma'], imports: ['./b'] }),
        file('src/b.ts', { defs: [['gamma', 1]], refs: ['alpha'], imports: [] }),
      ];
      saveIndex(dir, root, entries);
      const loaded = loadIndex(dir, root);

      assert.equal(loaded.files.size, 2);
      assert.equal(loaded.damaged, 0);
      const a = loaded.files.get('src/a.ts');
      assert.deepEqual(a?.defs.map((d) => d.name), ['alpha', 'beta']);
      assert.deepEqual(a?.defs.map((d) => d.line), [3, 9]);
      assert.deepEqual(a?.refs, ['beta', 'gamma']);
      assert.deepEqual(a?.imports, ['./b']);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('interns references so a repeated name is stored once in the dictionary', () => {
    const dir = tempDir();
    try {
      const root = '/tmp/some/project';
      saveIndex(dir, root, [
        file('a.ts', { refs: ['shared', 'only-a'] }),
        file('b.ts', { refs: ['shared', 'only-b'] }),
      ]);
      const raw = readFileSync(cachePath(dir, root), 'utf8');
      const header = JSON.parse(raw.split('\n')[0] as string) as { refs: string[] };
      assert.deepEqual(header.refs.sort(), ['only-a', 'only-b', 'shared']);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('refuses a cache built for a different root or version', () => {
    const dir = tempDir();
    try {
      saveIndex(dir, '/tmp/one', [file('a.ts', { defs: [['x', 1]] })]);
      assert.equal(loadIndex(dir, '/tmp/two').files.size, 0, 'a foreign cache must not be used');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('survives a corrupt line and keeps the rest', () => {
    const dir = tempDir();
    try {
      const root = '/tmp/some/project';
      saveIndex(dir, root, [file('a.ts', { defs: [['x', 1]] }), file('b.ts', { defs: [['y', 2]] })]);
      const path = cachePath(dir, root);
      const lines = readFileSync(path, 'utf8').split('\n');
      lines[1] = '{ this is not json';
      writeFileSync(path, lines.join('\n'));

      const loaded = loadIndex(dir, root);
      assert.equal(loaded.damaged, 1);
      assert.equal(loaded.files.size, 1, 'the intact entry should still load');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('returns empty rather than throwing when the cache is missing', () => {
    const dir = tempDir();
    try {
      const loaded = loadIndex(dir, '/tmp/nothing');
      assert.equal(loaded.files.size, 0);
      assert.equal(loaded.existed, false);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('keys the cache by workspace path', () => {
    assert.notEqual(workspaceKey('/a/b'), workspaceKey('/a/c'));
    assert.equal(workspaceKey('/a/b'), workspaceKey('/a/b'));
  });
});

/* ------------------------------------------------------------------ graph */

describe('import resolution', () => {
  const paths = ['src/app.ts', 'src/util.ts', 'src/deep/mod.ts', 'pkg/math_utils.py', 'internal/api/handler.go'];
  const r = new ImportResolver(paths);

  it('resolves a relative path with an omitted extension', () => {
    assert.equal(r.resolve('src/app.ts', './util'), 'src/util.ts');
    assert.equal(r.resolve('src/app.ts', './deep/mod'), 'src/deep/mod.ts');
  });

  it('resolves a relative path written with a .js extension to the .ts file', () => {
    assert.equal(r.resolve('src/app.ts', './util.js'), 'src/util.ts');
  });

  it('resolves a dotted Python module path', () => {
    assert.equal(r.resolve('main.py', 'pkg.math_utils'), 'pkg/math_utils.py');
  });

  it('resolves a package path by its trailing segments', () => {
    assert.equal(r.resolve('main.go', 'github.com/x/internal/api/handler'), 'internal/api/handler.go');
  });

  it('returns null rather than guessing when nothing matches', () => {
    assert.equal(r.resolve('src/app.ts', './does-not-exist'), null);
    assert.equal(r.resolve('src/app.ts', 'react'), null);
    assert.equal(r.resolve('src/app.ts', ''), null);
  });
});

describe('reference graph', () => {
  it('creates an edge from a file to the file that defines what it mentions', () => {
    const g = buildGraph(
      new Map([
        ['src/a.ts', file('src/a.ts', { defs: [['alpha', 1]], refs: ['beta'] })],
        ['src/b.ts', file('src/b.ts', { defs: [['beta', 1]], refs: [] })],
      ]),
    );
    const a = g.index.get('src/a.ts') as number;
    const b = g.index.get('src/b.ts') as number;
    assert.ok((g.out[a] ?? []).some((e) => e.to === b), 'no edge a -> b');
  });

  it('never creates a self-edge', () => {
    const g = buildGraph(new Map([['a.ts', file('a.ts', { defs: [['alpha', 1]], refs: ['alpha'] })]]));
    assert.deepEqual(g.out[0], []);
  });

  it('ignores a name defined in so many files that it carries no signal', () => {
    const entries = new Map<string, FileSymbols>();
    // 30 files define `Config`; one file references it. The guard should drop the edge.
    for (let i = 0; i < 30; i++) entries.set(`c${i}.ts`, file(`c${i}.ts`, { defs: [['Config', 1]], refs: [] }));
    entries.set('user.ts', file('user.ts', { defs: [['run', 1]], refs: ['Config'] }));
    const g = buildGraph(entries);
    const userIdx = g.index.get('user.ts') as number;
    assert.deepEqual(g.out[userIdx], [], 'a ubiquitous name should not link every file');
  });

  it('weights a resolved import above an incidental mention', () => {
    const g = buildGraph(
      new Map([
        ['a.ts', file('a.ts', { defs: [['run', 1]], refs: ['beta'], imports: ['./b'] })],
        ['b.ts', file('b.ts', { defs: [['beta', 1]], refs: [] })],
      ]),
    );
    const a = g.index.get('a.ts') as number;
    const edge = (g.out[a] ?? []).find((e) => e.to === (g.index.get('b.ts') as number));
    assert.ok(edge !== undefined && edge.w > 3, `import edge should dominate, got ${edge?.w}`);
  });
});

describe('ranking', () => {
  const entries = new Map([
    ['src/core.ts', file('src/core.ts', { defs: [['coreThing', 1], ['coreOther', 2]], refs: ['helper'] })],
    ['src/helper.ts', file('src/helper.ts', { defs: [['helper', 1]], refs: [] })],
    ['src/unrelated.ts', file('src/unrelated.ts', { defs: [['unrelated', 1]], refs: [] })],
  ]);

  it('ranks a task-named file first and marks it personal', () => {
    const g = buildGraph(entries);
    const ranked = rankFiles(g, { focus: ['src/unrelated.ts'] });
    assert.equal(ranked[0]?.path, 'src/unrelated.ts');
    assert.equal(ranked[0]?.personal, true);
  });

  it('ranks the definer of a task-named symbol first', () => {
    const g = buildGraph(entries);
    const ranked = rankFiles(g, { focusSymbols: ['helper'] });
    assert.equal(ranked[0]?.path, 'src/helper.ts');
  });

  it('changes the order when the focus changes, which is the whole point', () => {
    const g = buildGraph(entries);
    const a = rankFiles(g, { focus: ['src/helper.ts'] }).map((r) => r.path);
    const b = rankFiles(g, { focus: ['src/unrelated.ts'] }).map((r) => r.path);
    assert.notDeepEqual(a, b);
  });

  it('reaches a file through the graph even when the task does not name it', () => {
    // `run` mentions `helper`, and the focus is `core.ts`: helper should outrank the
    // file that nothing links to.
    const g = buildGraph(entries);
    const ranked = rankFiles(g, { focus: ['src/core.ts'] });
    const helperRank = ranked.findIndex((r) => r.path === 'src/helper.ts');
    const unrelatedRank = ranked.findIndex((r) => r.path === 'src/unrelated.ts');
    assert.ok(helperRank < unrelatedRank, 'graph proximity should beat no proximity');
  });

  it('omits files that define nothing', () => {
    const withEmpty = new Map(entries);
    withEmpty.set('README.md', file('README.md', {}));
    const ranked = rankFiles(buildGraph(withEmpty), {});
    assert.ok(!ranked.some((r) => r.path === 'README.md'));
  });

  it('survives an empty repository', () => {
    assert.deepEqual(rankFiles(buildGraph(new Map()), {}), []);
  });
});

describe('symbol lookup helpers', () => {
  const g = buildGraph(
    new Map([
      ['a.ts', file('a.ts', { defs: [['handler', 4], ['handleLogin', 9]], refs: [] })],
      ['b.ts', file('b.ts', { defs: [['handler', 1]], refs: [] })],
    ]),
  );

  it('finds a partial name when the exact one misses', () => {
    const hits = searchSymbols(g, 'handle', 10);
    // A name defined in two files legitimately produces two hits.
    assert.deepEqual([...new Set(hits.map((h) => h.name))].sort(), ['handleLogin', 'handler']);
  });

  it('excludes the defining files when listing references', () => {
    const files = new Map([
      ['a.ts', file('a.ts', { defs: [['handler', 1]], refs: [] })],
      ['b.ts', file('b.ts', { defs: [], refs: ['handler'] })],
    ]);
    assert.deepEqual(referringFiles(files, 'handler', new Set(['a.ts'])), ['b.ts']);
  });
});

/* ------------------------------------------------------------------ repo map */

describe('repo map rendering', () => {
  const ranked = [
    { path: 'src/core.ts', score: 1, personal: true, defs: [{ name: 'coreThing', kind: 'function' as const, line: 1, signature: '(a: string): void' }] },
    { path: 'src/helper.ts', score: 0.5, personal: false, defs: [{ name: 'helper', kind: 'function' as const, line: 3, signature: '()' }] },
    { path: 'src/other.ts', score: 0.2, personal: false, defs: [{ name: 'other', kind: 'class' as const, line: 7, signature: '' }] },
  ];

  it('includes signatures, not just names', () => {
    const map = renderRepoMap(ranked, { budgetChars: 4000 });
    assert.ok(map.text.includes('coreThing'));
    assert.ok(map.text.includes('(a: string): void'), 'signature was dropped');
  });

  it('marks the file the task named', () => {
    const map = renderRepoMap(ranked, { budgetChars: 4000 });
    assert.ok(/src\/core\.ts.*named by the task/.test(map.text), map.text);
  });

  it('stays inside the budget and says what it dropped', () => {
    // Enough files that any real budget has to cut some of them.
    const many = Array.from({ length: 60 }, (_, i) => ({
      path: `src/file${i}.ts`,
      score: 1 - i / 100,
      personal: false,
      defs: [{ name: `fn${i}`, kind: 'function' as const, line: 1, signature: '(a: string, b: number): Promise<void>' }],
    }));
    const map = renderRepoMap(many, { budgetChars: 400 });
    assert.ok(map.text.length <= 700, `map overshot badly: ${map.text.length}`);
    assert.ok(map.truncated, 'a cut map must say so');
    assert.ok(map.files.length < many.length);
    assert.ok(/not shown/.test(map.text), map.text.slice(-200));
  });

  it('reports the total file count so a cut map is not mistaken for the whole repo', () => {
    const map = renderRepoMap(ranked, { budgetChars: 120 });
    assert.equal(map.totalFiles, ranked.length);
  });

  it('keeps rank order rather than skipping ahead to smaller files', () => {
    const map = renderRepoMap(ranked, { budgetChars: 4000 });
    assert.deepEqual(map.files.map((f) => f.path), ['src/core.ts', 'src/helper.ts', 'src/other.ts']);
  });

  it('nests methods under their parent with the parent folded into the name', () => {
    const out = renderFileOutline(
      file('a.ts', { defs: [['Router', 1]] }),
    );
    assert.ok(out.includes('Router'));
    const nested = renderRepoMap(
      [
        {
          path: 'a.ts',
          score: 1,
          personal: false,
          defs: [
            { name: 'Router', kind: 'class', line: 1, signature: '' },
            { name: 'addRoute', kind: 'method', line: 4, signature: '(p: string)', parent: 'Router' },
          ],
        },
      ],
      { budgetChars: 2000 },
    );
    assert.ok(nested.text.includes('Router.addRoute'), nested.text);
  });

  it('estimates tokens from length and is honest that it is an estimate', () => {
    assert.equal(estimateTokens('abcd'), 1);
    assert.equal(estimateTokens('a'.repeat(400)), 100);
  });
});
