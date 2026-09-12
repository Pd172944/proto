/**
 * Tests for language identification and per-language symbol extraction.
 *
 * These are behavioural tests, not parser-conformance tests: they pin the handful of
 * properties the reference graph depends on. The important ones are negative — a symbol
 * inside a comment, a string, or a file that ends mid-literal must not appear, and a
 * line number after a multi-line literal must still point at the source line a human
 * would. The per-language table deliberately keeps one representative snippet per
 * language id so that adding a language without adding a snippet fails the suite.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import {
  INDEXABLE_EXTENSIONS,
  LANGUAGE_IDS,
  extract,
  isIndexable,
  languageFor,
} from '../src/index/lang.ts';
import { MAX_REFS_PER_FILE } from '../src/index/types.ts';
import type { SymbolDef } from '../src/index/types.ts';

/** A representative, minimal snippet per language id. */
const REPRESENTATIVE: Record<string, { path: string; src: string }> = {
  typescript: { path: 'src/a.ts', src: 'export function greet(name: string): string {\n  return name;\n}\n' },
  tsx: { path: 'src/Button.tsx', src: 'export function Button() {\n  return <button />;\n}\n' },
  javascript: { path: 'src/a.js', src: 'function hello() {\n  return 1;\n}\n' },
  jsx: { path: 'src/Widget.jsx', src: 'function Widget() {\n  return <div />;\n}\n' },
  python: { path: 'a.py', src: 'def greet():\n    pass\n' },
  go: { path: 'a.go', src: 'package main\n\nfunc Hello() {\n}\n' },
  rust: { path: 'a.rs', src: 'pub fn hello() {\n}\n' },
  java: { path: 'A.java', src: 'public class App {\n  public static void main(String[] args) {\n    System.out.println("hi");\n  }\n}\n' },
  c: { path: 'a.c', src: 'int main(void) {\n  return 0;\n}\n' },
  cpp: { path: 'a.cpp', src: 'class Widget {\npublic:\n  void draw();\n};\n' },
  ruby: { path: 'a.rb', src: 'def greet\nend\n' },
  php: { path: 'a.php', src: '<?php\nfunction greet() {}\n' },
  csharp: { path: 'A.cs', src: 'public class App {\n  public void Run() {}\n}\n' },
  kotlin: { path: 'A.kt', src: 'fun greet() {}\n' },
  swift: { path: 'A.swift', src: 'func greet() {}\n' },
  scala: { path: 'A.scala', src: 'def greet(): Unit = {}\n' },
  shell: { path: 'a.sh', src: 'greet() {\n  echo hi\n}\n' },
  sql: { path: 'a.sql', src: 'CREATE TABLE users (id INT);\n' },
  lua: { path: 'a.lua', src: 'function greet()\nend\n' },
  r: { path: 'a.r', src: 'greet <- function() {}\n' },
  vue: { path: 'A.vue', src: '<template>\n  <div>hi</div>\n</template>\n<script>\nexport function setup() {}\n</script>\n' },
  svelte: { path: 'A.svelte', src: '<script>\nfunction tick() {}\n</script>\n' },
  markdown: { path: 'a.md', src: '# Title\n\nBody text.\n' },
  json: { path: 'a.json', src: '{\n  "name": "x",\n  "version": "1.0"\n}\n' },
  yaml: { path: 'a.yaml', src: 'name: x\nversion: 1\n' },
  toml: { path: 'a.toml', src: 'name = "x"\nversion = "1"\n' },
};

function defsOf(path: string, src: string): SymbolDef[] {
  return extract(path, src).defs;
}

/** The def named `name`, failing with the actual names when it is missing. */
function must(list: SymbolDef[], name: string): SymbolDef {
  const d = list.find((x) => x.name === name);
  assert.ok(d, `expected a definition named "${name}"; got [${list.map((x) => x.name).join(', ')}]`);
  return d;
}

function names(list: SymbolDef[]): string[] {
  return list.map((d) => d.name);
}

describe('language identification', () => {
  it('maps every indexable extension and agrees with isIndexable', () => {
    assert.ok(INDEXABLE_EXTENSIONS.length > 0);
    for (const ext of INDEXABLE_EXTENSIONS) {
      assert.equal(ext, ext.toLowerCase(), `${ext} must be lowercase`);
      assert.ok(ext.startsWith('.'), `${ext} must start with a dot`);
      const id = languageFor(`src/file${ext}`);
      assert.ok(id !== null, `${ext} must map to a language`);
      assert.ok(LANGUAGE_IDS.includes(id), `${ext} mapped to unknown id ${id}`);
      assert.equal(isIndexable(`src/file${ext}`), true);
    }
    assert.deepEqual(INDEXABLE_EXTENSIONS, [...INDEXABLE_EXTENSIONS].sort());
    assert.deepEqual(LANGUAGE_IDS, [...LANGUAGE_IDS].sort());
    assert.equal(new Set(LANGUAGE_IDS).size, LANGUAGE_IDS.length);
  });

  it('returns null for unknown and extensionless paths', () => {
    for (const path of ['a.txt', 'Makefile', '.bashrc', 'archive.tar.gz', 'noext', 'a.JSONX']) {
      assert.equal(languageFor(path), null, path);
      assert.equal(isIndexable(path), false, path);
    }
  });

  it('is case-insensitive on the extension and handles nested paths', () => {
    assert.equal(languageFor('SRC/App.TS'), 'typescript');
    assert.equal(languageFor('src/thing.d.ts'), 'typescript');
    assert.equal(languageFor('C:\\work\\main.py'), 'python');
  });

  it('reports lang "text" for an unknown extension but still collects references', () => {
    const result = extract('notes.txt', 'alpha beta gamma');
    assert.equal(result.lang, 'text');
    assert.deepEqual(result.defs, []);
    assert.deepEqual(result.imports, []);
    assert.ok(result.refs.includes('alpha'));
  });
});

describe('every language id extracts a definition', () => {
  it('has a representative snippet for each id, and the id matches', () => {
    assert.deepEqual(Object.keys(REPRESENTATIVE).sort(), [...LANGUAGE_IDS].sort());
    for (const id of LANGUAGE_IDS) {
      const sample = REPRESENTATIVE[id];
      assert.ok(sample, `missing representative snippet for ${id}`);
      const result = extract(sample.path, sample.src);
      assert.equal(result.lang, id, `${id} extracted as ${result.lang}`);
      assert.ok(result.defs.length >= 1, `${id} extracted no definitions from ${JSON.stringify(sample.src)}`);
    }
  });
});

describe('masking keeps commented-out definitions out', () => {
  it('ignores // line and block comments in TypeScript', () => {
    const src = [
      '// function hiddenLine() {}',
      '/*',
      'class HiddenBlock {}',
      '*/',
      'function visible() {}',
    ].join('\n');
    const got = names(defsOf('a.ts', src));
    assert.ok(!got.includes('hiddenLine'), got.join(','));
    assert.ok(!got.includes('HiddenBlock'), got.join(','));
    assert.ok(got.includes('visible'), got.join(','));
  });

  it('ignores # comments in Python', () => {
    const got = names(defsOf('a.py', '# def hidden():\n#     pass\ndef visible():\n    pass\n'));
    assert.ok(!got.includes('hidden'), got.join(','));
    assert.ok(got.includes('visible'), got.join(','));
  });

  it('ignores -- comments in SQL and Lua', () => {
    const sql = names(defsOf('a.sql', '-- CREATE TABLE hidden (id INT);\nCREATE TABLE visible (id INT);\n'));
    assert.ok(!sql.includes('hidden'), sql.join(','));
    assert.ok(sql.includes('visible'), sql.join(','));

    const lua = names(defsOf('a.lua', '-- function hidden()\nfunction visible()\nend\n'));
    assert.ok(!lua.includes('hidden'), lua.join(','));
    assert.ok(lua.includes('visible'), lua.join(','));
  });
});

describe('definitions inside string literals stay out', () => {
  it('ignores a Python triple-quoted string spanning lines', () => {
    const src = ['template = """', 'def hidden():', '    pass', '"""', 'def visible():', '    pass'].join('\n');
    const got = names(defsOf('a.py', src));
    assert.ok(!got.includes('hidden'), got.join(','));
    assert.ok(got.includes('visible'), got.join(','));
  });

  it('ignores a JS string containing a function declaration', () => {
    const src = ['const s = "function hidden() {}";', 'function visible() {}'].join('\n');
    const got = names(defsOf('a.ts', src));
    assert.ok(!got.includes('hidden'), got.join(','));
    assert.ok(got.includes('visible'), got.join(','));
  });

  it('ignores a Go raw string containing a func', () => {
    const src = ['var s = `', 'func hidden() {}', '`', 'func visible() {}'].join('\n');
    const got = names(defsOf('a.go', src));
    assert.ok(!got.includes('hidden'), got.join(','));
    assert.ok(got.includes('visible'), got.join(','));
  });
});

describe('Python specifics', () => {
  it('handles decorators, async def and nested class methods with parents', () => {
    const src = [
      '@app.route("/")',
      'async def handler(request):',
      '    pass',
      '',
      'class Service:',
      '    def run(self):',
      '        pass',
      '',
      '    class Inner:',
      '        def inner_run(self):',
      '            pass',
    ].join('\n');
    const list = defsOf('a.py', src);

    const handler = must(list, 'handler');
    assert.equal(handler.kind, 'function');
    assert.equal(handler.line, 2, 'the decorator is not part of the def line');
    assert.equal(handler.parent, undefined);

    const run = must(list, 'run');
    assert.equal(run.kind, 'method');
    assert.equal(run.parent, 'Service');

    const inner = must(list, 'Inner');
    assert.equal(inner.kind, 'class');
    assert.equal(inner.parent, 'Service');

    const innerRun = must(list, 'inner_run');
    assert.equal(innerRun.kind, 'method');
    assert.equal(innerRun.parent, 'Inner');
  });

  it('extracts from/import specifiers as written', () => {
    const src = ['import os, sys as system', 'from a.b import c', 'from .mod import d'].join('\n');
    assert.deepEqual(extract('a.py', src).imports, ['os', 'sys', 'a.b', '.mod']);
  });
});

describe('TypeScript and JavaScript specifics', () => {
  const src = [
    'export interface Shape {',
    '  area(): number;',
    '}',
    '',
    'export type Alias = Shape;',
    '',
    'export default class Widget {',
    '  method(): void {}',
    '}',
    '',
    'export function build<T>(x: T): T {',
    '  return x;',
    '}',
    '',
    'const double = (n: number) => n * 2;',
  ].join('\n');
  const list = defsOf('a.ts', src);

  it('extracts export function, default class, interface, type and arrow const', () => {
    assert.equal(must(list, 'Shape').kind, 'interface');
    assert.equal(must(list, 'Shape').exported, true);
    assert.equal(must(list, 'Alias').kind, 'type');
    assert.equal(must(list, 'Alias').exported, true);

    const widget = must(list, 'Widget');
    assert.equal(widget.kind, 'class');
    assert.equal(widget.exported, true);

    const build = must(list, 'build');
    assert.equal(build.kind, 'function');
    assert.equal(build.exported, true);

    const double = must(list, 'double');
    assert.equal(double.kind, 'function');
    assert.equal(double.line, 15);
  });

  it('sets the class as parent of its methods', () => {
    const method = must(list, 'method');
    assert.equal(method.kind, 'method');
    assert.equal(method.parent, 'Widget');
  });

  it('extracts static imports, re-exports and require specifiers', () => {
    const imports = extract('a.ts', [
      "import x from 'pkg-x';",
      "import { y } from \"pkg-y\";",
      "export { z } from 'pkg-z';",
      "const w = require('pkg-w');",
      "const d = import('pkg-d');",
    ].join('\n')).imports;
    assert.deepEqual(imports, ['pkg-x', 'pkg-y', 'pkg-z', 'pkg-w', 'pkg-d']);
  });

  it('marks capitalised functions as components in tsx but not ts', () => {
    const src = 'export function Button() {\n  return <button />;\n}\n';
    assert.equal(must(defsOf('a.tsx', src), 'Button').kind, 'component');
    assert.equal(must(defsOf('a.ts', src), 'Button').kind, 'function');
  });

  it('trims the signature and strips a trailing brace, arrow or colon', () => {
    assert.equal(must(defsOf('a.ts', 'function f() {}\n'), 'f').signature, 'function f()');
    assert.equal(must(defsOf('a.ts', 'const g = () =>\n'), 'g').signature, 'const g = ()');
    assert.equal(must(defsOf('a.py', 'class C:\n'), 'C').signature, 'class C');
  });
});

describe('Go specifics', () => {
  it('sets the receiver type as the parent of a method', () => {
    const src = [
      'package main',
      '',
      'type Repo struct {',
      '\tName string',
      '}',
      '',
      'func (r *Repo) Save() error {',
      '\treturn nil',
      '}',
      '',
      'func New() *Repo {',
      '\treturn &Repo{}',
      '}',
    ].join('\n');
    const list = defsOf('a.go', src);

    const save = must(list, 'Save');
    assert.equal(save.kind, 'method');
    assert.equal(save.parent, 'Repo');
    assert.equal(save.line, 7);

    const fresh = must(list, 'New');
    assert.equal(fresh.kind, 'function');
    assert.equal(fresh.parent, undefined);

    assert.equal(must(list, 'Repo').kind, 'struct');
  });

  it('extracts import paths from single and grouped imports', () => {
    const src = ['package main', '', 'import "fmt"', '', 'import (', '\t"os"', '\t"net/http"', ')'].join('\n');
    assert.deepEqual(extract('a.go', src).imports, ['fmt', 'os', 'net/http']);
  });
});

describe('Rust specifics', () => {
  it('extracts struct, trait, impl methods and pub functions', () => {
    const src = [
      'pub struct Config {',
      '    value: u32,',
      '}',
      '',
      'pub trait Loader {',
      '    fn load(&self) -> u32;',
      '}',
      '',
      'impl Config {',
      '    pub fn new(value: u32) -> Self {',
      '        Self { value }',
      '    }',
      '}',
    ].join('\n');
    const list = defsOf('a.rs', src);

    const config = must(list, 'Config');
    assert.equal(config.kind, 'struct');
    assert.equal(config.exported, true);

    const loader = must(list, 'Loader');
    assert.equal(loader.kind, 'trait');

    const load = must(list, 'load');
    assert.equal(load.kind, 'method');
    assert.equal(load.parent, 'Loader');

    const fresh = must(list, 'new');
    assert.equal(fresh.kind, 'method');
    assert.equal(fresh.parent, 'Config');
    assert.equal(fresh.exported, true);
  });

  it('survives lifetimes, which look like unclosed char literals', () => {
    const src = "pub fn borrow<'a>(x: &'a str) -> &'a str {\n    x\n}\n";
    const list = defsOf('a.rs', src);
    assert.equal(must(list, 'borrow').kind, 'function');
  });

  it('extracts use paths, expanding a grouped import', () => {
    const src = 'use std::collections::HashMap;\nuse std::{fmt, io};\n';
    assert.deepEqual(extract('a.rs', src).imports, ['std::collections::HashMap', 'std::fmt', 'std::io']);
  });
});

describe('line numbers survive masking', () => {
  it('is exact after a multi-line block comment', () => {
    const src = ['/* first', '   second', '   third */', 'function real() {}'].join('\n');
    const real = must(defsOf('a.ts', src), 'real');
    assert.equal(real.line, 4);
  });

  it('is exact after a multi-line template string', () => {
    const src = ['const t = `line one', 'function fake() {}', '`;', 'function real() {}'].join('\n');
    const list = defsOf('a.ts', src);
    assert.ok(!names(list).includes('fake'), names(list).join(','));
    assert.equal(must(list, 'real').line, 4);
  });

  it('is exact after a multi-line Python string', () => {
    const src = ['text = """', 'def fake():', '    pass', '"""', 'def real():', '    pass'].join('\n');
    const list = defsOf('a.py', src);
    assert.ok(!names(list).includes('fake'));
    assert.equal(must(list, 'real').line, 5);
  });
});

describe('extract never throws', () => {
  const hostile: Array<[string, string]> = [
    ['empty', ''],
    ['whitespace', '   \n\t\n'],
    ['NUL bytes', 'def f():\u0000\u0001\x02 more'],
    ['binary-ish', '\u0000\u0001\u0002\x7fconst x = 1;'],
    ['ends mid-string', 'const s = "unterminated'],
    ['ends mid-template', 'const s = `unterminated'],
    ['ends mid-block-comment', '/* unterminated'],
    ['ends mid-python-string', 'x = """unterminated'],
    ['lone backslash', '\\'],
    ['only comment opener', '//'],
    ['only include', '#include'],
  ];

  it('returns a well-formed result for every hostile input and language', () => {
    for (const id of LANGUAGE_IDS) {
      const path = REPRESENTATIVE[id]?.path ?? 'a.ts';
      for (const [label, text] of hostile) {
        let result: ReturnType<typeof extract> | undefined;
        assert.doesNotThrow(() => {
          result = extract(path, text);
        }, `${id} threw on ${label}`);
        assert.ok(result, `${id}/${label} produced no result`);
        assert.equal(typeof result.lang, 'string');
        assert.ok(Array.isArray(result.defs));
        assert.ok(Array.isArray(result.refs));
        assert.ok(Array.isArray(result.imports));
      }
    }
  });

  it('never returns more refs than the cap', () => {
    const lines: string[] = [];
    for (let i = 0; i < MAX_REFS_PER_FILE + 100; i++) lines.push(`const value${i} = other${i};`);
    const result = extract('big.ts', lines.join('\n'));
    assert.ok(result.refs.length > 0);
    assert.ok(result.refs.length <= MAX_REFS_PER_FILE, `got ${result.refs.length}`);
    assert.equal(new Set(result.refs).size, result.refs.length, 'refs must be distinct');
  });

  it('keeps a file\'s own defined names among its references', () => {
    const result = extract('a.ts', 'export function uniqueSymbolName() {}\n');
    assert.ok(result.refs.includes('uniqueSymbolName'), result.refs.join(','));
  });
});
