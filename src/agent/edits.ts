/**
 * Classify agent file edits as production work vs disposable scratch.
 *
 * The one-shot no-edit guard used to treat *any* write as progress. On
 * SWE-bench that is the wrong signal: models write `repro.py`, summarise the
 * fix they would make, and stop. Those files are stripped from the submitted
 * patch, so the task looks "done" to the harness and empty to the grader.
 */

const ROOT_SCRATCH: RegExp = /^(repro\d*|reproduce|apply_fix|scratch|diag|foo)[^.]*\.(py|sh)$/i;
const NAMED_SCRATCH: RegExp = /(^|\/)(repro\d*|reproduce|apply_fix)\.(py|sh)$/i;
const TEST_PATH: RegExp = /(^|\/)tests?\//;
const TEST_FILE: RegExp = /(^|\/)test_[^/]+$/;

/** True when ``path`` is scratch or a test — ignored by the no-edit / stuck guards. */
export function isDisposableEdit(path: string): boolean {
  const norm: string = path.replace(/\\/g, '/').replace(/^\.\//, '');
  const base: string = norm.split('/').pop() ?? norm;
  if (ROOT_SCRATCH.test(base)) return true;
  if (NAMED_SCRATCH.test(norm)) return true;
  if (TEST_PATH.test(norm)) return true;
  if (TEST_FILE.test(norm)) return true;
  return false;
}

/** Production-source edits only: files that should count as progress on a mutating task. */
export function productionEdits(files: string[]): string[] {
  return files.filter((f) => !isDisposableEdit(f));
}
