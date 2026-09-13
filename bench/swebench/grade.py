"""Grade a run's predictions locally against the official instance tests.

For each instance: fresh checkout at the base commit, apply the model patch,
apply the dataset's ``test_patch`` (which adds/updates the tests the fix is
graded on), then run FAIL_TO_PASS and PASS_TO_PASS in the instance's
prepared virtualenv. ``resolved`` means every F2P test passes and every P2P
test still passes — the same definition sb-cli uses, evaluated locally.

This exists because the sb-cli key-generation endpoint was down when this
pipeline was built; results can be re-submitted to sb-cli for canonical
grading whenever it recovers.

Example:
    $ python grade.py --run-id iter01
"""

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE: Path = Path("/data/prithvi/sweb")


def sh(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None, timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run a command with captured output.

    Args:
        cmd: Argv list.
        cwd: Working directory.
        env: Environment (default: inherited).
        timeout: Kill after this many seconds.

    Returns:
        Completed process.
    """
    import os

    return subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True, timeout=timeout)


def django_label(test_id: str) -> Optional[str]:
    """Convert a Django-format test id to a runtests.py label.

    Args:
        test_id: e.g. ``test_clear_cache (apps.tests.AppsTests.test_clear_cache)``.

    Returns:
        Label such as ``apps.tests.AppsTests.test_clear_cache``, or None when
        the entry is a docstring line rather than a test id (SWE-bench keys
        some Django tests by the first line of their docstring).
    """
    m = re.match(r"^[\w.]+ \(([\w.]+)\)$", test_id.strip())
    return m.group(1) if m else None


def parse_django_verbose(output: str) -> Dict[str, str]:
    """Parse ``runtests.py --verbosity 2`` output into test -> status.

    Mirrors the official SWE-bench Django log parser: a test with a docstring
    prints its id on one line and ``<docstring> ... <status>`` on the next, so
    results are keyed by *both* the id form and the docstring line.

    Args:
        output: Combined stdout/stderr of the test run.

    Returns:
        Mapping from ``test_name (dotted.path)`` and docstring-line keys to
        ``ok`` / ``FAIL`` / ``ERROR`` / ``skipped``.
    """
    results: Dict[str, str] = {}
    pending: Optional[str] = None
    id_re = re.compile(r"^([\w]+ \([\w.]+\))(?: \.\.\. (.+))?$")
    for raw in output.splitlines():
        line = raw.rstrip()
        m = id_re.match(line)
        if m:
            test_id, status = m.group(1), m.group(2)
            if status:
                results[test_id] = status.split()[0]
                pending = None
            else:
                pending = test_id
            continue
        if pending and " ... " in line:
            doc, status = line.rsplit(" ... ", 1)
            results[pending] = status.split()[0]
            results[doc.strip()] = status.split()[0]
            pending = None
        elif pending and line.endswith(("... ok", "... FAIL", "... ERROR")):
            doc, status = line.rsplit("... ", 1)
            results[pending] = status.strip()
            results[doc.strip()] = status.strip()
            pending = None
    return results


def run_django_tests(instance: dict, repo: Path, tests: List[str], env: Dict[str, str], timeout: int) -> Tuple[bool, str]:
    """Run Django tests by module and check each expected test's status.

    Args:
        instance: Instance row.
        repo: Patched checkout.
        tests: Official test id strings (ids or docstring lines).
        env: Prepared environment for the venv.
        timeout: Seconds.

    Returns:
        (all_expected_tests_passed, diagnostic_tail).
    """
    python: str = str(BASE / "envs" / instance["instance_id"] / "bin" / "python")
    modules: List[str] = sorted(
        {".".join(label.split(".")[:2]) for t in tests if (label := django_label(t)) is not None}
    )
    res = subprocess.run(
        [python, "tests/runtests.py", "--verbosity", "2", "--parallel", "1", *modules],
        cwd=repo,
        env={**__import__("os").environ, **env},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    results: Dict[str, str] = parse_django_verbose(res.stdout + res.stderr)
    failures: List[str] = []
    for t in tests:
        label = django_label(t)
        key = t.strip() if label is None else f"{label.rsplit('.', 1)[1]} ({label})"
        status = results.get(key)
        if status is None and label is None:
            status = results.get(t.strip())
        if status not in ("ok", "skipped"):
            failures.append(f"{t} -> {status or 'not run'}")
    if failures:
        return False, "\n".join(failures[:10])
    return True, "ok"


def find_sympy_file(name: str, repo: Path) -> Optional[str]:
    """Locate the sympy test file defining ``name``.

    Args:
        name: Bare test function name, e.g. ``test_issue_24543``.
        repo: Repository root.

    Returns:
        Relative path to the file, or None.
    """
    res = sh(["grep", "-rl", "--include=test_*.py", f"def {name}(", "sympy"], cwd=repo, timeout=120)
    files = [line for line in res.stdout.splitlines() if line]
    return files[0] if files else None


def run_tests(instance: dict, repo: Path, tests: List[str], timeout: int) -> Tuple[bool, str]:
    """Run a set of official test ids in the instance venv.

    Args:
        instance: Instance row.
        repo: Checkout with model patch + test patch applied.
        tests: Official test id strings.
        timeout: Seconds.

    Returns:
        (all_passed, tail_of_output).
    """
    env_dir: Path = BASE / "envs" / instance["instance_id"]
    python: str = str(env_dir / "bin" / "python")
    env: Dict[str, str] = {
        "PATH": f"{env_dir}/bin:/usr/bin:/bin",
        "HOME": "/data/prithvi/home",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        # The venv's editable install points at the *unpatched* source the env
        # was built from; the patched checkout must win the import race.
        "PYTHONPATH": str(repo),
    }
    repo_name: str = instance["repo"]

    if repo_name == "django/django":
        return run_django_tests(instance, repo, tests, env, timeout)

    if repo_name == "sympy/sympy":
        by_file: Dict[str, List[str]] = {}
        for name in tests:
            f = find_sympy_file(name, repo)
            if f is None:
                return False, f"could not locate test {name}"
            by_file.setdefault(f, []).append(name)
        for f, names in by_file.items():
            expr = " or ".join(names)
            res = sh([python, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", f, "-k", expr], cwd=repo, env=env, timeout=timeout)
            if res.returncode != 0:
                return False, (res.stdout + res.stderr)[-3000:]
        return True, "ok"

    # pytest node-id repos: requests, xarray, pylint, pytest, sphinx
    res = sh([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", *tests], cwd=repo, env=env, timeout=timeout)
    return res.returncode == 0, (res.stdout + res.stderr)[-3000:]


def grade_instance(instance: dict, run_dir: Path, timeout: int) -> dict:
    """Grade one instance's prediction.

    Args:
        instance: Instance row (with test_patch).
        run_dir: The run directory containing per-instance patches.
        timeout: Per-test-set timeout in seconds.

    Returns:
        Grade row with ``resolved`` plus F2P/P2P detail.
    """
    iid: str = instance["instance_id"]
    patch_file: Path = run_dir / iid / "patch.diff"
    patch: str = patch_file.read_text() if patch_file.exists() else ""
    if not patch.strip():
        return {"instance_id": iid, "resolved": False, "reason": "empty patch"}

    env_ready: bool = (BASE / "envs" / iid / ".ready").exists()
    if not env_ready:
        return {"instance_id": iid, "resolved": None, "reason": "no test env"}

    grade_repo: Path = run_dir / iid / "grade-repo"
    mirror: Path = BASE / "repos" / (instance["repo"].replace("/", "__") + ".git")
    if grade_repo.exists():
        sh(["rm", "-rf", str(grade_repo)], cwd=run_dir)
    sh(["git", "clone", "--shared", "-q", str(mirror), str(grade_repo)], cwd=run_dir, timeout=600)
    sh(["git", "checkout", "-q", instance["base_commit"]], cwd=grade_repo)

    apply_model = subprocess.run(
        ["git", "apply", "--whitespace=nowarn"], cwd=grade_repo, input=patch, capture_output=True, text=True, timeout=60
    )
    if apply_model.returncode != 0:
        return {"instance_id": iid, "resolved": False, "reason": f"model patch failed to apply: {apply_model.stderr[-500:]}"}

    apply_test = subprocess.run(
        ["git", "apply", "--whitespace=nowarn"], cwd=grade_repo, input=instance["test_patch"], capture_output=True, text=True, timeout=60
    )
    if apply_test.returncode != 0:
        return {"instance_id": iid, "resolved": False, "reason": f"test patch failed to apply: {apply_test.stderr[-500:]}"}

    f2p: List[str] = json.loads(instance["FAIL_TO_PASS"])
    p2p: List[str] = json.loads(instance["PASS_TO_PASS"])

    f2p_ok, f2p_out = run_tests(instance, grade_repo, f2p, timeout)
    if not f2p_ok:
        return {"instance_id": iid, "resolved": False, "reason": "F2P failed", "detail": f2p_out[-800:]}
    p2p_ok, p2p_out = run_tests(instance, grade_repo, p2p, timeout)
    return {
        "instance_id": iid,
        "resolved": bool(f2p_ok and p2p_ok),
        "reason": "ok" if p2p_ok else "P2P regression",
        **({} if p2p_ok else {"detail": p2p_out[-800:]}),
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--instances", type=Path, default=Path(__file__).parent / "instances.json")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    instances: List[dict] = json.loads(args.instances.read_text())
    run_dir: Path = BASE / "runs" / args.run_id

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows: List[dict] = list(pool.map(lambda i: grade_instance(i, run_dir, args.timeout), instances))

    resolved = sum(1 for r in rows if r["resolved"])
    gradeable = sum(1 for r in rows if r["resolved"] is not None)
    for row in rows:
        mark = "PASS" if row["resolved"] else ("SKIP" if row["resolved"] is None else "FAIL")
        print(f"  {mark}  {row['instance_id']}  ({row['reason']})")
    print(f"\nresolved: {resolved}/{len(rows)} ({gradeable} gradeable locally)")
    (run_dir / "grades.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
