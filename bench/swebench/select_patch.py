"""Candidate patch selection by adjacent-regression testing.

Given several candidate patches for an instance, prefer the one that (a)
applies cleanly, (b) breaks the fewest of the repo's *existing* tests near the
touched files, and (c) is smallest. Hidden evaluation tests are never used —
this is exactly the signal a real user has locally, which is what makes it fair
and what makes it transferable outside the benchmark.

Test targets are guessed structurally: a touched ``pkg/foo/bar.py`` maps to
test files whose names contain ``bar`` or ``foo`` under the repo's test root.
That heuristic covers the seven repos in the fixed instance set well.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE: Path = Path("/data/prithvi/sweb")

# repo -> (test root, style). Style "django" uses runtests.py labels.
TEST_LAYOUT: Dict[str, Tuple[str, str]] = {
    "django/django": ("tests", "django"),
    "sympy/sympy": ("sympy", "colocated"),  # sympy/<pkg>/tests/test_<stem>.py
    "sphinx-doc/sphinx": ("tests", "flat"),
    "pytest-dev/pytest": ("testing", "flat"),
    "pylint-dev/pylint": ("tests", "flat"),
    "psf/requests": ("tests", "flat"),
    "pydata/xarray": ("xarray/tests", "flat"),
}


def touched_source_files(patch: str) -> List[str]:
    """Paths of non-test source files modified by a patch.

    Args:
        patch: Unified diff text.

    Returns:
        Repo-relative paths.
    """
    files: List[str] = re.findall(r"^\+\+\+ b/(\S+)", patch, flags=re.M)
    return [f for f in files if f.endswith(".py") and "/test" not in f and not f.startswith("test")]


def guess_test_files(repo_name: str, repo_dir: Path, touched: List[str]) -> List[str]:
    """Guess existing test files that exercise the touched source files.

    Args:
        repo_name: GitHub ``org/name``.
        repo_dir: Checkout root.
        touched: Touched source paths.

    Returns:
        Up to three repo-relative test file paths that exist.
    """
    root, style = TEST_LAYOUT[repo_name]
    found: List[str] = []
    for path in touched:
        parts = Path(path)
        keys = [parts.stem, parts.parent.name]
        if style == "colocated":
            candidate = parts.parent / "tests" / f"test_{parts.stem}.py"
            if (repo_dir / candidate).exists():
                found.append(str(candidate))
            continue
        for key in keys:
            if not key or key in (".", "src"):
                continue
            for hit in sorted((repo_dir / root).rglob(f"test_*{key}*.py"))[:2]:
                found.append(str(hit.relative_to(repo_dir)))
    seen: List[str] = []
    for f in found:
        if f not in seen:
            seen.append(f)
    return seen[:3]


def django_labels(repo_dir: Path, touched: List[str]) -> List[str]:
    """Map touched Django source files to runtests.py labels.

    Args:
        repo_dir: Checkout root.
        touched: Touched source paths (``django/<app>/...``).

    Returns:
        Existing test labels, at most three.
    """
    labels: List[str] = []
    for path in touched:
        parts = path.split("/")
        if len(parts) < 2 or parts[0] != "django":
            continue
        app = parts[1]
        if (repo_dir / "tests" / app).exists() and app not in labels:
            labels.append(app)
    return labels[:3]


def regression_failures(instance: dict, patch: str, scratch: Path) -> Optional[int]:
    """Count regression failures a candidate patch causes in adjacent tests.

    Args:
        instance: Instance row.
        patch: Candidate diff.
        scratch: Directory to build the throwaway checkout in.

    Returns:
        Number of failing adjacent test files/labels, or None when the patch
        does not apply (treated as worst by the caller).
    """
    iid: str = instance["instance_id"]
    repo_dir: Path = scratch / "check-repo"
    mirror: Path = BASE / "repos" / (instance["repo"].replace("/", "__") + ".git")
    if repo_dir.exists():
        subprocess.run(["rm", "-rf", str(repo_dir)], check=True)
    subprocess.run(["git", "clone", "--shared", "-q", str(mirror), str(repo_dir)], check=True, timeout=600)
    subprocess.run(["git", "checkout", "-q", instance["base_commit"]], cwd=repo_dir, check=True, timeout=120)
    applied = subprocess.run(
        ["git", "apply", "--whitespace=nowarn"], cwd=repo_dir, input=patch, capture_output=True, text=True, timeout=60
    )
    if applied.returncode != 0:
        return None

    env_dir: Path = BASE / "envs" / iid
    if not (env_dir / ".ready").exists():
        return 0
    import os

    env: Dict[str, str] = {
        **os.environ,
        "PATH": f"{env_dir}/bin:/usr/bin:/bin",
        "HOME": "/data/prithvi/home",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(repo_dir),
    }
    python: str = str(env_dir / "bin" / "python")
    touched: List[str] = touched_source_files(patch)

    if instance["repo"] == "django/django":
        labels: List[str] = django_labels(repo_dir, touched)
        if not labels:
            return 0
        res = subprocess.run(
            [python, "tests/runtests.py", "--verbosity", "0", "--parallel", "1", *labels],
            cwd=repo_dir, env=env, capture_output=True, text=True, timeout=90,
        )
        return 0 if res.returncode == 0 else 1

    test_files: List[str] = guess_test_files(instance["repo"], repo_dir, touched)
    if not test_files:
        return 0
    res = subprocess.run(
        [python, "-m", "pytest", "-q", "--no-header", "-x", "-p", "no:cacheprovider", *test_files],
        cwd=repo_dir, env=env, capture_output=True, text=True, timeout=90,
    )
    m = re.search(r"(\d+) failed", res.stdout + res.stderr)
    if m:
        return int(m.group(1))
    return 0 if res.returncode == 0 else 1


def normalize_patch(patch: str) -> str:
    """Normalize a diff for consensus comparison.

    Args:
        patch: Unified diff text.

    Returns:
        The diff with index/hash lines and hunk headers' line numbers removed,
        so two semantically identical patches from different runs compare equal.
    """
    lines: List[str] = []
    for line in patch.splitlines():
        if line.startswith("index ") or line.startswith("@@"):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def select_best(instance: dict, candidates: List[str], scratch: Path) -> Tuple[int, List[dict]]:
    """Choose the best candidate patch index.

    Args:
        instance: Instance row.
        candidates: Candidate diffs, one per attempt.
        scratch: Working directory for throwaway checkouts.

    Returns:
        (best_index, per-candidate score rows). Scoring prefers: applies
        cleanly, fewest adjacent regressions, most consensus (identical
        normalized patch produced by independent attempts), smallest diff.
        All-empty input returns index 0.
    """
    normalized: List[str] = [normalize_patch(p) for p in candidates]
    counts: Dict[str, int] = {}
    for n in normalized:
        if n:
            counts[n] = counts.get(n, 0) + 1

    rows: List[dict] = []
    scored: Dict[str, Optional[int]] = {}
    for i, patch in enumerate(candidates):
        if not patch.strip():
            rows.append({"idx": i, "empty": True, "applies": False, "failures": None, "consensus": 0, "size": 0})
            continue
        # Identical candidates share one regression run.
        key_n: str = normalized[i]
        if key_n not in scored:
            scored[key_n] = regression_failures(instance, patch, scratch)
        failures: Optional[int] = scored[key_n]
        rows.append(
            {
                "idx": i,
                "empty": False,
                "applies": failures is not None,
                "failures": failures,
                "consensus": counts.get(key_n, 1),
                "size": len(patch),
            }
        )

    def key(r: dict) -> Tuple[int, int, int, int, int]:
        # Smallest-diff-wins was discarding complete cloud patches in favour
        # of a 400-byte local no-op that happened not to break adjacent tests.
        # Prefer: applies, fewest regressions, most independent agreement,
        # then a *complete* (not tiny) change, then earlier attempts.
        size: int = r["size"]
        completeness: int = 0 if size >= 400 else (1 if size >= 120 else 2)
        return (
            0 if (not r["empty"] and r["applies"]) else 1,
            r["failures"] if r["failures"] is not None else 10_000,
            -r["consensus"],
            completeness,
            r["idx"],
        )

    best = sorted(rows, key=key)[0]["idx"] if rows else 0
    return best, rows


if __name__ == "__main__":
    import sys

    instance = json.loads(Path(sys.argv[1]).read_text())
    print(select_best(instance, [Path(p).read_text() for p in sys.argv[2:]], Path("/tmp/selcheck")))
