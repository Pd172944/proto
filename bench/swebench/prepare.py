"""Prepare local mirrors and per-instance test environments for SWE-bench runs.

Two one-time jobs, both idempotent:

1. ``--mirrors``: bare-mirror clone of every repository referenced by
   ``instances.json`` into ``BASE/repos/<org>__<name>.git``. Per-run scratch
   checkouts then clone from the mirror, which is a local, near-instant
   operation.
2. ``--envs``: a virtualenv per instance in ``BASE/envs/<instance_id>``, with
   the project installed editable at the instance's base commit, so the agent
   can actually run the project's test suite. A failed build is recorded but
   not fatal — the agent simply works without runnable tests for that
   instance, and grading still happens on the sb-cli side.

Example:
    $ python prepare.py --mirrors --envs
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

BASE: Path = Path("/data/prithvi/sweb")
PYTHON: str = sys.executable

# Extra packages some projects need for their test suites beyond `pip install -e .`.
TEST_DEPS: Dict[str, List[str]] = {
    "django/django": ["pytz"],
    "sympy/sympy": ["pytest"],
    "sphinx-doc/sphinx": ["pytest"],
    "pytest-dev/pytest": [],
    "pylint-dev/pylint": ["pytest", "GitPython"],
    "psf/requests": ["pytest", "pytest-httpbin==2.0.0", "trustme"],
    "pydata/xarray": ["pytest", "numpy<2", "pandas<2.1", "packaging"],
}


def run(cmd: List[str], cwd: Path | None = None, timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run a command, capturing output.

    Args:
        cmd: Argv list.
        cwd: Working directory.
        timeout: Seconds before the command is killed.

    Returns:
        The completed process (check ``returncode`` yourself).
    """
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def ensure_mirror(repo: str) -> Path:
    """Clone or update the bare mirror for ``repo``.

    Args:
        repo: GitHub ``org/name``.

    Returns:
        Path to the bare mirror.
    """
    dest: Path = BASE / "repos" / (repo.replace("/", "__") + ".git")
    if dest.exists():
        print(f"mirror exists: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {repo} ...")
    res = run(["git", "clone", "--bare", f"https://github.com/{repo}.git", str(dest)], timeout=3600)
    if res.returncode != 0:
        print(res.stderr[-2000:])
        raise RuntimeError(f"clone failed for {repo}")
    return dest


def build_env(instance: dict) -> bool:
    """Build the per-instance virtualenv with the project installed editable.

    Args:
        instance: A row from ``instances.json``.

    Returns:
        True if the environment was built (or already exists), False on failure.
    """
    iid: str = instance["instance_id"]
    env_dir: Path = BASE / "envs" / iid
    marker: Path = env_dir / ".ready"
    if marker.exists():
        print(f"env exists: {iid}")
        return True

    # Fresh checkout at the base commit to install from.
    src: Path = BASE / "envs" / f"{iid}.src"
    mirror: Path = BASE / "repos" / (instance["repo"].replace("/", "__") + ".git")
    if src.exists():
        run(["rm", "-rf", str(src)])
    run(["git", "clone", "--shared", str(mirror), str(src)])
    res = run(["git", "checkout", "-q", instance["base_commit"]], cwd=src)
    if res.returncode != 0:
        print(f"checkout failed for {iid}: {res.stderr[-500:]}")
        return False

    run([PYTHON, "-m", "venv", str(env_dir)])
    pip: str = str(env_dir / "bin" / "pip")
    steps: List[List[str]] = [
        [pip, "install", "-q", "--upgrade", "pip", "setuptools", "wheel"],
        [pip, "install", "-q", "-e", str(src)],
        *([[pip, "install", "-q", *TEST_DEPS[instance["repo"]]]] if TEST_DEPS.get(instance["repo"]) else []),
    ]
    for step in steps:
        res = run(step, cwd=src, timeout=2400)
        if res.returncode != 0:
            print(f"env build failed for {iid} at {' '.join(step[:4])}:\n{res.stderr[-1500:]}")
            return False
    marker.write_text("ok")
    print(f"env built: {iid}")
    return True


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirrors", action="store_true")
    parser.add_argument("--envs", action="store_true")
    parser.add_argument("--instances", type=Path, default=Path(__file__).parent / "instances.json")
    parser.add_argument("--workers", type=int, default=1, help="parallel env builds")
    args = parser.parse_args()

    instances: List[dict] = json.loads(args.instances.read_text())
    if args.mirrors:
        for repo in sorted({i["repo"] for i in instances}):
            ensure_mirror(repo)
    if args.envs:
        if args.workers <= 1:
            results: Dict[str, bool] = {i["instance_id"]: build_env(i) for i in instances}
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pairs: List[Tuple[str, bool]] = list(
                    pool.map(lambda inst: (inst["instance_id"], build_env(inst)), instances)
                )
            results = dict(pairs)
        ok = sum(results.values())
        print(f"\nenvs ready: {ok}/{len(results)}")
        for iid, good in results.items():
            if not good:
                print(f"  FAILED: {iid}")


if __name__ == "__main__":
    main()
