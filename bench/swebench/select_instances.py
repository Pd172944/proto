"""Select a fixed, deterministic set of SWE-bench Verified instances.

The iteration loop in this benchmark reruns the *same* instances across many
harness revisions, so the selection must be stable. We take a quota of
instances per repository (biased toward pure-Python projects whose test
environments can be built with a plain ``pip install -e .``), sorting
instance ids descending so we get recent issues whose project versions run
on a modern CPython.

Example:
    $ python select_instances.py --out instances.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

# repo -> number of instances to take. Total = 10.
REPO_QUOTA: Dict[str, int] = {
    "django/django": 3,
    "sympy/sympy": 2,
    "sphinx-doc/sphinx": 1,
    "pytest-dev/pytest": 1,
    "pylint-dev/pylint": 1,
    "psf/requests": 1,
    "pydata/xarray": 1,
}

FIELDS: List[str] = [
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "test_patch",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
]


def numeric_suffix(instance_id: str) -> int:
    """Return the trailing issue number of an instance id for sorting.

    Args:
        instance_id: An id such as ``django__django-15790``.

    Returns:
        The integer issue number (``15790`` in the example).
    """
    return int(instance_id.rsplit("-", 1)[1])


def main() -> None:
    """Build the fixed instance set and write it as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "instances.json")
    args = parser.parse_args()

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    by_repo: Dict[str, List[dict]] = {}
    for row in ds:
        by_repo.setdefault(row["repo"], []).append(row)

    chosen: List[dict] = []
    for repo, quota in REPO_QUOTA.items():
        rows = sorted(by_repo[repo], key=lambda r: numeric_suffix(r["instance_id"]), reverse=True)
        for row in rows[:quota]:
            chosen.append({k: row[k] for k in FIELDS})

    chosen.sort(key=lambda r: r["instance_id"])
    args.out.write_text(json.dumps(chosen, indent=2))
    print(f"wrote {len(chosen)} instances to {args.out}")
    for row in chosen:
        print(f"  {row['instance_id']}  ({row['version']})")


if __name__ == "__main__":
    main()
