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


def scaled_quotas(n: int) -> Dict[str, int]:
    """Scale the base 10-task repo mix to ``n`` instances.

    The original 10-task set is the most-recent slice of this mix, so n=40
    keeps those ten as a prefix of each repo's quota.

    Args:
        n: Target instance count.

    Returns:
        Repo to quota mapping that sums to ``n``.
    """
    base_total: int = sum(REPO_QUOTA.values())
    scaled: Dict[str, int] = {repo: (count * n) // base_total for repo, count in REPO_QUOTA.items()}
    remainder: int = n - sum(scaled.values())
    order: List[str] = sorted(REPO_QUOTA, key=lambda repo: REPO_QUOTA[repo], reverse=True)
    idx: int = 0
    while remainder > 0:
        scaled[order[idx % len(order)]] += 1
        remainder -= 1
        idx += 1
    return scaled


def main() -> None:
    """Build the fixed instance set and write it as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "instances.json")
    parser.add_argument("--n", type=int, default=sum(REPO_QUOTA.values()), help="instance count (default: 10)")
    args = parser.parse_args()

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    by_repo: Dict[str, List[dict]] = {}
    for row in ds:
        by_repo.setdefault(row["repo"], []).append(row)

    quotas: Dict[str, int] = scaled_quotas(args.n)
    chosen: List[dict] = []
    for repo, quota in quotas.items():
        rows = sorted(by_repo[repo], key=lambda r: numeric_suffix(r["instance_id"]), reverse=True)
        take: int = min(quota, len(rows))
        for row in rows[:take]:
            chosen.append({k: row[k] for k in FIELDS})

    chosen.sort(key=lambda r: r["instance_id"])
    args.out.write_text(json.dumps(chosen, indent=2))
    print(f"wrote {len(chosen)} instances to {args.out}")
    for row in chosen:
        print(f"  {row['instance_id']}  ({row['version']})")


if __name__ == "__main__":
    main()
