"""Compare two SWE-bench runs on the same instance set.

Prints resolve-rate agreement plus Inkling (cloud) vs local token totals so
an Inkling-only arm can be stacked against a hybrid proto arm.

Example:
    $ python compare_ab.py --a inkling40 --b hybrid40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE: Path = Path("/data/prithvi/sweb")


def load_metrics(run_id: str) -> Dict[str, dict]:
    """Load per-instance metrics for a run.

    Args:
        run_id: Directory name under ``BASE/runs``.

    Returns:
        instance_id to last metrics row.
    """
    path: Path = BASE / "runs" / run_id / "metrics.jsonl"
    out: Dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row: dict = json.loads(line)
        out[row["instance_id"]] = row
    return out


def load_grades(run_id: str) -> Dict[str, bool]:
    """Load resolved flags for a run.

    Args:
        run_id: Directory name under ``BASE/runs``.

    Returns:
        instance_id to resolved bool. Empty if grades are missing.
    """
    path: Path = BASE / "runs" / run_id / "grades.json"
    if not path.exists():
        return {}
    grades: List[dict] = json.loads(path.read_text())
    return {g["instance_id"]: bool(g.get("resolved")) for g in grades}


def tok(row: Optional[dict], key: str) -> int:
    """Integer token field, defaulting to 0.

    Args:
        row: Metrics row.
        key: Field name.

    Returns:
        Non-negative int.
    """
    if not row:
        return 0
    return int(row.get(key) or 0)


def main() -> None:
    """Print the A/B comparison table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="inkling-only run id")
    parser.add_argument("--b", required=True, help="hybrid proto run id")
    args = parser.parse_args()

    ma: Dict[str, dict] = load_metrics(args.a)
    mb: Dict[str, dict] = load_metrics(args.b)
    ga: Dict[str, bool] = load_grades(args.a)
    gb: Dict[str, bool] = load_grades(args.b)
    ids: List[str] = sorted(set(ma) | set(mb))

    def sums(metrics: Dict[str, dict]) -> Tuple[int, int, int, int]:
        """Return local_in, local_out, cloud_in, cloud_out totals."""
        return (
            sum(tok(r, "local_input_tokens") for r in metrics.values()),
            sum(tok(r, "local_output_tokens") for r in metrics.values()),
            sum(tok(r, "cloud_input_tokens") for r in metrics.values()),
            sum(tok(r, "cloud_output_tokens") for r in metrics.values()),
        )

    a_lin, a_lout, a_cin, a_cout = sums(ma)
    b_lin, b_lout, b_cin, b_cout = sums(mb)
    a_cloud: int = a_cin + a_cout
    b_cloud: int = b_cin + b_cout
    a_local: int = a_lin + a_lout
    b_local: int = b_lin + b_lout

    a_res: int = sum(1 for i in ids if ga.get(i))
    b_res: int = sum(1 for i in ids if gb.get(i))
    both: int = sum(1 for i in ids if ga.get(i) and gb.get(i))
    only_a: List[str] = [i for i in ids if ga.get(i) and not gb.get(i)]
    only_b: List[str] = [i for i in ids if gb.get(i) and not ga.get(i)]
    n_graded: int = sum(1 for i in ids if i in ga and i in gb)

    print(f"instances with metrics: {len(ids)}  graded both: {n_graded}")
    print()
    print("| arm | resolved | Inkling in/out | Inkling total | local in/out | local total | mean wall |")
    print("|---|---|---|---|---|---|---|")
    def wall(metrics: Dict[str, dict]) -> float:
        """Mean per-instance wall seconds."""
        if not metrics:
            return 0.0
        return sum(float(r.get("wall_s") or 0) for r in metrics.values()) / len(metrics)

    print(
        f"| {args.a} (Inkling-only) | {a_res}/{len(ga) or len(ma)} | "
        f"{a_cin:,}/{a_cout:,} | {a_cloud:,} | {a_lin:,}/{a_lout:,} | {a_local:,} | {wall(ma):.0f}s |"
    )
    print(
        f"| {args.b} (hybrid proto) | {b_res}/{len(gb) or len(mb)} | "
        f"{b_cin:,}/{b_cout:,} | {b_cloud:,} | {b_lin:,}/{b_lout:,} | {b_local:,} | {wall(mb):.0f}s |"
    )
    saved: int = a_cloud - b_cloud
    print()
    print(f"Inkling tokens saved by hybrid: {saved:,} ({(100 * saved / a_cloud) if a_cloud else 0:.1f}% of Inkling-only)")
    print(f"Hybrid extra local tokens: {b_local - a_local:,}")
    if n_graded:
        print(f"Agree resolved: {both}/{n_graded}")
        print(f"Only Inkling-only: {', '.join(only_a) or '—'}")
        print(f"Only hybrid: {', '.join(only_b) or '—'}")

    print()
    print("| instance | inkling-only tokens | hybrid local / Inkling | hybrid fixer | A | B |")
    print("|---|---|---|---|---|---|")
    for iid in ids:
        ra: Optional[dict] = ma.get(iid)
        rb: Optional[dict] = mb.get(iid)
        a_mark: str = "PASS" if ga.get(iid) else ("FAIL" if iid in ga else "…")
        b_mark: str = "PASS" if gb.get(iid) else ("FAIL" if iid in gb else "…")
        print(
            f"| {iid} | {tok(ra, 'cloud_tokens'):,} | "
            f"{tok(rb, 'local_tokens'):,} / {tok(rb, 'cloud_tokens'):,} | "
            f"{(rb or {}).get('fixer', '')} | {a_mark} | {b_mark} |"
        )

    summary: dict = {
        "a": args.a,
        "b": args.b,
        "n": len(ids),
        "a_resolved": a_res,
        "b_resolved": b_res,
        "a_inkling_tokens": a_cloud,
        "b_inkling_tokens": b_cloud,
        "b_local_tokens": b_local,
        "inkling_tokens_saved": saved,
        "agree": both,
        "only_a": only_a,
        "only_b": only_b,
    }
    out: Path = BASE / "runs" / f"compare_{args.a}_vs_{args.b}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
