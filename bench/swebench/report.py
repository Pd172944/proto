"""Aggregate SWE-bench iteration metrics into a single comparison table.

Reads each ``BASE/runs/<id>/{metrics.jsonl,grades.json}`` and prints a
markdown table: resolve rate, cost, wall time, local-vs-cloud mix.

Example:
    $ python report.py --runs iter01-baseline,iter02,iter03
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

BASE: Path = Path("/data/prithvi/sweb")


def load_rows(run_id: str) -> Dict[str, object]:
    """Load one run's metrics and grades into a summary row.

    Args:
        run_id: Directory name under ``BASE/runs``.

    Returns:
        Summary dict with resolve rate, cost, timing, and tier mix.
    """
    run_dir: Path = BASE / "runs" / run_id
    metrics: List[dict] = []
    metrics_path: Path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    grades: List[dict] = []
    grades_path: Path = run_dir / "grades.json"
    if grades_path.exists():
        grades = json.loads(grades_path.read_text())

    n: int = len(grades) if grades else len({m["instance_id"] for m in metrics})
    resolved: int = sum(1 for g in grades if g.get("resolved"))
    cost: float = sum((m.get("cost_usd") or 0) for m in metrics)
    walls: List[float] = [m["wall_s"] for m in metrics if m.get("wall_s") is not None]
    # Per-instance wall: if several attempts, sum them (that's what the user waited).
    per_inst: Dict[str, float] = {}
    for m in metrics:
        per_inst[m["instance_id"]] = per_inst.get(m["instance_id"], 0.0) + (m.get("wall_s") or 0)
    local_n: int = sum(1 for m in metrics if m.get("tier") == "local")
    cloud_n: int = sum(1 for m in metrics if isinstance(m.get("tier"), str) and str(m["tier"]).startswith("cloud"))
    # Scouts are a separate local-model pass and do not appear as metrics rows.
    scout_n: int = sum(1 for p in run_dir.iterdir() if p.is_dir() and (p / "scout.out").exists())
    if scout_n == 0:
        scout_n = sum(1 for _ in run_dir.rglob("scout.out"))
    patches: int = sum(1 for m in metrics if (m.get("patch_bytes") or 0) > 0)
    return {
        "run": run_id,
        "resolved": resolved,
        "n": n or 10,
        "rate": resolved / (n or 10),
        "cost": cost,
        "mean_wall_s": (sum(per_inst.values()) / len(per_inst)) if per_inst else 0.0,
        "local_calls": local_n,
        "cloud_calls": cloud_n,
        "scout_calls": scout_n,
        "attempts": len(metrics),
        "nonempty_patches": patches,
        "passed": [g["instance_id"] for g in grades if g.get("resolved")],
    }


def main() -> None:
    """Print the comparison table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, default="")
    args = parser.parse_args()
    ids: List[str] = [x for x in args.runs.split(",") if x] if args.runs else sorted(
        p.name for p in (BASE / "runs").iterdir() if (p / "metrics.jsonl").exists()
    )
    rows: List[Dict[str, object]] = [load_rows(i) for i in ids]
    print("| run | resolved | cost | mean wall | local patches / local scouts / cloud | passed |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['run']} | {r['resolved']}/{r['n']} ({100 * r['rate']:.0f}%) | "
            f"${r['cost']:.2f} | {r['mean_wall_s']:.0f}s | "
            f"{r['local_calls']}/{r['scout_calls']}/{r['cloud_calls']} | {', '.join(r['passed']) or '—'} |"
        )
    (BASE / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {BASE / 'summary.json'}")


if __name__ == "__main__":
    main()
