"""Re-run patch selection on an existing multi-attempt run.

Used when the selector changes and we want to measure the lift without
re-calling any model.

Example:
    $ python reselect.py --run-id iter05
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from select_patch import select_best
from run import collect_model_patch

BASE: Path = Path("/data/prithvi/sweb")


def main() -> None:
    """Re-select and rewrite preds.json / per-instance patch.diff."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--instances", type=Path, default=Path(__file__).parent / "instances.json")
    args = parser.parse_args()

    instances: List[dict] = json.loads(args.instances.read_text())
    run_dir: Path = BASE / "runs" / args.run_id
    preds: Dict[str, dict] = {}
    for instance in instances:
        iid: str = instance["instance_id"]
        candidates: List[str] = []
        for a in range(8):
            repo = run_dir / iid / f"a{a}" / "repo"
            p = run_dir / iid / f"a{a}" / "patch.diff"
            if not p.exists() and not repo.exists():
                break
            if repo.exists():
                patch_a = collect_model_patch(repo)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(patch_a)
            else:
                patch_a = p.read_text()
            candidates.append(patch_a)
        if not candidates:
            single = run_dir / iid / "patch.diff"
            patch = single.read_text() if single.exists() else ""
            best, scores = 0, []
        else:
            best, scores = select_best(instance, candidates, run_dir / iid / "reselect")
            patch = candidates[best]
            (run_dir / iid / "patch.diff").write_text(patch)
            (run_dir / iid / "selection.json").write_text(json.dumps({"best": best, "scores": scores}, indent=2))
        preds[iid] = {
            "instance_id": iid,
            "model_name_or_path": f"proto-harness-{args.run_id}",
            "model_patch": patch,
        }
        print(f"  {iid}: attempt {best} ({[s.get('failures') for s in scores]})")
    (run_dir / "preds.json").write_text(json.dumps(preds, indent=2))
    print(f"wrote {run_dir / 'preds.json'}")


if __name__ == "__main__":
    main()
