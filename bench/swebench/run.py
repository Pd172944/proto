"""Run proto-harness (`proto code`) against the fixed SWE-bench Verified set.

Per instance: a scratch checkout is cloned from the local mirror at the base
commit, the agent runs one-shot with the issue text, the resulting worktree
diff becomes the model patch, and per-instance metrics are logged. Output for
a run lands in ``BASE/runs/<run_id>/``:

- ``<instance_id>/repo``      the scratch worktree the agent edited
- ``<instance_id>/agent.out`` agent stdout (answer + trailing CLI JSON)
- ``<instance_id>/agent.err`` agent stderr (tool chips, routing chatter)
- ``preds.json``              sb-cli-shaped predictions
- ``metrics.jsonl``           one row per instance

Example:
    $ python run.py --run-id iter01 --mode route --workers 5
"""

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline import (
    adjacent_on_worktree,
    cloud_repair,
    constraint_block,
    local_verify,
    parse_scout_files,
)

BASE: Path = Path("/data/prithvi/sweb")
PROTO: Path = Path("/data/prithvi/proto")
PROTO_HOME_MAIN: Path = Path("/data/prithvi/proto-home")
HOME: str = "/data/prithvi/home"

PROMPT_TEMPLATE: str = """You are fixing a real reported issue in the open-source project {repo} (checked out at the commit where the issue reproduces). Fix it.

<issue>
{problem}
</issue>

Work in this order:
1. Locate the relevant code with find_symbol / repo_map / file_outline (prefer find_symbol over a text search for function names), and read enough context to understand the actual cause.
2. {repro_hint}
3. Make the smallest correct change that resolves the issue in the production source. Never modify tests, never add new test files. Match the project's existing style. Reproduction scripts belong in /tmp — never create repro.py / reproduce.py / apply_fix.py in the repo. Writing a reproduction script is not a fix.
4. Re-run the same reproduction. The issue is not fixed until that script now shows the expected behavior from the report (not merely "it no longer crashes").
5. Cover the edge cases the report names (nested includes, commas inside regex quantifiers, userinfo in URLs, MRO of base classes, string vs numeric constructors). Do not stop at the first plausible one-line change.
6. Never run a full test suite (`pytest` with no path, `tests/runtests.py` with no labels). Targeted files only, `timeout_ms` ≤ 60000.
7. Stop and summarize the change in 2-3 sentences.

Do not stop at analysis: the task is only done when the source is edited and verified."""

TEST_HINT_ENV: str = (
    "Write a minimal reproduction script (in /tmp, not the repo) from the issue's example and run it with `python` "
    "to see the broken behavior first — the project is installed in the Python environment on PATH."
)
TEST_HINT_NOENV: str = "Reason carefully about the cause; a runnable environment is not available, so read the code paths end to end."

SCOUT_PROMPT: str = """You are scouting a real issue in {repo}. Do NOT edit anything. Your job is only to locate the change site.

<issue>
{problem}
</issue>

Use repo_map, find_symbol, search, and short read_file ranges. Then reply with exactly this shape:

LIKELY_FILES:
- path/to/file.py
SUSPECTED_SYMBOLS:
- Class.method or function_name
CAUSE:
one or two sentences naming the actual bug, not a restatement of the issue.
"""


def git(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a git command in ``cwd`` with output captured.

    Args:
        args: Git arguments (without the leading ``git``).
        cwd: Repository directory.

    Returns:
        Completed process.
    """
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=300)


def extract_trailing_json(stdout: str) -> Optional[dict]:
    """Extract the pretty-printed JSON document the CLI appends to stdout.

    Args:
        stdout: Full agent stdout (answer text followed by the ``--json`` doc).

    Returns:
        Parsed JSON dict, or None if not found.
    """
    idx: int = stdout.rfind("\n{")
    while idx != -1:
        candidate: str = stdout[idx + 1 :]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            idx = stdout.rfind("\n{", 0, idx)
    return None


def parse_routing(stderr: str) -> Dict[str, object]:
    """Pull routing decision and per-turn status facts out of agent stderr.

    Args:
        stderr: The agent's stderr stream (chatter in --print mode).

    Returns:
        Dict with tier / p_local / tool call counts when present.
    """
    out: Dict[str, object] = {}
    m = re.search(r"routing\s+(\S+)\s+p\(local\)=([\d.]+)", strip_ansi(stderr))
    if m:
        out["tier"] = m.group(1)
        out["p_local"] = float(m.group(2))
    plain = strip_ansi(stderr)
    out["tool_calls"] = len(re.findall(r"\b(?:ok|fail)\s+\w+ ", plain))
    out["escalated"] = "escalated to" in plain
    return out


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences.

    Args:
        text: Raw terminal text.

    Returns:
        Plain text.
    """
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


JUNK_FILE: re.Pattern[str] = re.compile(
    r"(^|/)(repro|reproduce|tmp_|scratch|diag|apply_fix|foo)[^/]*\.(py|sh)$",
    re.I,
)
READ_CHIP: re.Pattern[str] = re.compile(r"\bread (\S+\.(?:py|c|h|js|ts|go|rs))\b")
SYMBOL_CHIP: re.Pattern[str] = re.compile(r"\bfind_symbol (\S+) →")


def salvage_scout_note(stdout: str, stderr: str) -> str:
    """Build a scout note from the model's answer, falling back to tool chips.

    The local scout often hits its step budget after reading the right files
    and never emits the LIKELY_FILES block. The chips in stderr still name
    those files; injecting them is what makes the cloud fixer start at the
    change site instead of scanning dataset.py from line 1.

    Args:
        stdout: Scout ``--print`` stdout (answer + trailing JSON).
        stderr: Scout tool-chip stream.

    Returns:
        A short note, possibly empty.
    """
    text: str = stdout
    idx: int = text.rfind("\n{")
    if idx >= 0:
        text = text[:idx]
    text = text.strip()
    if text.startswith("{"):
        text = ""

    files: List[str] = []
    for match in READ_CHIP.finditer(strip_ansi(stderr)):
        path: str = match.group(1).split(":")[0]
        if JUNK_FILE.search(path):
            continue
        if "/test" in path or path.startswith("test"):
            continue
        if path not in files:
            files.append(path)
    symbols: List[str] = []
    for match in SYMBOL_CHIP.finditer(strip_ansi(stderr)):
        name: str = match.group(1)
        if name not in symbols:
            symbols.append(name)

    if "LIKELY_FILES" in text or "CAUSE" in text:
        parts: List[str] = [text[-2000:]]
        if files:
            parts.append("OBSERVED_READS:\n" + "\n".join(f"- {p}" for p in files[:8]))
        return "\n".join(parts)

    parts = []
    if files:
        parts.append("LIKELY_FILES:\n" + "\n".join(f"- {p}" for p in files[:8]))
    if symbols:
        parts.append("SUSPECTED_SYMBOLS:\n" + "\n".join(f"- {s}" for s in symbols[:8]))
    if text and len(text) >= 80:
        parts.append(text[-1200:])
    return "\n".join(parts)


def scrub_junk_files(repo_dir: Path) -> None:
    """Delete agent-written repro scripts at the repository root.

    Args:
        repo_dir: Agent worktree.
    """
    for path in repo_dir.iterdir():
        if path.is_file() and JUNK_FILE.search(path.name):
            path.unlink()


def collect_model_patch(repo_dir: Path) -> str:
    """Collect a production-source-only unified diff from ``repo_dir``.

    Agents often write reproduction scripts into the worktree (``repro.py``,
    ``reproduce.py``). Those must not become the SWE-bench prediction: the
    official eval applies the patch onto a clean checkout, so a repro-only
    diff is a guaranteed fail.

    Args:
        repo_dir: Agent worktree.

    Returns:
        Filtered unified diff (may be empty).
    """
    git(["add", "-N", "."], repo_dir)
    raw: str = git(["diff"], repo_dir).stdout
    chunks: List[str] = re.split(r"(?=^diff --git )", raw, flags=re.M)
    kept: List[str] = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        m = re.search(r"^diff --git a/(\S+) b/(\S+)", chunk, flags=re.M)
        if not m:
            continue
        path: str = m.group(2)
        if JUNK_FILE.search(path):
            continue
        if "/test" in path or path.startswith("test") or "/tests/" in path:
            continue
        if not path.endswith((".py", ".c", ".h", ".cpp", ".js", ".ts", ".go", ".rs")):
            continue
        kept.append(chunk if chunk.endswith("\n") else chunk + "\n")
    return "".join(kept)


def scout_localize(instance: dict, inst_dir: Path, proto_home: Path, env: Dict[str, str]) -> str:
    """Run a short local-model scout to name likely files and the cause.

    The scout is read-only and bounded (10 steps, 3 minutes) so it stays cheap
    and cannot dirty the worktree the main agent will edit. Its notes are
    injected into the main prompt so the cloud/local fixer does not spend its
    first ten steps rediscovering the change site.

    Args:
        instance: Instance row.
        inst_dir: Per-attempt directory (scout artifacts land here).
        proto_home: Isolated PROTO_HOME for this attempt.
        env: Environment inherited by the main agent.

    Returns:
        A short note to prepend to the main prompt, or empty string on failure.
    """
    repo_dir: Path = inst_dir / "repo"
    prompt: str = SCOUT_PROMPT.format(repo=instance["repo"], problem=instance["problem_statement"].strip()[:4000])
    cmd: List[str] = [
        str(PROTO / "bin" / "proto"),
        "code",
        prompt,
        "--workspace",
        str(repo_dir),
        "--local",
        "--read-only",
        "--print",
        "--yes",
        "--json",
        "--no-save",
        "--max-steps",
        "14",
        "--deadline-min",
        "3",
    ]
    try:
        proc = subprocess.run(cmd, cwd=repo_dir, env=env, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return ""
    (inst_dir / "scout.out").write_text(proc.stdout)
    (inst_dir / "scout.err").write_text(proc.stderr)
    note: str = salvage_scout_note(proc.stdout, proc.stderr)
    (inst_dir / "scout.note.txt").write_text(note)
    return note


def run_instance(instance: dict, run_dir: Path, args: argparse.Namespace, attempt: int = 0) -> dict:
    """Run the agent once on one instance and return its metrics row.

    Args:
        instance: Row from ``instances.json``.
        run_dir: Directory for this run.
        args: Parsed CLI arguments.
        attempt: Attempt index (attempts are independent agent runs whose
            patches compete in selection).

    Returns:
        Metrics dict (also contains the model patch length; the patch itself
        goes into preds.json).
    """
    iid: str = instance["instance_id"]
    inst_dir: Path = run_dir / iid if args.attempts == 1 else run_dir / iid / f"a{attempt}"
    repo_dir: Path = inst_dir / "repo"
    inst_dir.mkdir(parents=True, exist_ok=True)

    mirror: Path = BASE / "repos" / (instance["repo"].replace("/", "__") + ".git")
    if repo_dir.exists():
        subprocess.run(["rm", "-rf", str(repo_dir)], check=True)
    res = subprocess.run(
        ["git", "clone", "--shared", "-q", str(mirror), str(repo_dir)], capture_output=True, text=True, timeout=600
    )
    assert res.returncode == 0, f"clone failed: {res.stderr}"
    res = git(["checkout", "-q", instance["base_commit"]], repo_dir)
    assert res.returncode == 0, f"checkout failed: {res.stderr}"

    # Isolated PROTO_HOME per instance so episodes/transcripts do not interleave;
    # config and secrets are inherited by copying from the main home.
    proto_home: Path = inst_dir / "proto-home"
    proto_home.mkdir(exist_ok=True)
    for name in ("config.json", "secrets.json"):
        src = PROTO_HOME_MAIN / name
        if src.exists():
            (proto_home / name).write_bytes(src.read_bytes())
            os.chmod(proto_home / name, 0o600)

    env: Dict[str, str] = {
        **os.environ,
        "HOME": HOME,
        "PROTO_HOME": str(proto_home),
        "NO_COLOR": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        # OpenRouter gates some free models on the calling app's identity;
        # present as a registered agentic-harness app.
        "PROTO_OR_REFERER": "https://cline.bot",
        "PROTO_OR_TITLE": "Cline",
    }
    env_dir: Path = BASE / "envs" / iid
    has_env: bool = (env_dir / ".ready").exists()
    if has_env:
        env["PATH"] = f"{env_dir}/bin:{env['PATH']}"
        # The venv's editable install points at pristine source; the agent's
        # edits must be what its own test runs and repro scripts import.
        env["PYTHONPATH"] = str(repo_dir)

    scout_note: str = ""
    if args.scout:
        if attempt == 0:
            scout_note = scout_localize(instance, inst_dir, proto_home, env)
        else:
            prev = run_dir / instance["instance_id"] / "a0" / "scout.note.txt"
            if prev.exists():
                scout_note = prev.read_text()[-2500:]

    scout_paths: List[str] = parse_scout_files(scout_note)
    prompt: str = PROMPT_TEMPLATE.format(
        repo=instance["repo"],
        problem=instance["problem_statement"].strip(),
        repro_hint=TEST_HINT_ENV if has_env else TEST_HINT_NOENV,
    )
    if scout_note:
        prompt = (
            constraint_block(scout_paths)
            + "A local scout already inspected the repository (read-only). "
            "Start at the named files; only leave them if you can say why the scout was wrong.\n\n"
            f"<scout>\n{scout_note}\n</scout>\n\n"
            + prompt
        )

    cmd: List[str] = [
        str(PROTO / "bin" / "proto"),
        "code",
        prompt,
        "--workspace",
        str(repo_dir),
        "--print",
        "--yes",
        "--json",
        "--no-save",
        "--max-steps",
        str(args.max_steps),
        "--deadline-min",
        str(args.deadline_min),
        "--budget-usd",
        str(args.budget_usd),
    ]
    # Attempt 0 follows --mode. Extra attempts are cloud resamples at higher
    # temperature. Local is reserved for the scout (localization), not for
    # competing patches: a small local diff that happens not to break adjacent
    # tests was beating complete cloud patches in selection.
    if attempt == 0:
        if args.mode == "local":
            cmd.append("--local")
        elif args.mode == "cloud":
            cmd.append("--no-route")
        else:
            cmd.append("--escalate-on-stuck")
    else:
        cmd.extend(["--no-route", "--temperature", str(0.4 + 0.2 * (attempt % 3))])

    start: float = time.time()
    try:
        proc = subprocess.run(
            # Escalation can add a continuation turn, so allow two turn budgets.
            cmd, cwd=repo_dir, env=env, capture_output=True, text=True, timeout=args.deadline_min * 60 + 180
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        proc = None
        timed_out = True
        (inst_dir / "agent.out").write_text((exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
        (inst_dir / "agent.err").write_text((exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
    wall_s: float = time.time() - start

    stats: Optional[dict] = None
    if proc is not None:
        (inst_dir / "agent.out").write_text(proc.stdout)
        (inst_dir / "agent.err").write_text(proc.stderr)
        stats = extract_trailing_json(proc.stdout)

    # Production-source diff only: drop repro scripts the agent left in-tree.
    patch: str = collect_model_patch(repo_dir)

    # If the model only wrote a repro script, one follow-up turn on the same
    # worktree asking it to actually edit library source. Cheap, and it is the
    # observed failure on xarray-style tasks.
    if not patch.strip() and not timed_out:
        scrub_junk_files(repo_dir)
        follow_prompt: str = (
            "You wrote a reproduction script but did not change any production source. "
            "The issue is still present. Use find_symbol on the names in the issue, then "
            "edit_file on an existing library module (not tests, not a new file at repo root) "
            "so the reported behavior is fixed, then stop."
        )
        if scout_note:
            follow_prompt = (
                constraint_block(scout_paths)
                + "A local scout already named likely files. Start there.\n\n"
                f"<scout>\n{scout_note}\n</scout>\n\n"
                + follow_prompt
            )
        follow: List[str] = [
            str(PROTO / "bin" / "proto"),
            "code",
            follow_prompt,
            "--workspace",
            str(repo_dir),
            "--print",
            "--yes",
            "--json",
            "--no-save",
            "--no-route",
            "--max-steps",
            "25",
            "--deadline-min",
            "5",
        ]
        try:
            proc2 = subprocess.run(follow, cwd=repo_dir, env=env, capture_output=True, text=True, timeout=300)
            (inst_dir / "agent.follow.out").write_text(proc2.stdout)
            (inst_dir / "agent.follow.err").write_text(proc2.stderr)
            if proc is not None:
                proc = proc2
            stats = extract_trailing_json(proc2.stdout) or stats
        except subprocess.TimeoutExpired:
            pass
        patch = collect_model_patch(repo_dir)
        wall_s = time.time() - start

    repaired: bool = False
    verify_passed: Optional[bool] = None
    if (
        args.scout
        and not args.no_verify
        and has_env
        and patch.strip()
        and not timed_out
    ):
        adj_ok: bool
        adj_log: str
        try:
            adj_ok, adj_log = adjacent_on_worktree(instance, repo_dir, env, patch)
        except subprocess.TimeoutExpired:
            adj_ok, adj_log = True, "adjacent tests timed out; skipped"
        (inst_dir / "verify.adjacent.txt").write_text(adj_log)
        v_ok: bool
        v_log: str
        v_ok, v_log = local_verify(instance, repo_dir, env, patch, inst_dir)
        verify_passed = adj_ok and v_ok
        if not verify_passed:
            evidence: str = (
                "ADJACENT TESTS:\n" + adj_log + "\n\nLOCAL REPRO:\n" + v_log
            )
            (inst_dir / "verify.evidence.txt").write_text(evidence)
            cloud_repair(instance, repo_dir, env, scout_paths, evidence, inst_dir)
            scrub_junk_files(repo_dir)
            patch = collect_model_patch(repo_dir)
            repaired = True
            wall_s = time.time() - start

    combined: str = "".join(
        (inst_dir / name).read_text() for name in ("agent.out", "agent.err") if (inst_dir / name).exists()
    )
    routing: Dict[str, object] = parse_routing(combined)
    session_stats: dict = (stats or {}).get("stats", {}) if isinstance(stats, dict) else {}
    row: dict = {
        "instance_id": iid,
        "attempt": attempt,
        "wall_s": round(wall_s, 1),
        "timed_out": timed_out,
        "exit_ok": bool(stats and stats.get("ok")),
        "patch_bytes": len(patch),
        "cost_usd": session_stats.get("costUsd"),
        "steps": session_stats.get("steps"),
        "input_tokens": session_stats.get("inputTokens"),
        "cached_input_tokens": session_stats.get("cachedInputTokens"),
        "output_tokens": session_stats.get("outputTokens"),
        "scout_files": scout_paths,
        "verify_passed": verify_passed,
        "repaired": repaired,
        **routing,
    }
    (inst_dir / "patch.diff").write_text(patch)
    extra: str = ""
    if repaired:
        extra = " repaired"
    elif verify_passed is False:
        extra = " verify-fail"
    elif verify_passed is True:
        extra = " verified"
    print(
        f"  {iid}: {'TIMEOUT' if timed_out else 'ok' if row['exit_ok'] else 'ERR'} "
        f"patch={len(patch)}B cost=${session_stats.get('costUsd', 0) or 0:.3f} "
        f"steps={session_stats.get('steps')} tier={routing.get('tier', '?')} {wall_s:.0f}s{extra}"
    )
    return row


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--instances", type=Path, default=Path(__file__).parent / "instances.json")
    parser.add_argument("--mode", choices=["route", "local", "cloud"], default="route")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--deadline-min", type=float, default=12)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--only", type=str, default=None, help="comma-separated instance ids")
    parser.add_argument("--scout", action="store_true", help="local localization scout, then local verify + cloud repair on failure")
    parser.add_argument("--no-verify", action="store_true", help="with --scout, skip the local verify / cloud repair phases")
    args = parser.parse_args()

    instances: List[dict] = json.loads(args.instances.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        instances = [i for i in instances if i["instance_id"] in wanted]

    run_dir: Path = BASE / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {args.run_id}: {len(instances)} instances, mode={args.mode}, attempts={args.attempts}, workers={args.workers}")

    jobs: List[Tuple[dict, int]] = [(i, a) for i in instances for a in range(args.attempts)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows: List[dict] = list(pool.map(lambda j: run_instance(j[0], run_dir, args, j[1]), jobs))

    # Selection: with several attempts per instance, the adjacent-regression
    # check picks the survivor (see select_patch.py).
    from select_patch import select_best

    preds: Dict[str, dict] = {}
    for instance in instances:
        iid = instance["instance_id"]
        if args.attempts == 1:
            patch = (run_dir / iid / "patch.diff").read_text()
        else:
            candidates: List[str] = [
                (run_dir / iid / f"a{a}" / "patch.diff").read_text() for a in range(args.attempts)
            ]
            best, score_rows = select_best(instance, candidates, run_dir / iid)
            patch = candidates[best]
            (run_dir / iid / "selection.json").write_text(json.dumps({"best": best, "scores": score_rows}, indent=2))
            (run_dir / iid / "patch.diff").write_text(patch)
            print(f"  select {iid}: attempt {best} of {len(candidates)} ({[r['failures'] for r in score_rows]})")
        preds[iid] = {
            "instance_id": iid,
            "model_name_or_path": f"proto-harness-{args.run_id}",
            "model_patch": patch,
        }
    (run_dir / "preds.json").write_text(json.dumps(preds, indent=2))
    with (run_dir / "metrics.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    total_cost = sum(r["cost_usd"] or 0 for r in rows)
    nonempty = sum(1 for r in rows if r["patch_bytes"] > 0)
    print(f"\ndone: {nonempty}/{len(rows)} non-empty patches across attempts, total cost ${total_cost:.2f}")
    print(f"predictions: {run_dir / 'preds.json'}")


if __name__ == "__main__":
    main()
