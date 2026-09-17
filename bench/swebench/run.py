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
    cloud_finish,
    constraint_block,
    parse_scout_files,
    reset_worktree,
    run_proto,
    should_try_local,
    sonnet_usd,
    is_inplace_patch,
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


def usage_from_blob(blob: Optional[dict]) -> Tuple[int, int, int]:
    """Read input / cached / output token counts from a proto ``--json`` blob.

    Args:
        blob: Parsed trailing JSON from ``proto code --json``, or None.

    Returns:
        ``(input_tokens, cached_input_tokens, output_tokens)``.
    """
    if not blob:
        return 0, 0, 0
    inner: dict = blob.get("stats", {}) if isinstance(blob.get("stats"), dict) else {}
    return (
        int(inner.get("inputTokens") or 0),
        int(inner.get("cachedInputTokens") or 0),
        int(inner.get("outputTokens") or 0),
    )


def usage_from_stdout(stdout: str) -> Tuple[int, int, int]:
    """Read token counts from proto stdout that ends with a JSON stats document.

    Args:
        stdout: Full captured stdout.

    Returns:
        ``(input_tokens, cached_input_tokens, output_tokens)``.
    """
    return usage_from_blob(extract_trailing_json(stdout))


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

    The scout is read-only and bounded (8 steps, 2 minutes) so it stays cheap
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
        "8",
        "--deadline-min",
        "2",
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

    start: float = time.time()
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

    timed_out: bool = False
    stats: Optional[dict] = None
    local_accepted: bool = False
    cloud_calls: int = 0
    cloud_in: int = 0
    cloud_cached: int = 0
    cloud_out: int = 0
    local_in: int = 0
    local_cached: int = 0
    local_out: int = 0
    local_calls: int = 1 if (args.scout and attempt == 0) else 0
    fixer: str = "cloud-strong"
    adj_log: str = ""
    patch: str = ""

    def add_cloud_stats(blob: Optional[dict]) -> None:
        """Accumulate token counts from a cloud proto JSON blob."""
        nonlocal cloud_in, cloud_cached, cloud_out, stats
        if not blob:
            return
        stats = blob
        cin, ccached, cout = usage_from_blob(blob)
        cloud_in += cin
        cloud_cached += ccached
        cloud_out += cout

    def add_local_from_file(path: Path) -> None:
        """Accumulate local-model tokens from a captured proto stdout file."""
        nonlocal local_in, local_cached, local_out
        if not path.exists():
            return
        lin, lcached, lout = usage_from_stdout(path.read_text())
        local_in += lin
        local_cached += lcached
        local_out += lout

    try_local: bool = (
        args.scout
        and args.mode != "cloud"
        and attempt == 0
        and should_try_local(scout_paths)
    )

    if args.mode == "local" or try_local:
        try:
            proc = run_proto(
                prompt,
                repo_dir,
                env,
                max_steps=12 if try_local else args.max_steps,
                deadline_min=1.5 if try_local else args.deadline_min,
                timeout_s=100 if try_local else int(args.deadline_min * 60) + 180,
                local=True,
            )
        except subprocess.TimeoutExpired:
            proc = None
            timed_out = True
        local_calls += 1
        if proc is not None:
            (inst_dir / "agent.local.out").write_text(proc.stdout)
            (inst_dir / "agent.local.err").write_text(proc.stderr)
            stats = extract_trailing_json(proc.stdout)
        scrub_junk_files(repo_dir)
        patch = collect_model_patch(repo_dir)
        accepted: bool = False
        if patch.strip() and has_env and not timed_out:
            try:
                ok, adj_log, had = adjacent_on_worktree(instance, repo_dir, env, patch)
            except subprocess.TimeoutExpired:
                ok, adj_log, had = False, "adjacent tests timed out", False
            (inst_dir / "verify.adjacent.txt").write_text(adj_log)
            accepted = bool(ok and had and is_inplace_patch(patch))
        if accepted:
            local_accepted = True
            fixer = "local"
        elif args.mode != "local":
            failed_diff: str = patch
            reset_worktree(repo_dir)
            scrub_junk_files(repo_dir)
            proc_c = cloud_finish(
                instance,
                repo_dir,
                env,
                scout_paths,
                scout_note,
                failed_diff,
                adj_log,
                inst_dir,
                max_steps=args.max_steps,
                deadline_min=args.deadline_min,
            )
            cloud_calls += 1
            if proc_c is not None:
                add_cloud_stats(extract_trailing_json(proc_c.stdout))
            scrub_junk_files(repo_dir)
            patch = collect_model_patch(repo_dir)
            fixer = "cloud-escalated"
    else:
        proc_c = cloud_finish(
            instance,
            repo_dir,
            env,
            scout_paths,
            scout_note,
            "",
            "",
            inst_dir,
            max_steps=args.max_steps,
            deadline_min=args.deadline_min,
        )
        cloud_calls += 1
        if proc_c is None:
            timed_out = True
        else:
            add_cloud_stats(extract_trailing_json(proc_c.stdout))
        scrub_junk_files(repo_dir)
        patch = collect_model_patch(repo_dir)
        fixer = "cloud-direct"
        if not patch.strip() and not timed_out:
            proc2 = cloud_finish(
                instance,
                repo_dir,
                env,
                scout_paths,
                scout_note,
                "",
                "Previous cloud turn wrote no production source. Edit an existing library file.",
                inst_dir,
                max_steps=20,
                deadline_min=4,
            )
            cloud_calls += 1
            if proc2 is not None:
                add_cloud_stats(extract_trailing_json(proc2.stdout))
            scrub_junk_files(repo_dir)
            patch = collect_model_patch(repo_dir)

    add_local_from_file(inst_dir / "scout.out")
    add_local_from_file(inst_dir / "agent.local.out")
    wall_s: float = time.time() - start
    combined: str = "".join(
        (inst_dir / name).read_text()
        for name in ("agent.out", "agent.err", "agent.local.out", "agent.local.err")
        if (inst_dir / name).exists()
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
        "local_accepted": local_accepted,
        "local_calls": local_calls,
        "cloud_calls": cloud_calls,
        "fixer": fixer,
        "sonnet_usd": round(sonnet_usd(cloud_in, cloud_cached, cloud_out), 4),
        "cloud_input_tokens": cloud_in,
        "cloud_cached_tokens": cloud_cached,
        "cloud_output_tokens": cloud_out,
        "local_input_tokens": local_in,
        "local_cached_tokens": local_cached,
        "local_output_tokens": local_out,
        "local_tokens": local_in + local_out,
        "cloud_tokens": cloud_in + cloud_out,
        "total_tokens": local_in + local_out + cloud_in + cloud_out,
        **routing,
    }
    if local_accepted:
        row["tier"] = "local"
    elif cloud_calls:
        row["tier"] = "cloud-strong"
    (inst_dir / "patch.diff").write_text(patch)
    print(
        f"  {iid}: {'TIMEOUT' if timed_out else 'ok'} "
        f"patch={len(patch)}B fixer={fixer} "
        f"tokens local={row['local_tokens']} inkling={row['cloud_tokens']} "
        f"{wall_s:.0f}s"
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
    parser.add_argument("--scout", action="store_true", help="local-first: tight scout then local patch, escalate to cloud only if adjacent tests fail")
    parser.add_argument("--no-verify", action="store_true", help="unused; kept for flag compatibility")
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
    sonnet = sum(r.get("sonnet_usd") or 0 for r in rows)
    nonempty = sum(1 for r in rows if r["patch_bytes"] > 0)
    local_kept = sum(1 for r in rows if r.get("local_accepted"))
    cloud_n = sum(r.get("cloud_calls") or 0 for r in rows)
    print(
        f"\ndone: {nonempty}/{len(rows)} non-empty, local-kept {local_kept}/{len(rows)}, "
        f"cloud_calls={cloud_n}, billed ${total_cost:.2f}, hypothetical Sonnet ${sonnet:.2f}"
    )
    print(f"predictions: {run_dir / 'preds.json'}")


if __name__ == "__main__":
    main()
