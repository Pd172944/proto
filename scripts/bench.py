#!/usr/bin/env python3
"""
Agentic benchmark suite for `proto code`.

Read this before trusting a number that comes out of it.

**Why this exists.** HumanEval measures a completion model: one function, one file,
no repository, no tests to run. Terminal-Bench and SWE-bench measure an *agent*: a
real environment, a hidden verifier, and freedom to iterate. Both of those run every
task in a Docker image, which is what makes them reproducible and which is also why
they cannot run on a machine without Docker.

This suite is the shape of Terminal-Bench with none of the infrastructure: each task
is a directory of files, a prompt, and a verifier script that lives *outside* the
workspace. No containers, no network, no per-task image pulls. That makes it a
measurement of the harness and the model rather than of a container registry.

**What it does and does not tell you.**

  - It tells you: does the agent loop actually finish real multi-step work, how long
    does a tier take, and does the local tier succeed on anything at all.
  - It does not tell you how you compare to published SWE-bench numbers. Ten
    hand-written tasks are not a leaderboard, and the intervals around a 10-sample
    rate are enormous — 6/10 has a 95% interval of roughly 26-76%. Treat it as a
    signal about the harness, not a score about the model.
  - Tasks are written to be *solvable* by a competent small model. A suite that
    floors at 0/10 teaches nothing, so several tasks are deliberately easy. If
    everything passes, the suite is too easy and needs harder tasks, not a
    victory lap.

**Isolation.** Each run gets a fresh copy of the task workspace, a fresh `PROTO_HOME`,
and a PATH whose first entry is a virtualenv built from the system interpreter. That
last part matters more than it sounds: without it the agent inherits whatever is on
your machine, and a broken pytest plugin chain can eat a third of its step budget
before it ever touches the task. A benchmark that measures your shell profile is
measuring the wrong thing.

**Grading honesty.** The verifier is never the workspace's own test file. Visible
tests may live in the workspace so the agent can iterate, but grading copies the
hidden tests in from the task directory and runs those. Deleting or weakening a test
in the workspace therefore cannot earn a pass.

Usage:
    python3 scripts/bench.py list [--suite NAME]
    python3 scripts/bench.py validate [--suite NAME]     # prove every task is honest
    python3 scripts/bench.py run --tier local [--suite NAME] [--only ID]...
                                   [--model M] [--max-steps N] [--timeout SEC]
                                   [--tag TAG] [--label TEXT]
    python3 scripts/bench.py report [--results PATH]

Typical first session:
    python3 scripts/bench.py validate          # every task must FAIL before any work
    python3 scripts/bench.py run --tier local  # the number you actually care about
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SUITE = "agentic-10"
DEFAULT_RUN_ROOT = Path("/tmp/proto-bench")

# Environment noise that has to go before anything is measured. `PYTEST_DISABLE_PLUGIN_AUTOLOAD`
# is the important one: a third-party plugin that fails to import aborts every pytest
# invocation, and the agent then spends its budget debugging your site-packages instead
# of the task. Found the hard way.
SANITIZED_ENV = {
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "NO_COLOR": "1",
    "PROTO_LOG": "silent",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
}


# --------------------------------------------------------------------------- tasks


@dataclass
class Task:
    id: str
    path: Path
    prompt: str
    meta: dict
    workspace: Path
    verify: Path
    solution: Path | None
    setup: Path | None = None
    solution_script: Path | None = None

    @property
    def tags(self) -> list[str]:
        return list(self.meta.get("tags", []))

    @property
    def timeout_sec(self) -> int:
        return int(self.meta.get("timeoutSec", 420))

    @property
    def hidden_tests(self) -> list[Path]:
        d = self.path / "hidden"
        return sorted(d.glob("*")) if d.is_dir() else []


def load_suite(suite: str) -> list[Task]:
    root = REPO / "bench" / "suite" / suite
    if not root.is_dir():
        raise SystemExit(f"no such suite: {root}")
    tasks: list[Task] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        prompt_file = d / "task.md"
        verify = d / "verify.sh"
        ws = d / "workspace"
        # An incomplete task is skipped loudly rather than aborting the suite: tasks get
        # authored incrementally, and one half-written directory should not make the
        # other nine unrunnable. The skip is visible in the task count.
        missing = [
            name
            for name, present in (("task.md", prompt_file.is_file()), ("verify.sh", verify.is_file()), ("workspace/", ws.is_dir()))
            if not present
        ]
        if missing:
            print(f"warning: skipping {d.name} — missing {', '.join(missing)}", file=sys.stderr)
            continue
        meta_file = d / "meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
        solution = d / "solution" if (d / "solution").is_dir() else None
        setup = d / "setup.sh" if (d / "setup.sh").is_file() else None
        solution_script = d / "solution.sh" if (d / "solution.sh").is_file() else None
        tasks.append(
            Task(
                id=d.name,
                path=d,
                prompt=prompt_file.read_text().strip(),
                meta=meta,
                workspace=ws,
                verify=verify,
                solution=solution,
                setup=setup,
                solution_script=solution_script,
            )
        )
    return tasks


# ------------------------------------------------------------------------ running


def build_venv(run_root: Path, quiet: bool = False) -> tuple[Path, str]:
    """
    A virtualenv the agent and the verifier share, so `python` resolves to the same
    interpreter for both. The agent inventing `python3 -m pytest` and hitting an
    unrelated plugin crash is a harness failure, not a model failure.

    Returns the venv and a warning string. The warning matters: `sys.executable` is
    whatever interpreter launched this script, and on a machine with several Pythons
    that is not necessarily the one with pytest. Discovering that *after* a run means
    reading "0/10 solved" as a model result when it is a broken verifier — the exact
    way a benchmark lies. So the dependency is checked here, installed if it can be,
    and reported if it cannot.
    """
    venv = run_root / "venv"
    py = venv / "bin" / "python"
    if not py.exists():
        if not quiet:
            print(f"  building shared virtualenv at {venv}")
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
            check=True,
            capture_output=True,
            text=True,
        )

    probe = subprocess.run([str(py), "-c", "import pytest"], capture_output=True, text=True)
    if probe.returncode == 0:
        return venv, ""

    # Inherited site-packages did not provide pytest. Install it into the venv rather
    # than failing: every task's verifier uses pytest, so without it the suite is not
    # runnable at all.
    if not quiet:
        print(f"  installing pytest into {venv} (the launching interpreter lacks it)")
    install = subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "pytest"],
        capture_output=True,
        text=True,
    )
    if install.returncode == 0 and subprocess.run([str(py), "-c", "import pytest"], capture_output=True).returncode == 0:
        return venv, ""

    return venv, (
        f"the verifier environment has no pytest and it could not be installed "
        f"(launching interpreter: {sys.executable}). Every task would be reported as "
        f"failed regardless of the model. Fix with: "
        f"{py} -m pip install pytest"
    )


def base_env(venv: Path | None) -> dict:
    env = dict(os.environ)
    env.update(SANITIZED_ENV)
    if venv is not None and (venv / "bin").is_dir():
        # Prepend, so a clean `python` shadows whatever the user's PATH resolves to.
        env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(venv)
    return env


def materialize(task: Task, run_dir: Path, run_root: Path) -> tuple[Path, Path]:
    """Copy the task's initial state into a private workspace plus a private home."""
    ws = run_dir / task.id / "ws"
    home = run_dir / task.id / "home"
    for p in (ws, home):
        if p.exists():
            shutil.rmtree(p)
    shutil.copytree(task.workspace, ws)
    home.mkdir(parents=True)
    # A private config keeps episodes and sessions from bleeding across tasks, and
    # pins the model so a stale global config cannot change what is being measured.
    if task.setup is not None:
        # Git-based tasks cannot ship a `.git` directory: a nested repository inside
        # the harness repo is a gitlink, not a directory of files. Building the state
        # from a script is also closer to how Terminal-Bench works, where a setup
        # step constructs the environment before the agent is handed the task.
        run_setup(task, ws)
    return ws, home


def run_setup(task: Task, ws: Path, quiet: bool = True) -> None:
    """Construct the task's initial state inside the fresh workspace."""
    assert task.setup is not None
    env = base_env(None)
    env["BENCH_WORKSPACE"] = str(ws)
    res = subprocess.run(
        ["bash", str(task.setup)],
        cwd=str(ws),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0 and not quiet:
        print(f"  setup for {task.id} failed:\n{res.stdout}{res.stderr}")


def run_verifier(task: Task, ws: Path, env: dict, timeout: int = 180) -> tuple[bool, str]:
    """
    Run the hidden verifier against the workspace. The verifier is invoked from a
    temp directory with the hidden tests copied in, so nothing the agent did to the
    workspace's own test files can influence the outcome.
    """
    scratch = ws.parent / "_verify"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    if task.hidden_tests:
        for f in task.hidden_tests:
            shutil.copy2(f, scratch / f.name)
    venv = dict(env)
    venv["BENCH_WORKSPACE"] = str(ws)
    venv["BENCH_VERIFY_DIR"] = str(scratch)
    venv["PYTHONPATH"] = f"{ws}{os.pathsep}{scratch}"
    try:
        res = subprocess.run(
            ["bash", str(task.verify)],
            cwd=str(scratch),
            env=venv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"verifier timed out after {timeout}s"
    output = (res.stdout + res.stderr).strip()
    return res.returncode == 0, output


def apply_solution(task: Task, ws: Path) -> bool:
    """
    Put the workspace into the solved state.

    `solution/` holds files; `solution.sh` holds *operations*. A task can need both and
    the order is not arbitrary: the files go in first, because the commands may stage
    them (a `.gitignore` that then has to be committed does not exist until the copy
    has happened).
    """
    copied = False
    if task.solution is not None:
        for src in task.solution.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(task.solution)
            dst = ws / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied = True

    if task.solution_script is not None:
        env = base_env(None)
        env["BENCH_WORKSPACE"] = str(ws)
        res = subprocess.run(
            ["bash", str(task.solution_script)],
            cwd=str(ws),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if res.returncode != 0:
            print(f"        solution.sh failed: {res.stdout}{res.stderr}"[:400])
        return res.returncode == 0
    return copied


# ---------------------------------------------------------------------- proto run


def proto_cmd(task: Task, ws: Path, tier: str, model: str | None, max_steps: int) -> list[str]:
    cmd = [
        "node",
        "--experimental-strip-types",
        "--disable-warning=ExperimentalWarning",
        str(REPO / "src" / "cli.ts"),
        "code",
        "--workspace",
        str(ws),
        "--yes",
        "--no-save",
        "--print",
        "--max-steps",
        str(max_steps),
    ]
    if tier == "local":
        cmd.append("--local")
    elif tier == "cloud":
        cmd += ["--no-route"]
    else:  # router decides
        cmd.append("--route")
    if model:
        cmd += ["--model", model]
    cmd.append(task.prompt)
    return cmd


@dataclass
class Outcome:
    task: str
    tier: str
    solved: bool
    seconds: float
    exit_code: int
    steps: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    verify_output: str = ""
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


STEP_RE = re.compile(r"steps (\d+) · tokens ([\d,]+)↓ ([\d,]+)↑ · cost \$([\d.]+)")
ERR_RE = re.compile(r"^\s*[✗x!]\s*(.+)$", re.MULTILINE)


def extract_stats(text: str) -> dict:
    """
    The agent prints a status line only in the interactive path; `--print` routes its
    chrome to stderr. Scrape both, and treat a miss as unknown rather than zero —
    an absent measurement and a measured zero are different things.
    """
    out: dict = {}
    m = STEP_RE.search(text)
    if m:
        out["steps"] = int(m.group(1))
        out["tokens_in"] = int(m.group(2).replace(",", ""))
        out["tokens_out"] = int(m.group(3).replace(",", ""))
        out["cost_usd"] = float(m.group(4))
    else:
        m2 = re.search(r"(\d+) steps", text)
        if m2:
            out["steps"] = int(m2.group(1))
    return out


def extract_error(text: str) -> str | None:
    """The agent's `--print` mode writes chrome to stderr; keep the last complaint."""
    hits = [h.strip() for h in ERR_RE.findall(text) if h.strip()]
    return hits[-1][:300] if hits else None


# ------------------------------------------------------------------------- modes


def cmd_list(tasks: list[Task]) -> int:
    print(f"{len(tasks)} task(s)\n")
    for t in tasks:
        tags = ",".join(t.tags) or "-"
        print(f"  {t.id:<22} {tags:<28} timeout {t.timeout_sec}s")
        first = t.prompt.split("\n")[0]
        print(f"    {first[:96]}")
    return 0


def cmd_validate(tasks: list[Task], run_root: Path) -> int:
    """
    A task is only a task if it fails before any work and passes after the reference
    solution. Anything else is measuring the verifier, not the agent. Both halves are
    checked because each catches a different bug: a verifier that always passes, and a
    solution that does not actually solve it.
    """
    print("validating suite — every task must FAIL on its initial state\n")
    venv, warning = build_venv(run_root)
    if warning:
        print(f"  ERROR: {warning}\n")
        return 2
    env = base_env(venv)
    run_dir = run_root / "validate"
    bad = 0
    for t in tasks:
        ws, _ = materialize(t, run_dir, run_root)
        ok_before, out_before = run_verifier(t, ws, env)
        if ok_before:
            print(f"  FAIL  {t.id:<22} verifier passes BEFORE any work")
            bad += 1
            continue
        if t.solution is None and t.solution_script is None:
            print(f"  ok    {t.id:<22} fails before (no reference solution to check)")
            continue
        if not apply_solution(t, ws):
            print(f"  FAIL  {t.id:<22} no usable reference solution")
            bad += 1
            continue
        ok_after, out_after = run_verifier(t, ws, env)
        if not ok_after:
            print(f"  FAIL  {t.id:<22} reference solution does not pass")
            print("        " + out_after.replace("\n", "\n        ")[:500])
            bad += 1
            continue
        print(f"  ok    {t.id:<22} fails before, passes after")
    print()
    if bad:
        print(f"{bad} task(s) are not honest — fix them before running the suite")
        return 1
    print(f"all {len(tasks)} task(s) are honest")
    return 0


def cmd_run(tasks: list[Task], args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    label = args.label or f"{args.tier}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = run_root / label
    run_dir.mkdir(parents=True, exist_ok=True)

    venv, venv_warning = build_venv(run_dir, quiet=args.no_venv)
    if venv_warning and not args.no_venv:
        # Refuse to run rather than produce a table of zeros that looks like a result.
        print(f"ERROR: {venv_warning}")
        return 2
    env = base_env(venv) if not args.no_venv else base_env(None)
    if args.model:
        env["PROTO_LOCAL_MODEL"] = args.model
    if args.tier == "local" and args.context:
        env["PROTO_LOCAL_CONTEXT"] = str(args.context)

    results: list[Outcome] = []
    print(f"suite    {args.suite}   ({len(tasks)} tasks)")
    print(f"tier     {args.tier}" + (f"  model {args.model}" if args.model else ""))
    print(f"run dir  {run_dir}\n")

    for i, t in enumerate(tasks, 1):
        ws, home = materialize(t, run_dir, run_root)
        env["PROTO_HOME"] = str(home)
        cmd = proto_cmd(t, ws, args.tier, args.model, args.max_steps)
        started = time.time()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO),
                env=env,
                capture_output=True,
                text=True,
                timeout=t.timeout_sec,
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            code = -1
        seconds = time.time() - started

        chrome = stderr + stdout
        if timed_out:
            solved, verify_out = False, f"agent timed out after {t.timeout_sec}s"
        else:
            solved, verify_out = run_verifier(t, ws, env)

        outcome = Outcome(
            task=t.id,
            tier=args.tier,
            solved=solved,
            seconds=round(seconds, 1),
            exit_code=code,
            error="timeout" if timed_out else extract_error(chrome),
            verify_output=verify_out[:600],
            **extract_stats(chrome),
        )
        # Keep the transcript. A benchmark that reports only pass/fail cannot answer
        # the first question anyone asks about a failure, which is "what did it do?".
        (run_dir / t.id).mkdir(parents=True, exist_ok=True)
        (run_dir / t.id / "agent.stdout").write_text(stdout)
        (run_dir / t.id / "agent.stderr").write_text(stderr)

        results.append(outcome)
        mark = "PASS" if solved else "fail"
        extra = ""
        if outcome.steps is not None:
            extra += f" {outcome.steps} steps"
        if outcome.tokens_out is not None:
            extra += f" {outcome.tokens_out} out-tok"
        print(f"  [{i:>2}/{len(tasks)}] {mark}  {t.id:<22} {outcome.seconds:>6.1f}s{extra}")
        if args.verbose and not solved:
            print("        " + verify_out.replace("\n", "\n        ")[:400])

    passed = sum(1 for r in results if r.solved)
    out_file = run_dir / "results.jsonl"
    with out_file.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r.to_json()) + "\n")

    total = sum(r.seconds for r in results)
    spend = sum(r.cost_usd or 0.0 for r in results)
    print(f"\n{passed}/{len(results)} solved   {total:.0f}s wall   ${spend:.4f}")
    print(f"results  {out_file}")
    if args.verbose:
        print_breakdown(results)
    return 0


def print_breakdown(results: list[Outcome]) -> None:
    from collections import Counter

    print("\n  by tag:")
    counts: Counter = Counter()
    solved: Counter = Counter()
    for r in results:
        counts[r.task] += 1
        if r.solved:
            solved[r.task] += 1
    for task in sorted(counts):
        print(f"    {task:<24} {'PASS' if solved[task] else 'fail'}")


def cmd_report(results_path: Path) -> int:
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    if not rows:
        print("no results")
        return 0
    passed = sum(1 for r in rows if r["solved"])
    print(f"{results_path}")
    print(f"  {passed}/{len(rows)} solved   tier {rows[0]['tier']}")
    total = sum(r["seconds"] for r in rows)
    print(f"  {total:.0f}s total   {total / len(rows):.1f}s mean/task")
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    if costs:
        print(f"  ${sum(costs):.4f} total   ${sum(costs) / len(costs):.4f} mean/task")
    print("\n  task                   result   seconds  steps  out-tok")
    for r in sorted(rows, key=lambda r: r["task"]):
        print(
            f"  {r['task']:<22} {'PASS' if r['solved'] else 'fail':<8} "
            f"{r['seconds']:>7.1f}  {str(r.get('steps') or '-'):>5}  {str(r.get('tokens_out') or '-'):>7}"
        )
    return 0


# -------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="agentic benchmark suite for proto code")
    ap.add_argument("mode", choices=["list", "validate", "run", "report"])
    ap.add_argument("--suite", default=DEFAULT_SUITE)
    ap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    ap.add_argument("--tier", default="local", choices=["local", "cloud", "route"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--only", action="append", default=None, help="run only this task id (repeatable)")
    ap.add_argument("--tag", default=None, help="run only tasks carrying this tag")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--context", type=int, default=None, help="local context window override")
    ap.add_argument("--label", default=None, help="name this run directory")
    ap.add_argument("--no-venv", action="store_true", help="do not build the shared virtualenv")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    if args.mode == "report":
        if not args.results:
            raise SystemExit("report needs --results PATH")
        return cmd_report(Path(args.results))

    tasks = load_suite(args.suite)
    if args.only:
        want = set(args.only)
        tasks = [t for t in tasks if t.id in want]
    if args.tag:
        tasks = [t for t in tasks if args.tag in t.tags]
    if not tasks:
        raise SystemExit("no tasks selected")

    if args.mode == "list":
        return cmd_list(tasks)
    if args.mode == "validate":
        return cmd_validate(tasks, Path(args.run_root))
    return cmd_run(tasks, args)


if __name__ == "__main__":
    sys.exit(main())
