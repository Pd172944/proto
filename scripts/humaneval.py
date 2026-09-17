#!/usr/bin/env python3
"""
HumanEval playground and runner for `proto code`.

HumanEval is 164 single-function problems: you get a signature and a docstring, you
write the body, and hidden assertions decide whether you were right. It is 45 KB
gzipped and clones to about 270 KB, so size is a non-issue.

**Read this before drawing conclusions from the numbers.** HumanEval is close to the
worst possible fit for an *agentic* harness, and a good fit for a plain completion
model. It has:

  - no repository to explore  (nothing to search, nothing to read)
  - no tests the agent can run (the assertions are deliberately hidden)
  - one file, one function     (no multi-file work, no refactors)

So it exercises almost none of what makes an agent useful, and it *does* penalise one
thing agents do: several turns where a single completion would have sufficed. Treat
`run` as a smoke test that the whole loop works end to end on real tasks, not as a
verdict on the harness. For that you want tasks with a repo, a failing test, and
freedom to iterate — this project's own `proto code` on your own codebase is a better
eval than HumanEval will ever be here.

Two honest notes about what `run` measures:

  - The prompt tells the agent it cannot see the tests, but nothing stops it writing
    its own quick check. That is legitimate agentic behaviour and it is an advantage
    over a bare completion — so this is "agentic pass@1", not model pass@1.
  - Model-generated Python is executed on your machine to check the answer. That is
    the same trust you already extend by letting the agent run commands.

Usage:
    python3 scripts/humaneval.py setup [--dir DIR] [--force]
    python3 scripts/humaneval.py list [--dir DIR] [--filter TEXT] [--limit N]
    python3 scripts/humaneval.py show <n> [--dir DIR]        # prepare one problem to play with
    python3 scripts/humaneval.py check <n> [--dir DIR]       # run the hidden tests
    python3 scripts/humaneval.py reset <n> [--dir DIR]
    python3 scripts/humaneval.py run [--count N] [--agent cloud|local|demo]
                                     [--model M] [--provider P] [--temperature T]
                                     [--start N] [--dir DIR] [--verbose]

Typical first session:
    python3 scripts/humaneval.py setup
    python3 scripts/humaneval.py show 0
    #   ... follow the printed instructions, playing with the CLI ...
    python3 scripts/humaneval.py check 0
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "src" / "cli.ts"
NODE = shutil.which("node") or "node"
UPSTREAM = "https://github.com/openai/human-eval.git"
RAW_JSONL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


# ----------------------------------------------------------------------------- io


def default_dir() -> Path:
    return REPO / "humaneval"


def problems_path(base: Path) -> Path:
    return base / "data" / "HumanEval.jsonl"


def load_problems(base: Path) -> list[dict]:
    path = problems_path(base)
    if not path.exists():
        sys.exit(f"no dataset at {path}\nrun: python3 scripts/humaneval.py setup")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def number_of(task_id: str) -> int:
    return int(task_id.split("/")[-1])


def work_dir(base: Path, n: int) -> Path:
    return base / "work" / f"{n:03d}"


def solution_path(base: Path, n: int) -> Path:
    return work_dir(base, n) / "solution.py"


def problem_by_number(problems: list[dict], n: int) -> dict:
    for p in problems:
        if number_of(p["task_id"]) == n:
            return p
    sys.exit(f"no problem numbered {n} (valid: 0-{len(problems) - 1})")


# ------------------------------------------------------------------------- setup


def cmd_setup(args: argparse.Namespace) -> int:
    base: Path = args.dir
    base.mkdir(parents=True, exist_ok=True)
    data = problems_path(base)

    if data.exists() and not args.force:
        print(f"dataset already present: {data} ({len(load_problems(base))} problems)")
        print("re-run with --force to refresh it")
        return 0

    data.parent.mkdir(parents=True, exist_ok=True)
    upstream = base / "upstream"

    if shutil.which("git") and not upstream.exists():
        print(f"cloning {UPSTREAM} …")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "-q", UPSTREAM, str(upstream)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"clone failed ({result.stderr.strip()[:120]}); falling back to the raw dataset")

    source_gz = upstream / "data" / "HumanEval.jsonl.gz"
    if source_gz.exists():
        data.write_bytes(gzip.decompress(source_gz.read_bytes()))
    else:
        print(f"downloading {RAW_JSONL} …")
        import urllib.request

        with urllib.request.urlopen(RAW_JSONL, timeout=60) as response:
            data.write_bytes(gzip.decompress(response.read()))

    problems = load_problems(base)
    size_kb = data.stat().st_size / 1024
    print(f"ready: {len(problems)} problems at {data} ({size_kb:.0f} KB)")
    print()
    print("next:")
    print("  python3 scripts/humaneval.py show 0        # prepare problem 0 and get instructions")
    return 0


# -------------------------------------------------------------------------- list


def cmd_list(args: argparse.Namespace) -> int:
    problems = load_problems(args.dir)
    rows = problems
    if args.filter:
        needle = args.filter.lower()
        rows = [p for p in rows if needle in p["entry_point"].lower() or needle in p["prompt"].lower()]
    for p in rows[: args.limit]:
        first_line = next((l for l in p["prompt"].splitlines() if l.strip().startswith("def ")), "")
        print(f"{number_of(p['task_id']):>4}  {p['entry_point']:<34} {first_line.strip()[:60]}")
    if len(rows) > args.limit:
        print(f"… {len(rows) - args.limit} more (raise --limit)")
    return 0


# -------------------------------------------------------------------------- show


def cmd_show(args: argparse.Namespace) -> int:
    problems = load_problems(args.dir)
    p = problem_by_number(problems, args.number)
    target = solution_path(args.dir, args.number)
    target.parent.mkdir(parents=True, exist_ok=True)

    # The HumanEval prompt ends at the docstring, which leaves the file syntactically
    # invalid. Adding `pass` keeps it parseable so read/edit behave normally from the
    # start, and an untouched file fails the tests rather than erroring on import.
    stub = p["prompt"] + "    pass\n"
    target.write_text(stub)

    rel = target.relative_to(REPO) if str(target).startswith(str(REPO)) else target
    print(f"problem {args.number}: {p['entry_point']}")
    print(f"workspace: {target.parent}")
    print(f"file:      {rel}")
    print()
    print("=" * 78)
    print(p["prompt"].rstrip())
    print("=" * 78)
    print()
    print("Play with it:")
    print(f"  cd {target.parent}")
    print(f"  {REPO / 'bin' / 'proto-code'}")
    print()
    print("Then paste this as your first message:")
    print()
    print(textwrap.indent(task_prompt(p), "    "))
    print()
    print("When you are done (or want to give up), check the hidden tests with:")
    print(f"  python3 scripts/humaneval.py check {args.number}")
    print()
    print("To start over:")
    print(f"  python3 scripts/humaneval.py reset {args.number}")
    return 0


def task_prompt(p: dict) -> str:
    return (
        f"Complete the function `{p['entry_point']}` in solution.py so that it satisfies its "
        f"docstring and the examples in it.\n\n"
        "The file currently contains the signature, the docstring and an empty body. "
        "Implement the body, then check that the file imports cleanly "
        "(for example `python3 -c \"import solution\"`).\n\n"
        "The official test suite is not in this directory and you cannot see it, so make the "
        "implementation correct in general rather than special-casing the docstring examples. "
        "Keep it to the standard library plus whatever solution.py already imports."
    )


# ------------------------------------------------------------------------- check


def build_checker(base: Path, p: dict) -> str:
    """A runner that imports the candidate and executes HumanEval's hidden tests."""
    sol = solution_path(base, number_of(p["task_id"]))
    return (
        "import importlib.util, sys, traceback\n"
        f"spec = importlib.util.spec_from_file_location('candidate', {str(sol)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "try:\n"
        "    spec.loader.exec_module(module)\n"
        "except Exception:\n"
        "    traceback.print_exc()\n"
        "    print('RESULT: FAIL (solution.py did not import)')\n"
        "    sys.exit(0)\n"
        f"candidate = getattr(module, {p['entry_point']!r}, None)\n"
        "if candidate is None:\n"
        f"    print('RESULT: FAIL (no function named {p['entry_point']})')\n"
        "    sys.exit(0)\n"
        "# ---- hidden HumanEval tests, verbatim ----\n"
        f"{p['test']}\n"
        "# ---- run them ----\n"
        "try:\n"
        "    check(candidate)\n"
        "    print('RESULT: PASS')\n"
        "except Exception:\n"
        "    traceback.print_exc()\n"
        "    print('RESULT: FAIL')\n"
    )


def run_check(base: Path, p: dict, *, quiet: bool = False) -> tuple[bool, str]:
    """Returns (passed, output)."""
    if not solution_path(base, number_of(p["task_id"])).exists():
        return False, "no solution.py (run `show` first)"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(build_checker(base, p))
        runner = handle.name
    try:
        result = subprocess.run(
            [sys.executable, runner],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(work_dir(base, number_of(p["task_id"]))),
        )
        output = (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        output = "timed out after 60s (probably an infinite loop)"
    finally:
        os.unlink(runner)
    passed = "RESULT: PASS" in output
    if not quiet:
        print(output)
    return passed, output


def cmd_check(args: argparse.Namespace) -> int:
    problems = load_problems(args.dir)
    p = problem_by_number(problems, args.number)
    passed, _ = run_check(args.dir, p)
    print()
    print(f"problem {args.number} ({p['entry_point']}): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def cmd_reset(args: argparse.Namespace) -> int:
    problems = load_problems(args.dir)
    p = problem_by_number(problems, args.number)
    target = solution_path(args.dir, args.number)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(p["prompt"] + "    pass\n")
    print(f"reset {target}")
    return 0


# --------------------------------------------------------------------------- run


def extract_agent_error(stdout: str, stderr: str) -> str:
    """Best-effort: the reason the agent failed, for the report's last column.

    Prefers the `error` field of the CLI's --json result, since that is designed to be
    read by a machine; falls back to scanning both streams for the error line the CLI
    prints for a human.
    """
    for blob in (stdout, stderr):
        for line in reversed(blob.splitlines()):
            stripped = line.strip()
            if stripped.startswith("{") and '"error"' in stripped:
                try:
                    payload = json.loads(stripped)
                    if isinstance(payload, dict) and payload.get("error"):
                        return str(payload["error"])[:90]
                except Exception:
                    pass
    # The JSON result may be pretty-printed across many lines; find it whole.
    start = stdout.find("{")
    if start != -1:
        try:
            payload = json.loads(stdout[start:])
            if isinstance(payload, dict) and payload.get("error"):
                return str(payload["error"])[:90]
        except Exception:
            pass
    for blob in (stderr, stdout):
        for line in reversed([l.strip() for l in blob.splitlines() if l.strip()]):
            if line.startswith(("{", "}", '"', "[")):
                continue
            if "✗" in line or "Error" in line or "error" in line or "HTTP" in line:
                return line.lstrip("✗ !").strip()[:90]
    return ""


def read_session_stats(proto_home: Path) -> dict:
    """Pull token/cost totals out of the session the CLI saved."""
    sessions = proto_home / "sessions"
    if not sessions.exists():
        return {}
    files = sorted(sessions.glob("*.json"))
    if not files:
        return {}
    try:
        record = json.loads(files[-1].read_text())
    except Exception:
        return {}
    stats = record.get("stats") or {}
    return {
        "turns": stats.get("turns", 0),
        "cost": stats.get("costUsd", 0.0),
        "tokens_in": stats.get("inputTokens", 0),
        "tokens_out": stats.get("outputTokens", 0),
        "files": stats.get("filesEdited", []),
    }


def cmd_run(args: argparse.Namespace) -> int:
    problems = load_problems(args.dir)
    numbers = [number_of(p["task_id"]) for p in problems]
    numbers = [n for n in numbers if n >= args.start][: args.count]
    if not numbers:
        sys.exit("no problems selected")

    print(f"running {len(numbers)} problem(s) with agent={args.agent}", end="")
    if args.model:
        print(f" model={args.model}", end="")
    print()
    print("=" * 92)

    passed_count = 0
    rows: list[tuple[int, str, bool, float, str]] = []
    total_cost = 0.0

    for n in numbers:
        p = problem_by_number(problems, n)
        target = work_dir(args.dir, n)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        # A per-problem PROTO_HOME keeps each run's session isolated, so
        # a HumanEval run never pollutes the real var/ directory.
        proto_home = target / ".proto"
        (target / "solution.py").write_text(p["prompt"] + "    pass\n")

        command = [
            NODE,
            "--experimental-strip-types",
            "--disable-warning=ExperimentalWarning",
            str(CLI),
            "code",
            task_prompt(p),
            "--workspace",
            str(target),
            "--yes",
            "--temperature",
            str(args.temperature),
            "--json",
        ]
        if args.model:
            command += ["--model", args.model]
        if args.provider:
            command += ["--provider", args.provider]
        if args.agent == "local":
            command += ["--local"]
        elif args.agent == "demo":
            command += ["--demo"]

        env = dict(os.environ)
        env["PROTO_HOME"] = str(proto_home)
        env["PROTO_LOG"] = "silent"

        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, cwd=str(target), env=env)
            # When the agent itself failed, its reason is the actionable information —
            # a failing assertion afterwards is a *consequence*, not the cause. Pull the
            # last stderr line, which is where the CLI reports provider and config errors.
            cli_note = ""
            if result.returncode != 0:
                reason = extract_agent_error(result.stdout, result.stderr)
                cli_note = f"agent exit {result.returncode}{': ' + reason if reason else ''}"
        except subprocess.TimeoutExpired:
            cli_note = f"agent timed out after {args.timeout}s (raise --timeout)"

        passed, output = run_check(args.dir, p, quiet=True)
        stats = read_session_stats(proto_home)
        cost = float(stats.get("cost", 0.0) or 0.0)
        total_cost += cost
        if passed:
            passed_count += 1

        # The agent's own failure wins the column; otherwise show the first assertion
        # that failed, which is the useful signal when the model simply got it wrong.
        first_failure = ""
        if not passed:
            for line in output.splitlines():
                if "Error" in line or "assert" in line:
                    first_failure = line.strip()[:70]
                    break
        rows.append((n, p["entry_point"], passed, cost, cli_note or first_failure))

        mark = "PASS" if passed else "FAIL"
        print(f"{mark}  {n:>4}  {p['entry_point']:<32} ${cost:.4f}  {rows[-1][4]}")
        if args.verbose and not passed:
            print(textwrap.indent(output[-1500:], "      "))

    print("=" * 92)
    rate = passed_count / len(numbers)
    print(f"pass@1: {passed_count}/{len(numbers)} = {rate:.1%}   estimated spend: ${total_cost:.4f}")
    print()
    print("what this does and does not tell you:")
    print("  - it is a smoke test that the agent loop, tools and provider all work on real tasks")
    print("  - HumanEval has no repo, no searchable code and no visible tests, so it exercises")
    print("    almost none of what makes an agent useful. A single-shot completion model will")
    print("    often score higher on it than an agent that spends extra turns reasoning.")
    print("  - the agent may write its own checks, so this is agentic pass@1, not model pass@1")
    print("  - an unpinned temperature and one sample per problem makes this noisy")
    return 0


# -------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HumanEval playground and runner for proto code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1].strip(),
    )
    parser.add_argument("--dir", type=Path, default=default_dir(), help="where to keep the dataset and work dirs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="clone/extract the dataset")
    p_setup.add_argument("--force", action="store_true", help="re-extract even if present")
    p_setup.set_defaults(func=cmd_setup)

    p_list = sub.add_parser("list", help="list problems")
    p_list.add_argument("--filter", type=str, default="", help="substring match on the name or prompt")
    p_list.add_argument("--limit", type=int, default=25)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="prepare one problem and print instructions")
    p_show.add_argument("number", type=int)
    p_show.set_defaults(func=cmd_show)

    p_check = sub.add_parser("check", help="run the hidden tests against your solution")
    p_check.add_argument("number", type=int)
    p_check.set_defaults(func=cmd_check)

    p_reset = sub.add_parser("reset", help="restore the original stub")
    p_reset.add_argument("number", type=int)
    p_reset.set_defaults(func=cmd_reset)

    p_run = sub.add_parser("run", help="run the agent over a range of problems and score pass@1")
    p_run.add_argument("--count", type=int, default=5)
    p_run.add_argument("--start", type=int, default=0)
    p_run.add_argument("--agent", choices=["cloud", "local", "demo"], default="cloud")
    p_run.add_argument("--model", type=str, default=None)
    p_run.add_argument("--provider", type=str, default=None)
    p_run.add_argument("--temperature", type=float, default=0.2)
    p_run.add_argument("--timeout", type=int, default=600, help="per-problem wall clock, seconds")
    p_run.add_argument("--verbose", action="store_true")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
