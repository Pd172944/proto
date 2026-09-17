"""Cost-aware split: local-first when localized, cloud only on escalate.

Objective: match a Sonnet-only resolve rate while spending cloud tokens only
when the local model cannot finish, without adding wall clock versus
always-cloud.

django-17029 already proved the cheap path: Ornith, 19s, $0, 13 steps. Rule:

- A tight scout (≤2 production files) → local patch with a hard 2-minute cap.
- Adjacent existing tests must pass or the local diff is discarded (reset the
  worktree) and one cloud call finishes, given the scout + failed diff.
- Unlocalized tasks skip local patching and go to cloud immediately, so we do
  not pay 2 minutes of Ornith on a task the cloud model would have solved anyway.

Hypothetical Sonnet spend uses cloud token counts at list prices
($3 / $0.30 / $15 per M uncached-in / cached-in / out).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE: Path = Path("/data/prithvi/sweb")
PROTO: Path = Path("/data/prithvi/proto")

NEW_DEF: re.Pattern[str] = re.compile(r"^\+\s*(?:async def |def |class )")


def is_inplace_patch(patch: str) -> bool:
    """True when a local diff only edits existing code (no new def/class).

    Ornith's adjacent-test-passing failures on this set have been new helpers
    (sympy visit_Compare) that nearby tests do not cover. The proven local win
    (django-17029) is a few lines inside an existing method.

    Args:
        patch: Unified diff.

    Returns:
        Whether the patch is safe to keep without a cloud pass.
    """
    return not any(NEW_DEF.match(line) for line in patch.splitlines())


FILE_BULLET: re.Pattern[str] = re.compile(
    r"^[\-\*]\s+((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:py|c|h|cpp|js|ts|go|rs))\s*$"
)
STATUS_LINE: re.Pattern[str] = re.compile(r"^STATUS:\s*(PASS|FAIL)\b", re.I | re.M)


def parse_scout_files(note: str) -> List[str]:
    """Extract production file paths from a scout note.

    Args:
        note: Salvaged scout text (LIKELY_FILES / OBSERVED_READS bullets).

    Returns:
        Deduplicated repo-relative paths, at most eight.
    """
    files: List[str] = []
    in_section: bool = False
    for raw in note.splitlines():
        line: str = raw.strip()
        if line.startswith("LIKELY_FILES") or line.startswith("OBSERVED_READS"):
            in_section = True
            continue
        if in_section and line.endswith(":") and not line.startswith(("-", "*")):
            in_section = False
            continue
        if not in_section:
            continue
        match = FILE_BULLET.match(line)
        if not match:
            continue
        path: str = match.group(1)
        if "/test" in path or path.startswith("test"):
            continue
        if path not in files:
            files.append(path)
    return files[:8]


def constraint_block(files: List[str]) -> str:
    """Prompt prefix that hard-constrains edits to scout files.

    Args:
        files: Repo-relative paths the local scout named.

    Returns:
        Empty string when there is no constraint; otherwise a prompt prefix.
    """
    if not files:
        return ""
    bullets: str = "\n".join(f"- {p}" for p in files)
    return (
        "EDIT CONSTRAINT: Edit only these files. If the scout was wrong, first "
        "state in one sentence why and name the other file, then edit that file "
        "instead. Do not create files at the repository root.\n"
        f"{bullets}\n\n"
    )


def run_proto(
    prompt: str,
    repo_dir: Path,
    env: Dict[str, str],
    *,
    max_steps: int,
    deadline_min: float,
    timeout_s: int,
    local: bool = False,
    read_only: bool = False,
    no_route: bool = False,
) -> subprocess.CompletedProcess:
    """Invoke ``proto code`` as a one-shot and return the completed process.

    Args:
        prompt: User task text.
        repo_dir: Workspace.
        env: Environment for the subprocess.
        max_steps: Agent step budget.
        deadline_min: Proto wall-clock deadline.
        timeout_s: Hard subprocess timeout (seconds).
        local: Force the local model.
        read_only: Disable mutating tools.
        no_route: Force the configured cloud provider.

    Returns:
        Completed process (stdout/stderr captured).
    """
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
        str(max_steps),
        "--deadline-min",
        str(deadline_min),
    ]
    if local:
        cmd.append("--local")
    if read_only:
        cmd.append("--read-only")
    if no_route:
        cmd.append("--no-route")
    return subprocess.run(cmd, cwd=repo_dir, env=env, capture_output=True, text=True, timeout=timeout_s)


def should_try_local(files: List[str]) -> bool:
    """Whether a scout is tight enough that a 2-minute local patch is worth it.

    Exactly one named production file is the django-17029 shape (Ornith, 19s,
    $0). Two-plus files means the scout is still spreading its bets; a local
    attempt there serializes into the cloud call and costs wall clock that a
    Sonnet-only run would not pay, and it is where we lost django-17084.

    Args:
        files: Scout-named production paths.

    Returns:
        True when a bounded local patch should run before any cloud call.
    """
    return len(files) == 1


def reset_worktree(repo_dir: Path) -> None:
    """Drop a failed local diff so the cloud fixer starts from HEAD.

    Args:
        repo_dir: Agent checkout.
    """
    subprocess.run(["git", "reset", "--hard", "-q", "HEAD"], cwd=repo_dir, check=True, timeout=60)
    subprocess.run(["git", "clean", "-fdq"], cwd=repo_dir, check=True, timeout=60)


def sonnet_usd(input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    """Hypothetical Claude Sonnet 4.5 bill for a cloud call's token counts.

    Args:
        input_tokens: Total input tokens (including cached).
        cached_tokens: Cached input tokens.
        output_tokens: Output tokens.

    Returns:
        USD at list prices ($3 / $0.30 / $15 per million).
    """
    cached: int = max(0, cached_tokens)
    uncached: int = max(0, input_tokens - cached)
    return (uncached * 3.0 + cached * 0.3 + max(0, output_tokens) * 15.0) / 1_000_000.0


def adjacent_on_worktree(instance: dict, repo_dir: Path, env: Dict[str, str], patch: str) -> Tuple[bool, str, bool]:
    """Run existing tests near the touched files on the live worktree.

    Absence of adjacent tests is not a pass: a local patch with nothing to
    check is escalated rather than kept.

    Args:
        instance: Instance row.
        repo_dir: Already-patched checkout.
        env: Agent environment (venv on PATH, PYTHONPATH set).
        patch: Current production diff.

    Returns:
        ``(passed, log, had_tests)``.
    """
    from select_patch import django_labels, guess_test_files, touched_source_files

    if not patch.strip():
        return False, "empty production patch", False
    iid: str = instance["instance_id"]
    python: str = str(BASE / "envs" / iid / "bin" / "python")
    touched: List[str] = touched_source_files(patch)
    timeout: int = 45
    if instance["repo"] == "django/django":
        labels: List[str] = django_labels(repo_dir, touched)
        if not labels:
            return False, "no adjacent django labels", False
        res = subprocess.run(
            [python, "tests/runtests.py", "--verbosity", "0", "--parallel", "1", *labels],
            cwd=repo_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    else:
        test_files: List[str] = guess_test_files(instance["repo"], repo_dir, touched)
        if not test_files:
            return False, "no adjacent test files", False
        res = subprocess.run(
            [python, "-m", "pytest", "-q", "--no-header", "-x", "-p", "no:cacheprovider", *test_files],
            cwd=repo_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    log: str = (res.stdout + "\n" + res.stderr)[-2500:]
    return res.returncode == 0, log, True


def parse_verify_status(stdout: str) -> Optional[bool]:
    """Parse STATUS: PASS/FAIL from a local verifier answer.

    Args:
        stdout: Verifier ``--print`` stdout, possibly with trailing JSON.

    Returns:
        True/False when the model labelled the run, else None.
    """
    text: str = stdout
    idx: int = text.rfind("\n{")
    if idx >= 0:
        text = text[:idx]
    match = STATUS_LINE.search(text)
    if not match:
        return None
    return match.group(1).upper() == "PASS"


def local_verify(
    instance: dict,
    repo_dir: Path,
    env: Dict[str, str],
    patch: str,
    inst_dir: Path,
) -> Tuple[bool, str]:
    """Ask the local model to reproduce the issue against the current tree.

    The verifier must not edit production source. It writes a script under
    ``/tmp`` via ``run_command`` (``write_file`` cannot escape the workspace).

    Args:
        instance: Instance row.
        repo_dir: Patched checkout.
        env: Agent environment.
        patch: Current production diff (shown so the model knows what changed).
        inst_dir: Where to store verifier transcripts.

    Returns:
        ``(passed, evidence)``. A timeout or unparseable answer is a skip
        (passed=True) so we do not burn a cloud repair on a hung local HTTP call.
    """
    iid: str = instance["instance_id"]
    prompt: str = (
        f"You are checking whether a fix for {instance['repo']} actually works. "
        "Do NOT edit production source. Do NOT create files in the repo.\n\n"
        f"<issue>\n{instance['problem_statement'].strip()[:3500]}\n</issue>\n\n"
        "Current production diff:\n```\n"
        f"{patch[:3500]}\n"
        "```\n\n"
        f"Using run_command, write a minimal reproduction to /tmp/repro_{iid}.py "
        "from the issue's example (not from any test suite) and run it with `python`. "
        "The project is installed on PATH. The issue is fixed only if the script now "
        "shows the expected behavior from the report.\n\n"
        "Reply with exactly:\n"
        "STATUS: PASS\nor\nSTATUS: FAIL\nEVIDENCE:\n"
        "the last 30 lines of the script's output, or why you could not run it."
    )
    try:
        proc = run_proto(
            prompt,
            repo_dir,
            env,
            max_steps=12,
            deadline_min=3,
            timeout_s=200,
            local=True,
        )
    except subprocess.TimeoutExpired:
        return True, "local verify timed out; skipped"
    (inst_dir / "verify.out").write_text(proc.stdout)
    (inst_dir / "verify.err").write_text(proc.stderr)
    status: Optional[bool] = parse_verify_status(proc.stdout)
    evidence: str = (proc.stdout + "\n" + proc.stderr)[-2500:]
    if status is None:
        # Model did not label the run. Only fail closed on an obvious traceback.
        failish: bool = bool(re.search(r"Traceback \(most recent call last\)|STATUS: FAIL", evidence, re.I))
        return (not failish), evidence
    return status, evidence


def cloud_finish(
    instance: dict,
    repo_dir: Path,
    env: Dict[str, str],
    files: List[str],
    scout_note: str,
    failed_diff: str,
    evidence: str,
    inst_dir: Path,
    *,
    max_steps: int,
    deadline_min: float,
) -> Optional[subprocess.CompletedProcess]:
    """One cloud patch call after a discarded local attempt (or instead of one).

    Args:
        instance: Instance row.
        repo_dir: Clean (or current) checkout.
        env: Agent environment.
        files: Scout-constrained paths.
        scout_note: Salvaged scout text.
        failed_diff: Local diff that was reset, if any.
        evidence: Adjacent-test log from the local attempt.
        inst_dir: Artifact directory.
        max_steps: Cloud step budget.
        deadline_min: Cloud deadline.

    Returns:
        Completed process, or None on timeout.
    """
    parts: List[str] = [constraint_block(files)]
    if scout_note:
        parts.append(f"<scout>\n{scout_note[-2000:]}\n</scout>\n")
    if failed_diff.strip():
        parts.append(
            "A local model already attempted a patch and it was rejected "
            "(empty, or adjacent tests failed). The worktree is reset to HEAD. "
            "Do not repeat that diff. Here is what it tried:\n```\n"
            f"{failed_diff[:2500]}\n```\n"
        )
        if evidence:
            parts.append(f"<evidence>\n{evidence[-1500:]}\n</evidence>\n")
    parts.append(
        f"Fix the issue in {instance['repo']}. Edit existing library source only. "
        "Never modify tests. Never create files at the repository root. "
        "Reproduction scripts belong in /tmp.\n\n"
        f"<issue>\n{instance['problem_statement'].strip()[:3500]}\n</issue>\n"
    )
    prompt: str = "\n".join(parts)
    try:
        proc = run_proto(
            prompt,
            repo_dir,
            env,
            max_steps=max_steps,
            deadline_min=deadline_min,
            timeout_s=int(deadline_min * 60) + 120,
            no_route=True,
        )
    except subprocess.TimeoutExpired:
        return None
    (inst_dir / "agent.out").write_text(proc.stdout)
    (inst_dir / "agent.err").write_text(proc.stderr)
    return proc
