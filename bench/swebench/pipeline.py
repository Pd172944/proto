"""Subtask split: local localize/verify, cloud patch/repair.

The SWE-bench runner keeps one model in charge of each phase so GPU 7 does
the cheap I/O and the cloud model only spends tokens on the edit:

1. **Localize** (Ornith, read-only) already lives in ``run.scout_localize``.
2. **Patch** (cloud) is the main ``proto code`` turn, constrained to scout files.
3. **Check** (this module): adjacent existing tests plus a local-model repro.
4. **Repair** (cloud, only on failure): one more edit turn with the traceback.

Local never proposes a competing patch. A failed local check discards nothing
that already landed; it only asks cloud to revise the worktree in place.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE: Path = Path("/data/prithvi/sweb")
PROTO: Path = Path("/data/prithvi/proto")

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


def adjacent_on_worktree(instance: dict, repo_dir: Path, env: Dict[str, str], patch: str) -> Tuple[bool, str]:
    """Run existing tests near the touched files on the live worktree.

    This is the check a real user has: did we break nearby tests? Hidden
    FAIL_TO_PASS tests are never used.

    Args:
        instance: Instance row.
        repo_dir: Already-patched checkout.
        env: Agent environment (venv on PATH, PYTHONPATH set).
        patch: Current production diff.

    Returns:
        ``(passed, log)``. No adjacent tests is treated as a pass.
    """
    from select_patch import django_labels, guess_test_files, touched_source_files

    if not patch.strip():
        return False, "empty production patch"
    iid: str = instance["instance_id"]
    python: str = str(BASE / "envs" / iid / "bin" / "python")
    touched: List[str] = touched_source_files(patch)
    timeout: int = 60
    if instance["repo"] == "django/django":
        labels: List[str] = django_labels(repo_dir, touched)
        if not labels:
            return True, "no adjacent django labels"
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
            return True, "no adjacent test files"
        res = subprocess.run(
            [python, "-m", "pytest", "-q", "--no-header", "-x", "-p", "no:cacheprovider", *test_files],
            cwd=repo_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    log: str = (res.stdout + "\n" + res.stderr)[-2500:]
    return res.returncode == 0, log


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


def cloud_repair(
    instance: dict,
    repo_dir: Path,
    env: Dict[str, str],
    files: List[str],
    evidence: str,
    inst_dir: Path,
) -> None:
    """One cloud turn to revise the worktree after a failed local check.

    Args:
        instance: Instance row.
        repo_dir: Already-patched checkout to revise in place.
        env: Agent environment.
        files: Scout-constrained paths.
        evidence: Adjacent-test and/or repro output.
        inst_dir: Where to store the repair transcript.
    """
    prompt: str = (
        constraint_block(files)
        + "A local check of the current tree says the issue is still present or "
        "adjacent tests failed. Do not rewrite the whole patch from scratch. "
        "Read the failing evidence, then edit_file on existing library source so "
        "the reported behavior holds. Never modify tests. Never create files at "
        "the repository root.\n\n"
        f"<issue>\n{instance['problem_statement'].strip()[:2500]}\n</issue>\n\n"
        f"<evidence>\n{evidence[-2000:]}\n</evidence>\n"
    )
    try:
        proc = run_proto(
            prompt,
            repo_dir,
            env,
            max_steps=25,
            deadline_min=5,
            timeout_s=360,
            no_route=True,
        )
    except subprocess.TimeoutExpired:
        return
    (inst_dir / "repair.out").write_text(proc.stdout)
    (inst_dir / "repair.err").write_text(proc.stderr)
