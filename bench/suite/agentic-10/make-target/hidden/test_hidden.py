import os
import re
import subprocess
import sys

sys.path.insert(0, os.environ["BENCH_WORKSPACE"])

WORKSPACE = os.environ["BENCH_WORKSPACE"]


def _make(*args):
    return subprocess.run(
        ["make", *args],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )


def _assert_tests_ran(result):
    output = result.stdout + result.stderr
    assert result.returncode == 0, "make failed:\n" + output
    ran = "test_calc" in output or re.search(r"\d+ passed", output)
    assert ran, "make did not appear to run the test suite:\n" + output


def test_make_test_target_runs_the_suite():
    _assert_tests_ran(_make("test"))


def test_bare_make_runs_the_suite():
    _assert_tests_ran(_make())
