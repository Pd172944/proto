import os
import pathlib

WS = pathlib.Path(os.environ["BENCH_WORKSPACE"])

EXPECTED = [
    "2024-03-11T09:14:02Z ERROR db  connection refused",
    "2024-03-11T09:14:05Z ERROR http GET /orders 500 upstream timeout",
    "2024-03-11T09:14:07Z ERROR db  deadlock detected on orders",
]


def test_errors_file_exists():
    assert (WS / "errors.txt").is_file(), "errors.txt was not created in the project root"


def test_contains_exactly_the_error_lines():
    lines = [l for l in (WS / "errors.txt").read_text().splitlines() if l.strip()]
    assert lines == EXPECTED, "expected exactly the three ERROR lines in order, got:\n" + "\n".join(lines)


def test_no_non_error_lines_leaked_in():
    text = (WS / "errors.txt").read_text()
    for bad in ("INFO", "WARN", "DEBUG"):
        assert bad not in text, "%s line leaked into errors.txt" % bad
