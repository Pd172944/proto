import os
import sys

sys.path.insert(0, os.environ["BENCH_WORKSPACE"])
sys.path.insert(0, os.path.join(os.environ["BENCH_WORKSPACE"], "src"))

import cli  # noqa: E402
import math_utils  # noqa: E402
import report  # noqa: E402

ITEMS = [{"amount": 10}, {"amount": 5}, {"amount": 2}]


def test_compute_total_is_importable():
    from math_utils import compute_total

    assert compute_total(ITEMS) == 17


def test_compute_total_handles_empty_input():
    from math_utils import compute_total

    assert compute_total([]) == 0


def test_old_name_is_gone():
    assert not hasattr(math_utils, "calc_total")


def test_report_still_works():
    assert report.summarize(ITEMS) == {"count": 3, "total": 17}
    assert report.format_summary(ITEMS) == "items=3 total=17"


def test_cli_still_works():
    assert cli.run(ITEMS) == "total=17"
