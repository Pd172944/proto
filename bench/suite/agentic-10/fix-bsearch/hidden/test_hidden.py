import os
import sys

sys.path.insert(0, os.environ["BENCH_WORKSPACE"])

from search import bsearch


def test_every_position_in_a_small_list():
    items = [2, 4, 6, 8, 10, 12, 14]
    for i, value in enumerate(items):
        assert bsearch(items, value) == i, "wrong index for %r" % value


def test_absent_values_do_not_loop():
    items = [1, 3, 5, 7, 9]
    for missing in (0, 2, 4, 6, 8, 10, 100, -5):
        assert bsearch(items, missing) == -1


def test_single_element():
    assert bsearch([42], 42) == 0
    assert bsearch([42], 41) == -1


def test_duplicates_return_a_valid_index():
    items = [1, 1, 1, 1]
    assert bsearch(items, 1) in (0, 1, 2, 3)


def test_large_list():
    items = list(range(0, 2000, 2))
    assert bsearch(items, 1000) == 500
    assert bsearch(items, 1001) == -1
