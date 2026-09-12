import os
import sys

sys.path.insert(0, os.environ["BENCH_WORKSPACE"])

import pytest

from prices import apply_discount, total


def test_tax_is_applied_as_a_fraction():
    assert total([100.0], 0.20) == pytest.approx(120.0)
    assert total([50.0, 50.0], 0.10) == pytest.approx(110.0)


def test_zero_tax_is_a_no_op():
    assert total([7.5, 2.5], 0.0) == pytest.approx(10.0)


def test_empty_input():
    assert total([], 0.5) == pytest.approx(0.0)


def test_discount_scales_with_amount():
    assert apply_discount(200.0, 0.25) == pytest.approx(150.0)
    assert apply_discount(50.0, 0.10) == pytest.approx(45.0)


def test_full_discount():
    assert apply_discount(30.0, 1.0) == pytest.approx(0.0)
