from prices import total, apply_discount

import pytest


def test_total_adds_tax():
    assert total([10.0, 5.0], 0.10) == pytest.approx(16.5)


def test_total_no_tax():
    assert total([2.0, 3.0], 0.0) == 5.0


def test_total_empty():
    assert total([], 0.5) == 0.0


def test_discount_is_a_fraction():
    assert apply_discount(200.0, 0.25) == pytest.approx(150.0)


def test_discount_zero():
    assert apply_discount(80.0, 0.0) == pytest.approx(80.0)
