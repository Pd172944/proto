import os
import pathlib

WS = pathlib.Path(os.environ["BENCH_WORKSPACE"])

EXPECTED = [
    "east: 27.50",
    "north: 124.99",
    "south: 35.00",
    "west: 0.00",
]


def _lines():
    p = WS / "report.txt"
    assert p.is_file(), "report.txt was not created"
    return [l for l in p.read_text().splitlines() if l.strip()]


def test_regions_in_alphabetical_order_with_correct_totals():
    assert _lines() == EXPECTED, "got:\n" + "\n".join(_lines())


def test_two_decimal_places_everywhere():
    for line in _lines():
        amount = line.split(": ")[1]
        assert len(amount.split(".")[1]) == 2, "not two decimal places: %r" % line
