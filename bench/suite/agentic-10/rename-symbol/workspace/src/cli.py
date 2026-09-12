"""Command-line front end for the reporting tools."""

from math_utils import calc_total


def run(items):
    """Return the line the CLI prints for ``items``."""
    return "total=%d" % calc_total(items)


def main(items=None):
    print(run(items or []))
    return 0
