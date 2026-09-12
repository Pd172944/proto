"""Command-line front end for the reporting tools."""

from math_utils import compute_total


def run(items):
    """Return the line the CLI prints for ``items``."""
    return "total=%d" % compute_total(items)


def main(items=None):
    print(run(items or []))
    return 0
