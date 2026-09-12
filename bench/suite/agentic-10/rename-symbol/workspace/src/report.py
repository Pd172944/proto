"""Human-readable summaries built on top of math_utils."""

from math_utils import calc_total


def summarize(items):
    total = calc_total(items)
    return {"count": len(items), "total": total}


def format_summary(items):
    return "items=%d total=%d" % (len(items), calc_total(items))
