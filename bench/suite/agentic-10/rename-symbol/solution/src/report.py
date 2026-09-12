"""Human-readable summaries built on top of math_utils."""

from math_utils import compute_total


def summarize(items):
    total = compute_total(items)
    return {"count": len(items), "total": total}


def format_summary(items):
    return "items=%d total=%d" % (len(items), compute_total(items))
