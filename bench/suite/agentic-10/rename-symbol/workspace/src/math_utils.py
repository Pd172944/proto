"""Numeric helpers shared by the reporting tools."""


def calc_total(items):
    """Return the sum of the ``amount`` field of every item."""
    return sum(item["amount"] for item in items)
