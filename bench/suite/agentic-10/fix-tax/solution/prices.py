def total(prices, tax_rate):
    """Sum prices, then add tax. tax_rate is a fraction: 0.08 means 8%."""
    subtotal = 0
    for p in prices:
        subtotal += p
    return subtotal * (1 + tax_rate)


def apply_discount(amount, pct):
    """Reduce amount by pct, a fraction: 0.25 means 25% off."""
    return amount - amount * pct
