`prices.py` has two bugs.

1. `total(prices, tax_rate)` sums the prices but ignores `tax_rate`, which is a fraction (0.08 means 8%).
2. `apply_discount(amount, pct)` treats `pct` as an absolute amount to subtract instead of a fraction of `amount`.

Fix both functions so `test_prices.py` passes. Run the tests to confirm.
